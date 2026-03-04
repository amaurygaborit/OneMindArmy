#pragma once
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <memory>
#include <cmath>

#include "TreeSearch.hpp"
#include "NeuralNet.cuh"
#include "../bootstrap/GameConfig.hpp"
#include "../util/AlignedVec.hpp"

namespace Core
{
    // ============================================================================
    // BLOCKING QUEUE ROBUSTE
    // ============================================================================
    template<typename T>
    class BlockingQueue
    {
        AlignedVec<T> m_buffer;
        size_t m_capacity;
        size_t m_head = 0;
        size_t m_tail = 0;
        size_t m_count = 0;

        std::mutex m_mutex;
        std::condition_variable m_cv_push;
        std::condition_variable m_cv_pop;
        bool m_closed = false;

    public:
        BlockingQueue(size_t capacity)
            : m_buffer(reserve_only, capacity, capacity)
            , m_capacity(capacity)
        {
        }

        void close() {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_closed = true;
            }
            m_cv_push.notify_all();
            m_cv_pop.notify_all();
        }

        bool push(const T& item) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv_push.wait(lock, [this] { return m_count < m_capacity || m_closed; });
            if (m_closed) return false;

            m_buffer[m_tail] = item;
            m_tail = (m_tail + 1) % m_capacity;
            m_count++;

            lock.unlock();
            m_cv_pop.notify_one();
            return true;
        }

        size_t pop_batch(AlignedVec<T>& out, size_t maxItems, std::chrono::microseconds timeout) {
            std::unique_lock<std::mutex> lock(m_mutex);

            if (!m_cv_pop.wait_for(lock, timeout, [this] { return m_count > 0 || m_closed; })) {
                return 0;
            }
            if (m_closed && m_count == 0) return 0;

            size_t n = std::min(m_count, maxItems);
            for (size_t i = 0; i < n; ++i) {
                out.push_back(m_buffer[m_head]);
                m_head = (m_head + 1) % m_capacity;
            }
            m_count -= n;

            lock.unlock();
            m_cv_push.notify_all();
            return n;
        }

        bool pop(T& out) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv_pop.wait(lock, [this] { return m_count > 0 || m_closed; });
            if (m_count == 0 && m_closed) return false;

            out = m_buffer[m_head];
            m_head = (m_head + 1) % m_capacity;
            m_count--;

            lock.unlock();
            m_cv_push.notify_one();
            return true;
        }
    };

    // ============================================================================
    // THREAD POOL
    // ============================================================================
    template<ValidGameTraits GT>
    class ThreadPool
    {
    private:
        USING_GAME_TYPES(GT);
        using Event = NodeEvent<GT>;
        using ModelResults = ModelResultsT<GT>;

        std::shared_ptr<IEngine<GT>> m_engine;
        AlignedVec<std::unique_ptr<NeuralNet<GT>>> m_neuralNets;
        std::atomic<TreeSearch<GT>*> m_currentTree{ nullptr };

        AlignedVec<std::unique_ptr<Event>> m_eventPool;

        BlockingQueue<Event*> m_qFree;
        BlockingQueue<Event*> m_qEval;
        BlockingQueue<Event*> m_qBackprop;

        std::vector<std::thread> m_workers;

        std::atomic<bool> m_running{ true };
        std::atomic<uint32_t> m_activeJobs{ 0 };
        std::atomic<uint32_t> m_targetSims{ 0 };

        std::mutex m_mainMutex;
        std::condition_variable m_mainCV;

    public:
        ThreadPool(std::shared_ptr<IEngine<GT>> engine,
            AlignedVec<std::unique_ptr<NeuralNet<GT>>>&& nets,
            const SystemConfig& sysCfg,
            const TreeSearchConfig& tsCfg)
            : m_engine(engine)
            , m_neuralNets(std::move(nets))
            , m_qFree(sysCfg.batchSize* m_neuralNets.size() * 2 + 256)
            , m_qEval(sysCfg.batchSize* m_neuralNets.size() * 2 + 256)
            , m_qBackprop(sysCfg.batchSize* m_neuralNets.size() * 2 + 256)
            , m_eventPool(reserve_only, sysCfg.batchSize* m_neuralNets.size() * 2 + 256)
        {
            size_t nContexts = sysCfg.batchSize * m_neuralNets.size() * 2 +
                sysCfg.numSearchThreads + sysCfg.numBackpropThreads + 64;

            for (size_t i = 0; i < nContexts; ++i) {
                m_eventPool.push_back(std::make_unique<Event>(tsCfg.historySize, tsCfg.maxDepth));
                m_qFree.push(m_eventPool.back().get());
            }

            for (int i = 0; i < sysCfg.numSearchThreads; ++i) {
                m_workers.emplace_back(&ThreadPool::loopGather, this);
            }
            for (size_t g = 0; g < m_neuralNets.size(); ++g) {
                for (int k = 0; k < sysCfg.numInferenceThreadsPerGPU; ++k) {
                    m_workers.emplace_back(&ThreadPool::loopInference, this, g, sysCfg.batchSize);
                }
            }
            for (int i = 0; i < sysCfg.numBackpropThreads; ++i) {
                m_workers.emplace_back(&ThreadPool::loopBackprop, this);
            }
        }

        ~ThreadPool() {
            m_running = false;
            m_qFree.close();
            m_qEval.close();
            m_qBackprop.close();
            for (auto& t : m_workers) if (t.joinable()) t.join();
        }

        void executeTreeSearch(TreeSearch<GT>* tree, uint32_t numSims)
        {
            m_currentTree.store(tree, std::memory_order_release);
            m_targetSims.store(numSims, std::memory_order_relaxed);

            std::unique_lock<std::mutex> lock(m_mainMutex);
            m_mainCV.wait(lock, [&]() {
                bool simDone = tree->getSimulationCount() >= numSims;
                bool jobsDone = m_activeJobs.load(std::memory_order_relaxed) == 0;
                return simDone && jobsDone;
                });
            m_currentTree.store(nullptr, std::memory_order_release);
        }

    private:
        void loopGather()
        {
            Event* ctx = nullptr;
            while (m_running)
            {
                // 1. Lecture SÉCURISÉE (atomique) du pointeur de l'arbre
                TreeSearch<GT>* tree = m_currentTree.load(std::memory_order_acquire);

                // 2. Mode "Veille" (Évite de brûler le CPU à 100% quand l'IA ne réfléchit pas)
                if (!tree || tree->getSimulationCount() >= m_targetSims.load(std::memory_order_relaxed)) {
                    // On dort 1 milliseconde et on recommence la boucle.
                    // Ça consomme 0% de CPU.
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                // 3. On attend qu'un contexte mémoire soit libre pour travailler
                if (!m_qFree.pop(ctx)) break; // Si le programme se ferme (m_closed), on quitte

                // 4. On déclare qu'on commence un travail AVANT de faire quoi que ce soit
                // C'est ça qui empêche le thread principal de détruire l'arbre sous notre nez !
                m_activeJobs.fetch_add(1, std::memory_order_acquire);

                // 5. DOUBLE VÉRIFICATION (La clé de la sécurité)
                // Entre le moment où on s'est réveillé et maintenant, le thread principal
                // a peut-être annulé l'arbre ou fini la recherche. On relit le pointeur !
                tree = m_currentTree.load(std::memory_order_acquire);

                if (!tree || tree->getSimulationCount() >= m_targetSims.load(std::memory_order_relaxed)) {
                    // Fausse alerte : le travail est déjà fini.
                    // On range le contexte, on annule notre job et on prévient le thread principal.
                    m_qFree.push(ctx);
                    m_activeJobs.fetch_sub(1, std::memory_order_release);
                    m_mainCV.notify_all();
                    continue;
                }

                // 6. LE VRAI TRAVAIL (On utilise la variable locale 'tree', jamais m_currentTree ici)
                bool needEval = tree->gather(*ctx);

                // 7. Routage du contexte
                if (needEval) {
                    m_qEval.push(ctx); // Le noeud a besoin du GPU
                }
                else {
                    m_qBackprop.push(ctx); // Le noeud est terminal, on met à jour direct
                }
            }
        }

        void loopInference(size_t gpuIdx, uint32_t batchSize)
        {
            AlignedVec<Event*> batchEvents(reserve_only, batchSize);
            AlignedVec<State> batchStates(reserve_only, batchSize);
            AlignedVec<AlignedVec<Action>> batchHistories(reserve_only, batchSize);
            AlignedVec<ModelResults> batchOutputs(reserve_only, batchSize);

            auto& net = m_neuralNets[gpuIdx];

            while (m_running)
            {
                batchEvents.clear();
                size_t count = m_qEval.pop_batch(batchEvents, batchSize, std::chrono::microseconds(200));
                if (count == 0) continue;

                batchStates.clear();
                batchHistories.clear();

                for (auto* e : batchEvents) {
                    batchStates.push_back(e->nnInputState);
                    batchHistories.push_back(e->nnInputHistory);
                }

                batchOutputs.resize(count);

                // Exécution PyTorch / TensorRT 
                net->forwardBatch(batchStates, batchHistories, batchOutputs);

                for (size_t i = 0; i < count; ++i) {
                    Event* e = batchEvents[i];
                    const auto& res = batchOutputs[i];

                    // 1. Mise à jour des Values
                    for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                        e->values[p] = res.values[p];
                    }

                    // 2. Mise à jour de la Policy (avec std::array)
                    // La policy de l'Event a déjà été remise à 0.0f par ctx.reset() dans gather() !
                    float sum = 0.0f;

                    for (const auto& act : e->validActions) {
                        uint32_t idx = m_engine->actionToIdx(act);

                        // Sécurité : on s'assure que l'index renvoyé par l'engine est valide
                        if (idx < Defs::kActionSpace && idx < res.policy.size()) {
                            float p = res.policy[idx];
                            e->policy[idx] = p; // On écrit directement à l'index absolu !
                            sum += p;
                        }
                    }

                    // Normalisation sur les coups légaux uniquement
                    if (sum > 1e-9f) {
                        float invSum = 1.0f / sum;
                        // On ne parcourt QUE les coups valides pour optimiser
                        for (const auto& act : e->validActions) {
                            uint32_t idx = m_engine->actionToIdx(act);
                            if (idx < Defs::kActionSpace) {
                                e->policy[idx] *= invSum;
                            }
                        }
                    }
                    else if (!e->validActions.empty()) {
                        // Fallback : Si le réseau donne 0 partout, on met une proba uniforme
                        float uniform = 1.0f / e->validActions.size();
                        for (const auto& act : e->validActions) {
                            uint32_t idx = m_engine->actionToIdx(act);
                            if (idx < Defs::kActionSpace) {
                                e->policy[idx] = uniform;
                            }
                        }
                    }

                    m_qBackprop.push(e);
                }
            }
        }

        void loopBackprop()
        {
            Event* ctx = nullptr;
            while (m_running)
            {
                if (!m_qBackprop.pop(ctx)) break;

                TreeSearch<GT>* tree = m_currentTree.load(std::memory_order_acquire);
                if (tree) {
                    tree->backprop(*ctx);
                }

                m_qFree.push(ctx);

                uint32_t remaining = m_activeJobs.fetch_sub(1, std::memory_order_relaxed) - 1;

                if (tree &&
                    tree->getSimulationCount() >= m_targetSims.load(std::memory_order_relaxed) &&
                    remaining == 0)
                {
                    m_mainCV.notify_all();
                }
            }
        }
    };
}