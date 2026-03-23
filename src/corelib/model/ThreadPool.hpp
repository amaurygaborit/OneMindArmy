#pragma once

#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#include "BlockingQueue.hpp"
#include "TreeSearch.hpp"
#include "NeuralNet.hpp"
#include "../util/AlignedVec.hpp"

namespace Core
{
    template<ValidGameTraits GT>
    class ThreadPool
    {
    private:
        USING_GAME_TYPES(GT);
        using Event = NodeEvent<GT>;
        using ModelResults = ModelResultsT<GT>;

        struct TreeTask { TreeSearch<GT>* tree; uint32_t targetSims; bool isSelfPlay; };
        struct EvalTask { TreeSearch<GT>* tree; Event* ctx; uint32_t targetSims; bool isSelfPlay; };

        std::shared_ptr<IEngine<GT>>               m_engine;
        AlignedVec<std::unique_ptr<NeuralNet<GT>>> m_neuralNets;
        AlignedVec<std::unique_ptr<Event>>         m_eventPool;

        BlockingQueue<TreeTask> m_qReadyTrees;
        BlockingQueue<Event*>   m_qFree;
        BlockingQueue<EvalTask> m_qEval;
        BlockingQueue<EvalTask> m_qBackprop;

        std::vector<std::thread> m_workers;
        std::atomic<bool>        m_running{ true };
        bool                     m_fastDrain;

        std::atomic<uint32_t>    m_pendingTasks{ 0 };
        std::mutex               m_mainMutex;
        std::condition_variable  m_mainCV;

        static size_t calcPoolSize(const BackendConfig& cfg, size_t nNets) {
            return static_cast<size_t>(cfg.numParallelGames * nNets * cfg.queueScale * 2) + 256;
        }

    public:
        ThreadPool(std::shared_ptr<IEngine<GT>> engine,
            AlignedVec<std::unique_ptr<NeuralNet<GT>>>&& nets,
            const BackendConfig& backendCfg,
            const EngineConfig& engineCfg)
            : m_engine(engine)
            , m_neuralNets(std::move(nets))
            , m_fastDrain(backendCfg.fastDrain)
            // L'astuce anti-deadlock absolue : Toutes les queues (sauf TreeTask) font EXACTEMENT la taille du pool
            , m_qReadyTrees(calcPoolSize(backendCfg, m_neuralNets.size()) * 4) // Peut être inondée de tâches
            , m_qFree(calcPoolSize(backendCfg, m_neuralNets.size()))
            , m_qEval(calcPoolSize(backendCfg, m_neuralNets.size()))
            , m_qBackprop(calcPoolSize(backendCfg, m_neuralNets.size()))
            , m_eventPool(reserve_only, calcPoolSize(backendCfg, m_neuralNets.size()))
        {
            const size_t nCtx = m_eventPool.capacity();
            for (size_t i = 0; i < nCtx; ++i) {
                m_eventPool.push_back(std::make_unique<Event>(engineCfg.maxDepth));
                m_qFree.push(m_eventPool.back().get());
            }

            for (uint32_t i = 0; i < backendCfg.numSearchThreads; ++i)
                m_workers.emplace_back(&ThreadPool::loopGather, this);

            for (uint32_t g = 0; g < static_cast<uint32_t>(m_neuralNets.size()); ++g)
                for (uint32_t k = 0; k < backendCfg.numInferenceThreads; ++k)
                    m_workers.emplace_back(&ThreadPool::loopInference, this, static_cast<size_t>(g), backendCfg.inferenceBatchSize);

            for (uint32_t i = 0; i < backendCfg.numBackpropThreads; ++i)
                m_workers.emplace_back(&ThreadPool::loopBackprop, this);
        }

        ~ThreadPool()
        {
            m_running = false;
            m_qReadyTrees.close(m_fastDrain);
            m_qFree.close(m_fastDrain);
            m_qEval.close(m_fastDrain);
            m_qBackprop.close(m_fastDrain);
            for (auto& t : m_workers) if (t.joinable()) t.join();
        }

        void executeMultipleTrees(const std::vector<TreeSearch<GT>*>& trees, uint32_t numSims)
        {
            if (trees.empty() || numSims == 0) return;
            for (auto* tree : trees) tree->resetCounters();

            const bool isSelfPlay = (trees.size() > 1);

            uint32_t initialTasksTotal = 0;
            for (auto* tree : trees) {
                initialTasksTotal += isSelfPlay ? 1 : std::min<uint32_t>(numSims, 32);
            }

            // CORRECTION CRITIQUE : fetch_add empêche la Race Condition si plusieurs handlers appellent ThreadPool
            m_pendingTasks.fetch_add(initialTasksTotal, std::memory_order_release);

            for (auto* tree : trees) {
                uint32_t initialThreads = isSelfPlay ? 1 : std::min<uint32_t>(numSims, 32);
                for (uint32_t i = 0; i < initialThreads; ++i) {
                    m_qReadyTrees.push({ tree, numSims, isSelfPlay });
                }
            }

            // Attente propre et isolée sur la condition
            std::unique_lock lock(m_mainMutex);
            m_mainCV.wait(lock, [this] {
                return m_pendingTasks.load(std::memory_order_acquire) == 0;
                });
        }

        void executeTreeSearch(TreeSearch<GT>* tree, uint32_t numSims) {
            executeMultipleTrees({ tree }, numSims);
        }

    private:
        void loopGather()
        {
            TreeTask tTask;
            Event* ctx = nullptr;

            while (m_running)
            {
                if (!m_qReadyTrees.pop(tTask)) break;

                if (tTask.tree->getSimulationCount() >= tTask.targetSims) {
                    notifyTaskDone();
                    continue;
                }

                if (!m_qFree.pop(ctx)) {
                    notifyTaskDone();
                    break;
                }

                tTask.tree->incrementLaunched();

                const bool needEval = tTask.tree->gather(*ctx);
                EvalTask eTask{ tTask.tree, ctx, tTask.targetSims, tTask.isSelfPlay };

                if (needEval) m_qEval.push(eTask);
                else          m_qBackprop.push(eTask);
            }
        }

        void loopInference(size_t gpuIdx, uint32_t configBatchSize)
        {
            if (cudaSetDevice(static_cast<int>(gpuIdx)) != cudaSuccess) {
                std::cerr << "[ThreadPool] Fatal: cannot bind to GPU " << gpuIdx << "\n";
                return;
            }

            auto& net = m_neuralNets[gpuIdx];

            AlignedVec<EvalTask> batchTasks(reserve_only, configBatchSize);
            AlignedVec<ModelResults> batchOutputs(reserve_only, configBatchSize);

            // NOUVEAU : Un simple tableau de pointeurs (très léger, aucune copie de tenseur)
            AlignedVec<const std::array<float, Defs::kNNInputSize>*> batchPtrs(reserve_only, configBatchSize);

            while (m_running)
            {
                batchTasks.clear();
                const size_t count = m_qEval.pop_batch(batchTasks, configBatchSize, std::chrono::microseconds(1000));
                if (count == 0) continue;

                batchPtrs.clear();
                batchOutputs.resize(count);

                // On ne stocke que l'adresse mémoire de l'input du context MCTS
                for (const auto& task : batchTasks)
                    batchPtrs.push_back(&task.ctx->nnInput);

                net->forwardBatch(batchPtrs, batchOutputs);

                for (size_t i = 0; i < count; ++i)
                {
                    EvalTask& eTask = batchTasks[i];
                    Event* e = eTask.ctx;
                    const ModelResults& res = batchOutputs[i];

                    e->nnWDL = res.values;

                    float maxLogit = -1e9f;
                    for (const auto& act : e->validActions) {
                        const uint32_t idx = m_engine->actionToIdx(act);
                        if (idx < Defs::kActionSpace) maxLogit = std::max(maxLogit, res.policy[idx]);
                    }

                    float sumExp = 0.0f;
                    for (const auto& act : e->validActions) {
                        const uint32_t idx = m_engine->actionToIdx(act);
                        if (idx < Defs::kActionSpace) {
                            e->policy[idx] = std::exp(res.policy[idx] - maxLogit);
                            sumExp += e->policy[idx];
                        }
                    }

                    if (sumExp > 1e-9f) {
                        const float inv = 1.0f / sumExp;
                        for (const auto& act : e->validActions) {
                            const uint32_t idx = m_engine->actionToIdx(act);
                            if (idx < Defs::kActionSpace) e->policy[idx] *= inv;
                        }
                    }
                    else if (!e->validActions.empty()) {
                        const float uni = 1.0f / static_cast<float>(e->validActions.size());
                        for (const auto& act : e->validActions) {
                            const uint32_t idx = m_engine->actionToIdx(act);
                            if (idx < Defs::kActionSpace) e->policy[idx] = uni;
                        }
                    }

                    m_qBackprop.push(eTask);
                }
            }
        }

        void loopBackprop()
        {
            EvalTask eTask;
            while (m_running)
            {
                if (!m_qBackprop.pop(eTask)) break;

                eTask.tree->backprop(*(eTask.ctx));
                m_qFree.push(eTask.ctx);

                if (eTask.tree->getSimulationCount() < eTask.targetSims) {
                    m_qReadyTrees.push({ eTask.tree, eTask.targetSims, eTask.isSelfPlay });
                }
                else {
                    notifyTaskDone();
                }
            }
        }

        inline void notifyTaskDone() {
            if (m_pendingTasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                m_mainCV.notify_all();
            }
        }
    };
}