#pragma once

#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <memory>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#include "TreeSearch.hpp"
#include "NeuralNet.hpp"
#include "../util/AlignedVec.hpp"

namespace Core
{
    // ========================================================================
    // BLOCKING QUEUE (ring buffer, thread-safe)
    // ========================================================================
    template<typename T>
    class BlockingQueue
    {
        AlignedVec<T> m_buffer;
        size_t        m_capacity;
        size_t        m_head = 0;
        size_t        m_tail = 0;
        size_t        m_count = 0;

        std::mutex              m_mutex;
        std::condition_variable m_cv_push;
        std::condition_variable m_cv_pop;
        bool m_closed = false;
        bool m_fastDrain = false;

    public:
        explicit BlockingQueue(size_t capacity)
            : m_buffer(reserve_only, capacity, capacity)
            , m_capacity(capacity)
        {
        }

        void close(bool fastDrain = false)
        {
            { std::lock_guard lock(m_mutex); m_closed = true; m_fastDrain = fastDrain; if (m_fastDrain) m_count = 0; }
            m_cv_push.notify_all();
            m_cv_pop.notify_all();
        }

        bool push(const T& item)
        {
            std::unique_lock lock(m_mutex);
            m_cv_push.wait(lock, [this] { return m_count < m_capacity || m_closed; });
            if (m_closed) return false;
            m_buffer[m_tail] = item;
            m_tail = (m_tail + 1) % m_capacity;
            ++m_count;
            lock.unlock();
            m_cv_pop.notify_one();
            return true;
        }

        size_t pop_batch(AlignedVec<T>& out, size_t maxItems, std::chrono::microseconds timeout)
        {
            std::unique_lock lock(m_mutex);
            m_cv_pop.wait_for(lock, timeout, [this, maxItems] { return m_count >= maxItems || m_closed; });
            if (m_closed && (m_count == 0 || m_fastDrain)) return 0;
            size_t n = std::min(m_count, maxItems);
            if (n == 0) return 0;
            for (size_t i = 0; i < n; ++i) { out.push_back(m_buffer[m_head]); m_head = (m_head + 1) % m_capacity; }
            m_count -= n;
            lock.unlock();
            m_cv_push.notify_all();
            return n;
        }

        bool pop(T& out)
        {
            std::unique_lock lock(m_mutex);
            m_cv_pop.wait(lock, [this] { return m_count > 0 || m_closed; });
            if (m_closed && (m_count == 0 || m_fastDrain)) return false;
            out = m_buffer[m_head];
            m_head = (m_head + 1) % m_capacity;
            --m_count;
            lock.unlock();
            m_cv_push.notify_one();
            return true;
        }
    };

    // ========================================================================
    // THREAD POOL
    //
    // Pipeline: SelfPlay → Gather → Inference → Backprop
    //
    // WDL contract:
    //   GameResult::wdl      = std::array<float, kNumPlayers * 3>
    //   ModelResultsT::values = std::array<float, kNumPlayers * 3>
    //   NodeEvent::nnWDL     = std::array<float, kNumPlayers * 3>
    //   NodeEvent::trueWDL   = std::array<float, kNumPlayers * 3>
    //
    //   All four arrays are the same size. loopInference copies
    //   res.values → e->nnWDL with std::copy_n, no OOB risk.
    // ========================================================================
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

        std::atomic<uint32_t>   m_pendingSimulations{ 0 };
        std::mutex              m_mainMutex;
        std::condition_variable m_mainCV;

        std::atomic<uint32_t>   m_starvationCount{ 0 };

        static size_t qSize(const BackendConfig& cfg, size_t nNets, size_t extra)
        {
            return static_cast<size_t>(cfg.numParallelGames * nNets * cfg.queueScale * 2) + extra;
        }

    public:
        ThreadPool(std::shared_ptr<IEngine<GT>>                engine,
            AlignedVec<std::unique_ptr<NeuralNet<GT>>>&& nets,
            const BackendConfig& backendCfg,
            const EngineConfig& engineCfg)
            : m_engine(engine)
            , m_neuralNets(std::move(nets))
            , m_fastDrain(backendCfg.fastDrain)
            , m_qReadyTrees(qSize(backendCfg, m_neuralNets.size(), 512))
            , m_qFree(qSize(backendCfg, m_neuralNets.size(), 256))
            , m_qEval(qSize(backendCfg, m_neuralNets.size(), 256))
            , m_qBackprop(qSize(backendCfg, m_neuralNets.size(), 256))
            , m_eventPool(reserve_only, qSize(backendCfg, m_neuralNets.size(), 256))
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
                    m_workers.emplace_back(&ThreadPool::loopInference, this,
                        static_cast<size_t>(g), backendCfg.inferenceBatchSize);

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

            const uint32_t totalSims = static_cast<uint32_t>(trees.size()) * numSims;
            m_pendingSimulations.store(totalSims, std::memory_order_release);

            const bool isSelfPlay = (trees.size() > 1);
            for (auto* tree : trees) m_qReadyTrees.push({ tree, numSims, isSelfPlay });

            std::unique_lock lock(m_mainMutex);
            m_mainCV.wait(lock, [&] {
                return m_pendingSimulations.load(std::memory_order_acquire) == 0;
                });
        }

        void executeTreeSearch(TreeSearch<GT>* tree, uint32_t numSims)
        {
            executeMultipleTrees({ tree }, numSims);
        }

    private:
        // ====================================================================
        // GATHER — tree traversal, virtual loss, encode NN input
        // ====================================================================
        void loopGather()
        {
            TreeTask tTask;
            Event* ctx = nullptr;

            while (m_running)
            {
                if (!m_qReadyTrees.pop(tTask)) break;
                if (!m_qFree.pop(ctx)) {
                    m_starvationCount.fetch_add(1, std::memory_order_relaxed);
                    std::cerr << "[ThreadPool] Context starvation! count=" << m_starvationCount.load() << "\n";
                    break;
                }

                if (tTask.tree->incrementLaunched() > tTask.targetSims) {
                    m_qFree.push(ctx);
                    continue;
                }

                // Swarm mode: flood queue for pure-inference
                if (!tTask.isSelfPlay && tTask.tree->getLaunchedCount() < tTask.targetSims)
                    m_qReadyTrees.push(tTask);

                const bool needEval = tTask.tree->gather(*ctx);
                EvalTask eTask{ tTask.tree, ctx, tTask.targetSims, tTask.isSelfPlay };
                if (needEval) m_qEval.push(eTask);
                else          m_qBackprop.push(eTask);
            }
        }

        // ====================================================================
        // INFERENCE — GPU batch forward pass
        //
        // ModelResultsT::values is std::array<float, kNumPlayers * 3>
        // NodeEvent::nnWDL      is std::array<float, kNumPlayers * 3>
        // Both have the same size → std::copy_n is safe.
        //
        // GameResult is a struct { wdl: array<float, kNumPlayers*3>; reason: uint32_t }.
        // We do NOT touch GameResult here — that is handled in TreeSearch::gather()
        // via copyWDLFromResult(outcome->wdl, ctx.trueWDL).
        // ====================================================================
        void loopInference(size_t gpuIdx, uint32_t configBatchSize)
        {
            if (cudaSetDevice(static_cast<int>(gpuIdx)) != cudaSuccess) {
                std::cerr << "[ThreadPool] Fatal: cannot bind to GPU " << gpuIdx << "\n";
                return;
            }

            auto& net = m_neuralNets[gpuIdx];

            AlignedVec<EvalTask>                              batchTasks(reserve_only, configBatchSize);
            AlignedVec<std::array<float, Defs::kNNInputSize>> batchInputs(reserve_only, configBatchSize);
            AlignedVec<ModelResults>                          batchOutputs(reserve_only, configBatchSize);

            while (m_running)
            {
                batchTasks.clear();
                const size_t count = m_qEval.pop_batch(batchTasks, configBatchSize,
                    std::chrono::microseconds(1000));
                if (count == 0) continue;

                batchInputs.clear();
                batchOutputs.resize(count);
                for (const auto& task : batchTasks) batchInputs.push_back(task.ctx->nnInput);

                net->forwardBatch(batchInputs, batchOutputs);

                for (size_t i = 0; i < count; ++i)
                {
                    EvalTask& eTask = batchTasks[i];
                    Event* e = eTask.ctx;
                    const ModelResults& res = batchOutputs[i];

                    // -------------------------------------------------------
                    // Copy WDL: kNumPlayers * 3 floats from res.values to e->nnWDL.
                    //
                    // res.values : std::array<float, kNumPlayers * 3>  (NeuralNet.hpp)
                    // e->nnWDL   : std::array<float, kNumPlayers * 3>  (TreeSearch.hpp)
                    //
                    // They are the same type — a simple array assignment works.
                    // No OOB risk. No static_assert needed (same template parameter).
                    // -------------------------------------------------------
                    e->nnWDL = res.values;

                    // Normalise policy over legal moves only
                    float sum = 0.0f;
                    for (const auto& act : e->validActions) {
                        const uint32_t idx = m_engine->actionToIdx(act);
                        if (idx < Defs::kActionSpace) { e->policy[idx] = res.policy[idx]; sum += res.policy[idx]; }
                    }

                    if (sum > 1e-9f) {
                        const float inv = 1.0f / sum;
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

        // ====================================================================
        // BACKPROP — update tree + release virtual loss
        // ====================================================================
        void loopBackprop()
        {
            EvalTask eTask;
            while (m_running)
            {
                if (!m_qBackprop.pop(eTask)) break;

                const bool wasCollision = eTask.ctx->collision;
                eTask.tree->backprop(*(eTask.ctx));
                m_qFree.push(eTask.ctx);

                if ((eTask.isSelfPlay || wasCollision) && eTask.tree->getLaunchedCount() < eTask.targetSims)
                    m_qReadyTrees.push({ eTask.tree, eTask.targetSims, eTask.isSelfPlay });

                if (!wasCollision) {
                    if (m_pendingSimulations.fetch_sub(1, std::memory_order_acq_rel) == 1)
                        m_mainCV.notify_all();
                }
            }
        }
    };
}