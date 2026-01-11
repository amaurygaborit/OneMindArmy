#pragma once
#include "MCTS.hpp"
#include "../bootstrap/GameConfig.hpp"
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>

template<typename GameTag>
class MCTSThreadPool
{
public:
    using Event = NodeEvent<GameTag>;
    using IdxAction = typename MCTS<GameTag>::IdxAction;
    using IdxStateAction = typename MCTS<GameTag>::IdxStateAction;
    using ModelResults = typename MCTS<GameTag>::ModelResults;

private:
    template<typename T>
    class ThreadSafeQueue
    {
        std::deque<T> queue;
        mutable std::mutex m_mutex;
        std::condition_variable m_cv;
        bool m_done = false;

    public:
        void push(T item)
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                queue.push_back(item);
            }
            m_cv.notify_one();
        }

        bool pop(T& item)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return !queue.empty() || m_done; });
            if (queue.empty()) return false;
            item = queue.front();
            queue.pop_front();
            return true;
        }

        size_t pop_batch_opportunistic(AlignedVec<T>& outBatch, size_t maxItems, std::chrono::microseconds timeout)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!m_cv.wait_for(lock, timeout, [this] { return !queue.empty() || m_done; }))
            {
                if (queue.empty()) return 0;
            }
            size_t count = 0;
            while (!queue.empty() && count < maxItems)
            {
                outBatch.push_back(queue.front());
                queue.pop_front();
                count++;
            }
            return count;
        }

        void signal_done()
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_done = true;
            }
            m_cv.notify_all();
        }

        void reset()
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_done = false;
            queue.clear();
        }

        bool empty() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return queue.empty();
        }
    };

    std::mutex m_waitMutex;
    std::condition_variable m_waitCV;
    std::atomic<int> m_busyEvents{ 0 };
    std::atomic<uint32_t> m_targetSimulations{ 0 };

    std::shared_ptr<IEngine<GameTag>> m_engine;
    AlignedVec<std::unique_ptr<NeuralNet<GameTag>>> m_neuralNets;
    AlignedVec<std::unique_ptr<std::mutex>> m_netMutexes;

    // Configuration Système (Backend)
    const SystemConfig m_sysConfig;

    AlignedVec<Event> m_eventStorage;
    ThreadSafeQueue<Event*> m_freeEvents;

    AlignedVec<std::thread> m_gatherThreads;
    AlignedVec<std::thread> m_inferenceThreads;
    AlignedVec<std::thread> m_backpropThreads;

    ThreadSafeQueue<Event*> m_evalQueue;
    ThreadSafeQueue<Event*> m_backpropQueue;

    std::atomic<bool> m_stopFlag{ false };
    std::atomic<bool> m_draining{ false };

    MCTS<GameTag>* m_currentMCTS{ nullptr };

public:
    MCTSThreadPool(std::shared_ptr<IEngine<GameTag>> engine,
        AlignedVec<std::unique_ptr<NeuralNet<GameTag>>>&& neuralNets,
        const SystemConfig& sysConfig,
        const MCTSConfig& mctsConfig) // MCTSConfig pour l'init des Events
        : m_engine(std::move(engine)),
        m_neuralNets(std::move(neuralNets)),
        m_sysConfig(sysConfig)
    {
        if (m_neuralNets.empty())
            throw std::runtime_error("MCTSThreadPool: 0 NeuralNets provided");

        m_netMutexes.reserve(m_neuralNets.size());
        for (size_t i = 0; i < m_neuralNets.size(); ++i)
            m_netMutexes.push_back(std::make_unique<std::mutex>());

        size_t totalInferenceThreads = m_neuralNets.size() * m_sysConfig.numInferenceThreadsPerGPU;
        size_t totalThreads = m_sysConfig.numSearchThreads + totalInferenceThreads + m_sysConfig.numBackpropThreads;

        // Note: BatchSize est maintenant dans sysConfig
        size_t poolSize = static_cast<size_t>(
            (m_sysConfig.batchSize * m_neuralNets.size() * m_sysConfig.queueScale) + (totalThreads * 4)
            );

        m_eventStorage.reserve(poolSize);
        for (size_t i = 0; i < poolSize; ++i)
        {
            // Init Events avec params MCTS (history/depth)
            m_eventStorage.emplace_back(mctsConfig.historySize, mctsConfig.maxDepth);
            m_freeEvents.push(&m_eventStorage.back());
        }

        for (int i = 0; i < m_sysConfig.numBackpropThreads; ++i)
            m_backpropThreads.emplace_back(&MCTSThreadPool::backpropLoop, this);

        for (size_t gpuIdx = 0; gpuIdx < m_neuralNets.size(); ++gpuIdx)
        {
            for (size_t k = 0; k < m_sysConfig.numInferenceThreadsPerGPU; ++k)
            {
                m_inferenceThreads.emplace_back(&MCTSThreadPool::inferenceLoop, this, gpuIdx);
            }
        }

        for (int i = 0; i < m_sysConfig.numSearchThreads; ++i)
        {
            m_gatherThreads.emplace_back(&MCTSThreadPool::gatherLoop, this);
        }
    }

    ~MCTSThreadPool()
    {
        m_stopFlag = true;
        m_draining = true;

        m_freeEvents.signal_done();
        m_evalQueue.signal_done();
        m_backpropQueue.signal_done();

        for (auto& t : m_gatherThreads) if (t.joinable()) t.join();
        for (auto& t : m_inferenceThreads) if (t.joinable()) t.join();
        for (auto& t : m_backpropThreads) if (t.joinable()) t.join();
    }

    void executeMCTS(MCTS<GameTag>* mcts, uint32_t numSimulations)
    {
        m_currentMCTS = mcts;
        m_evalQueue.reset();
        m_backpropQueue.reset();

        m_targetSimulations.store(numSimulations, std::memory_order_relaxed);
        m_draining = false;

        {
            std::unique_lock<std::mutex> lock(m_waitMutex);
            while (mcts->getSimulationCount() < numSimulations)
            {
                m_waitCV.wait_for(lock, std::chrono::microseconds(500));
            }
        }

        m_draining = true;

        {
            std::unique_lock<std::mutex> lock(m_waitMutex);
            while (m_busyEvents.load(std::memory_order_relaxed) > 0)
            {
                m_waitCV.wait_for(lock, std::chrono::microseconds(500));
            }
        }

        m_currentMCTS = nullptr;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

private:
    void gatherLoop()
    {
        while (!m_stopFlag)
        {
            MCTS<GameTag>* mcts = m_currentMCTS;

            if (!mcts || m_draining)
            {
                std::this_thread::yield();
                continue;
            }

            Event* event = nullptr;
            if (!m_freeEvents.pop(event))
            {
                std::this_thread::yield();
                continue;
            }

            m_busyEvents.fetch_add(1, std::memory_order_relaxed);

            if (mcts->gatherWalk(*event))
            {
                m_evalQueue.push(event);
            }
            else
            {
                m_freeEvents.push(event);
                m_busyEvents.fetch_sub(1, std::memory_order_relaxed);
                std::this_thread::yield();
            }
        }
    }

    void inferenceLoop(size_t netIndex)
    {
        AlignedVec<Event*> batch;
        batch.reserve(m_sysConfig.batchSize);

        AlignedVec<IdxStateAction> nnInput;
        nnInput.reserve(m_sysConfig.batchSize * 16);

        AlignedVec<ModelResults> nnOutput;
        nnOutput.reserve(m_sysConfig.batchSize);

        IdxAction idxAct;
        const auto kWaitTimeout = std::chrono::microseconds(100);

        auto& myNet = m_neuralNets[netIndex];
        auto& myMutex = *m_netMutexes[netIndex];

        while (!m_stopFlag)
        {
            batch.clear();
            // Utilisation de m_sysConfig.batchSize
            size_t count = m_evalQueue.pop_batch_opportunistic(batch, m_sysConfig.batchSize, kWaitTimeout);

            if (count == 0) continue;

            // Utilisation de m_sysConfig.fastDrain
            if (m_draining && m_sysConfig.fastDrain)
            {
                for (size_t i = 0; i < count; ++i) m_freeEvents.push(batch[i]);
                m_busyEvents.fetch_sub(count, std::memory_order_relaxed);
                m_waitCV.notify_all();
                continue;
            }

            MCTS<GameTag>* mcts = m_currentMCTS;
            if (!mcts)
            {
                for (auto* e : batch) m_freeEvents.push(e);
                m_busyEvents.fetch_sub(count, std::memory_order_relaxed);
                m_waitCV.notify_all();
                continue;
            }

            nnInput.clear();
            for (size_t i = 0; i < count; ++i)
            {
                Event* e = batch[i];
                if (!e->collision && !e->isTerminal)
                {
                    nnInput.insert(nnInput.end(), e->nnHistory.begin(), e->nnHistory.end());
                }
            }

            nnOutput.assign(count, ModelResults());
            if (!nnInput.empty())
            {
                std::lock_guard<std::mutex> lock(myMutex);
                myNet->forwardBatch(nnInput, nnOutput);
            }

            size_t nnReadIdx = 0;
            for (size_t i = 0; i < count; ++i)
            {
                Event* e = batch[i];
                if (!e->collision && !e->isTerminal)
                {
                    if (nnReadIdx < nnOutput.size())
                    {
                        const auto& res = nnOutput[nnReadIdx++];
                        m_engine->getValidActions(e->leafState, e->validActions);

                        if (e->validActions.empty())
                        {
                            e->isTerminal = true;
                            m_engine->isTerminal(e->leafState, e->values);
                        }
                        else
                        {
                            e->values = res.values;
                            float sum = 0.0f;
                            e->policy.clear();
                            for (const auto& act : e->validActions)
                            {
                                m_engine->actionToIdx(act, idxAct);
                                float p = 0.0f;
                                if (idxAct.factIdx < res.policy.size()) p = res.policy[idxAct.factIdx];
                                e->policy.push_back(p);
                                sum += p;
                            }
                            if (sum > 1e-9f) {
                                float norm = 1.0f / sum;
                                for (float& p : e->policy) p *= norm;
                            }
                            else {
                                float uniform = 1.0f / e->policy.size();
                                for (float& p : e->policy) p = uniform;
                            }
                        }
                    }
                }
                m_backpropQueue.push(e);
            }
        }
    }

    void backpropLoop()
    {
        Event* event = nullptr;
        while (!m_stopFlag)
        {
            if (!m_backpropQueue.pop(event))
            {
                if (m_draining && !m_stopFlag) std::this_thread::yield();
                continue;
            }

            MCTS<GameTag>* mcts = m_currentMCTS;
            if (mcts)
            {
                mcts->applyBackprop(*event);
            }
            m_freeEvents.push(event);

            int remaining = m_busyEvents.fetch_sub(1, std::memory_order_relaxed) - 1;
            bool notify = false;

            if (remaining == 0) notify = true;
            if (!notify && mcts) {
                if (mcts->getSimulationCount() >= m_targetSimulations.load(std::memory_order_relaxed)) {
                    notify = true;
                }
            }

            if (notify) {
                m_waitCV.notify_all();
            }
        }
    }
};