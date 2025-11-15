#pragma once
#include "MCTS.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>

template<typename GameTag>
class MCTSThreadPool
{
private:
    using GT = ITraits<GameTag>;
    using ObsState = typename ObsStateT<GameTag>;
    using Action = typename ActionT<GameTag>;
    using IdxStateAction = typename IdxStateActionT<GameTag>;
    using ModelResults = typename ModelResults<GameTag>;
    using ThreadLocalData = typename MCTS<GameTag>::ThreadLocalData;
    using SimulationResult = typename MCTS<GameTag>::SimulationResult;

    // ============================================================================
    // WORKER CONTEXT
    // ============================================================================
    struct WorkerContext {
        std::unique_ptr<ThreadLocalData> tld;
        std::thread thread;
        uint32_t threadId;

        // Results from inference
        std::vector<AlignedVec<float>> inferenceResults;

        WorkerContext(uint32_t id, uint16_t maxDepth, uint32_t batchCapacity, uint8_t historySize)
            : threadId(id) {
            tld = std::make_unique<ThreadLocalData>(maxDepth, batchCapacity, historySize);
        }
    };

    // ============================================================================
    // MEMBER VARIABLES
    // ============================================================================

    std::shared_ptr<NeuralNet<GameTag>> m_neuralNet;

    const uint32_t m_batchSize;
    const uint8_t m_historySize;
    const uint16_t m_maxDepth;
    const uint8_t m_numThreads;

    // Worker threads
    std::vector<std::unique_ptr<WorkerContext>> m_workers;

    // Current MCTS instance
    MCTS<GameTag>* m_currentMCTS{ nullptr };

    // Stop flag
    std::atomic<bool> m_stopFlag{ false };

    // Synchronization for batch collection
    std::mutex m_syncMutex;
    std::condition_variable m_syncCV;
    std::atomic<uint32_t> m_workersWaiting{ 0 };
    std::atomic<bool> m_batchReady{ false };
    std::atomic<bool> m_resultsReady{ false };
    std::atomic<uint32_t> m_workersActive{ 0 };

    // Synchronization for adaptive barrier
    std::atomic<uint32_t> m_batchId{ 0 };           // Current batch generation
    std::atomic<uint32_t> m_workersWithWork{ 0 };    // How many workers have work this batch
    std::atomic<bool> m_coordinationInProgress{ false };

private:
    // ============================================================================
    // WORKER THREAD - Simplified
    // ============================================================================

    void workerThread(uint32_t threadId) {
        WorkerContext& ctx = *m_workers[threadId];
        uint32_t localBatchId = 0;

        while (!m_stopFlag.load(std::memory_order_acquire)) {
            MCTS<GameTag>* mcts = m_currentMCTS;

            if (!mcts || !mcts->m_searchActive.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // CRITICAL: Mark as active at start
            m_workersActive.fetch_add(1, std::memory_order_release);

            try {
                ctx.tld->reset();

                // ================================================================
                // SIMULATION PHASE
                // ================================================================
                while (mcts->m_searchActive.load(std::memory_order_acquire) &&
                    !m_stopFlag.load(std::memory_order_acquire))
                {
                    if (mcts->m_simulationCount.load(std::memory_order_acquire) >=
                        mcts->m_targetSimulations.load(std::memory_order_acquire)) {
                        break;
                    }

                    bool canContinue = mcts->runSimulation(*ctx.tld);
                    if (!canContinue) break;
                }

                // ================================================================
                // DECIDE IF WE PARTICIPATE IN THIS BATCH
                // ================================================================

                bool hasWork = !ctx.tld->localBatch.empty();
                bool searchActive = mcts->m_searchActive.load(std::memory_order_acquire);

                // CRITICAL: Decrement active BEFORE barrier decision
                m_workersActive.fetch_sub(1, std::memory_order_release);

                if (!hasWork || !searchActive) {
                    // No work - skip this batch entirely
                    continue;
                }

                // ================================================================
                // ADAPTIVE BARRIER - Phase 1: Announce participation
                // ================================================================

                // Announce we have work and will participate
                uint32_t workersWithWork = m_workersWithWork.fetch_add(1, std::memory_order_acq_rel) + 1;

                // Wait a bit for all workers to declare their intention
                std::this_thread::sleep_for(std::chrono::microseconds(100));

                // ================================================================
                // ADAPTIVE BARRIER - Phase 2: Synchronize with participants only
                // ================================================================

                uint32_t expectedWorkers = m_workersWithWork.load(std::memory_order_acquire);
                uint32_t arrived = m_workersWaiting.fetch_add(1, std::memory_order_acq_rel) + 1;

                // Check if we're the coordinator (last to arrive)
                bool isCoordinator = (arrived == expectedWorkers);

                if (isCoordinator) {
                    // Extra safety: ensure all workers are at barrier
                    int spinCount = 0;
                    while (m_workersWaiting.load(std::memory_order_acquire) < expectedWorkers
                        && spinCount++ < 10000) {
                        std::this_thread::yield();
                    }

                    if (m_workersWaiting.load(std::memory_order_acquire) != expectedWorkers) {
                        std::cerr << "[Coordinator] WARNING: Not all workers arrived! Expected: "
                            << expectedWorkers << ", Got: "
                            << m_workersWaiting.load() << std::endl;
                    }

                    try {
                        m_coordinationInProgress.store(true, std::memory_order_release);
                        coordinatorPhase();
                    }
                    catch (const std::exception& e) {
                        std::cerr << "[Coordinator] ERROR: " << e.what() << std::endl;
                    }

                    m_coordinationInProgress.store(false, std::memory_order_release);
                    m_resultsReady.store(true, std::memory_order_release);
                    m_syncCV.notify_all();
                }
                else {
                    // Wait for coordinator
                    std::unique_lock<std::mutex> lock(m_syncMutex);

                    auto timeout = std::chrono::seconds(30);
                    if (!m_syncCV.wait_for(lock, timeout, [this] {
                        return m_resultsReady.load(std::memory_order_acquire) ||
                            m_stopFlag.load(std::memory_order_acquire);
                        })) {
                        std::cerr << "[Worker " << threadId << "] TIMEOUT! Expected: "
                            << expectedWorkers << ", Arrived: "
                            << m_workersWaiting.load() << std::endl;

                        // Emergency exit
                        m_workersWaiting.fetch_sub(1, std::memory_order_acq_rel);
                        m_workersWithWork.fetch_sub(1, std::memory_order_acq_rel);
                        continue;
                    }
                }

                if (m_stopFlag.load(std::memory_order_acquire)) {
                    m_workersWaiting.fetch_sub(1, std::memory_order_acq_rel);
                    m_workersWithWork.fetch_sub(1, std::memory_order_acq_rel);
                    continue;
                }

                // ================================================================
                // BACKPROPAGATION
                // ================================================================
                for (size_t i = 0; i < ctx.tld->localBatch.size(); ++i) {
                    if (i < ctx.inferenceResults.size()) {
                        mcts->backpropagate(ctx.tld->localBatch[i], ctx.inferenceResults[i]);
                        mcts->m_simulationCount.fetch_add(1u, std::memory_order_relaxed);
                    }
                }
                ctx.inferenceResults.clear();

                // ================================================================
                // BARRIER EXIT - Reset for next batch
                // ================================================================
                uint32_t remaining = m_workersWaiting.fetch_sub(1, std::memory_order_acq_rel) - 1;
                m_workersWithWork.fetch_sub(1, std::memory_order_acq_rel);

                if (remaining == 0) {
                    // Last worker out resets the barrier
                    m_resultsReady.store(false, std::memory_order_release);
                    m_batchId.fetch_add(1, std::memory_order_release);
                }

            }
            catch (const std::exception& e) {
                std::cerr << "[Worker " << threadId << "] Exception: " << e.what() << std::endl;

                // Emergency cleanup
                uint32_t active = m_workersActive.load(std::memory_order_acquire);
                uint32_t waiting = m_workersWaiting.load(std::memory_order_acquire);
                uint32_t withWork = m_workersWithWork.load(std::memory_order_acquire);

                if (active > 0) m_workersActive.fetch_sub(1, std::memory_order_release);
                if (waiting > 0) m_workersWaiting.fetch_sub(1, std::memory_order_acq_rel);
                if (withWork > 0) m_workersWithWork.fetch_sub(1, std::memory_order_acq_rel);
            }
            catch (...) {
                std::cerr << "[Worker " << threadId << "] Unknown exception" << std::endl;

                uint32_t active = m_workersActive.load(std::memory_order_acquire);
                uint32_t waiting = m_workersWaiting.load(std::memory_order_acquire);
                uint32_t withWork = m_workersWithWork.load(std::memory_order_acquire);

                if (active > 0) m_workersActive.fetch_sub(1, std::memory_order_release);
                if (waiting > 0) m_workersWaiting.fetch_sub(1, std::memory_order_acq_rel);
                if (withWork > 0) m_workersWithWork.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    // ============================================================================
    // COORDINATOR PHASE - Collect batches and run inference
    // ============================================================================

    void coordinatorPhase() {
        MCTS<GameTag>* mcts = m_currentMCTS;
        if (!mcts) return;

        // Count total simulations
        uint32_t totalSims = 0;
        for (auto& worker : m_workers) {
            totalSims += static_cast<uint32_t>(worker->tld->localBatch.size());
        }

        if (totalSims == 0) return;

        // Build unified batch
        AlignedVec<IdxStateAction> batchHistory;
        batchHistory.reserve(totalSims * m_historySize);

        std::vector<std::pair<uint32_t, uint32_t>> simLocations;
        simLocations.reserve(totalSims);

        for (uint32_t wId = 0; wId < m_numThreads; ++wId) {
            auto& worker = m_workers[wId];
            for (uint32_t sId = 0; sId < worker->tld->localBatch.size(); ++sId) {
                const auto& sim = worker->tld->localBatch[sId];

                batchHistory.insert(batchHistory.end(),
                    sim.historyCopy.begin(),
                    sim.historyCopy.end());

                simLocations.push_back({ wId, sId });
            }
        }

        // Run inference
        AlignedVec<ModelResults> results;
        results.resize(totalSims);

        try {
            m_neuralNet->forwardBatch(batchHistory, results);
        }
        catch (const std::exception& e) {
            std::cerr << "ERROR during inference: " << e.what() << std::endl;

            // Clear all batches on error
            for (auto& worker : m_workers) {
                worker->tld->localBatch.clear();
            }
            return;
        }

        // Distribute results
        for (uint32_t i = 0; i < totalSims; ++i) {
            auto [workerId, localIdx] = simLocations[i];
            const ModelResults& res = results[i];

            AlignedVec<float> values(GT::kNumPlayers);
            for (uint8_t p = 0; p < GT::kNumPlayers; ++p) {
                values[p] = res.values[p];
            }

            m_workers[workerId]->inferenceResults.push_back(std::move(values));
        }
    }

public:
    // ============================================================================
    // CONSTRUCTOR & DESTRUCTOR
    // ============================================================================

    MCTSThreadPool(std::shared_ptr<NeuralNet<GameTag>> neuralNet,
        uint32_t batchSize,
        uint8_t historySize,
        uint16_t maxDepth,
        uint8_t numThreads)
        : m_neuralNet(std::move(neuralNet))
        , m_batchSize(batchSize)
        , m_historySize(historySize)
        , m_maxDepth(maxDepth)
        , m_numThreads(numThreads)
    {
        uint32_t perThreadCapacity = (batchSize + numThreads - 1) / numThreads;

        m_workers.reserve(numThreads);
        for (uint8_t i = 0; i < numThreads; ++i) {
            m_workers.emplace_back(
                std::make_unique<WorkerContext>(i, maxDepth, perThreadCapacity, historySize)
            );
        }

        for (uint8_t i = 0; i < numThreads; ++i) {
            m_workers[i]->thread = std::thread(&MCTSThreadPool::workerThread, this, i);
        }

        std::cout << "MCTSThreadPool initialized: " << static_cast<int>(numThreads)
            << " threads, batch=" << batchSize << std::endl;
    }

    ~MCTSThreadPool() {
        m_stopFlag.store(true, std::memory_order_release);
        m_syncCV.notify_all();

        for (auto& worker : m_workers) {
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }
    }

    MCTSThreadPool(const MCTSThreadPool&) = delete;
    MCTSThreadPool& operator=(const MCTSThreadPool&) = delete;

    // ============================================================================
    // PUBLIC INTERFACE
    // ============================================================================

    void clearWorkerCaches()
    {
        // Cette fonction n'a pas besoin d'être thread-safe car elle
        // est appelée par le thread principal PENDANT que les workers sont inactifs.
        for (auto& worker : m_workers)
        {
            if (worker && worker->tld)
            {
                worker->tld->localFreeNodes.clear();
            }
        }
    }

    void executeMCTS(MCTS<GameTag>* mcts, uint32_t numSimulations) {
        if (mcts->m_rootIdx.load(std::memory_order_acquire) == UINT32_MAX) {
            throw std::runtime_error("MCTS root not initialized");
        }

        // Reset all synchronization state
        std::cout << "Resetting thread pool state..." << std::endl;
        m_workersActive.store(0, std::memory_order_release);
        m_workersWaiting.store(0, std::memory_order_release);
        m_workersWithWork.store(0, std::memory_order_release);
        m_resultsReady.store(false, std::memory_order_release);
        m_coordinationInProgress.store(false, std::memory_order_release);
        m_batchId.store(0, std::memory_order_release);

        if (!waitForIdle(10000)) {
            std::cerr << "WARNING: Workers not idle before search!" << std::endl;

            // Force stop
            if (m_currentMCTS) {
                m_currentMCTS->m_searchActive.store(false, std::memory_order_release);
            }
            m_syncCV.notify_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            // Force reset
            m_workersActive.store(0, std::memory_order_release);
            m_workersWaiting.store(0, std::memory_order_release);
            m_workersWithWork.store(0, std::memory_order_release);
            m_resultsReady.store(false, std::memory_order_release);

            if (!waitForIdle(5000)) {
                throw std::runtime_error("Cannot start search: workers stuck");
            }
        }

        // Initialize search
        mcts->m_targetSimulations.store(numSimulations, std::memory_order_release);
        mcts->m_simulationCount.store(0u, std::memory_order_release);
        mcts->cacheRootHistory();

        m_currentMCTS = mcts;

        std::atomic_thread_fence(std::memory_order_seq_cst);
        mcts->m_searchActive.store(true, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        m_syncCV.notify_all();

        // Monitor progress
        uint32_t lastCount = 0;
        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            uint32_t currentCount = mcts->m_simulationCount.load(std::memory_order_acquire);
            if (currentCount >= numSimulations) break;

            if (currentCount != lastCount && currentCount % 500 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - startTime).count();
                float simsPerSec = (elapsedMs > 0) ? (currentCount * 1000.0f / elapsedMs) : 0.0f;

                std::cout << "Progress: " << currentCount << "/" << numSimulations
                    << " (" << static_cast<int>(simsPerSec) << " sims/s)" << std::endl;
                lastCount = currentCount;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Stop search
        std::atomic_thread_fence(std::memory_order_seq_cst);
        mcts->m_searchActive.store(false, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        m_syncCV.notify_all();

        if (!waitForIdle(30000)) {
            std::cerr << "ERROR: Workers did not become idle after search!" << std::endl;
            throw std::runtime_error("Thread pool synchronization failed");
        }

        auto endTime = std::chrono::steady_clock::now();
        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime).count();

        std::cout << "Search complete: " << mcts->m_simulationCount.load()
            << " sims in " << totalMs << "ms "
            << "(" << (totalMs > 0 ? mcts->m_simulationCount.load() * 1000 / totalMs : 0)
            << " sims/s)" << std::endl;
    }

    bool waitForIdle(int timeoutMs = 10000) {
        auto startTime = std::chrono::steady_clock::now();

        std::cout << "Waiting for workers to become idle..." << std::endl;

        while (true) {
            std::atomic_thread_fence(std::memory_order_seq_cst);

            uint32_t active = m_workersActive.load(std::memory_order_seq_cst);
            uint32_t waiting = m_workersWaiting.load(std::memory_order_seq_cst);
            uint32_t withWork = m_workersWithWork.load(std::memory_order_seq_cst);
            bool coordinating = m_coordinationInProgress.load(std::memory_order_acquire);

            // Idle = no workers active, none waiting, none with work, no coordination
            if (active == 0 && waiting == 0 && withWork == 0 && !coordinating) {
                // Double-check
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

                std::atomic_thread_fence(std::memory_order_seq_cst);
                active = m_workersActive.load(std::memory_order_seq_cst);
                waiting = m_workersWaiting.load(std::memory_order_seq_cst);
                withWork = m_workersWithWork.load(std::memory_order_seq_cst);
                coordinating = m_coordinationInProgress.load(std::memory_order_acquire);

                if (active == 0 && waiting == 0 && withWork == 0 && !coordinating) {
                    std::cout << "Workers are now idle." << std::endl;
                    return true;
                }
            }

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - startTime).count();

            if (elapsed > timeoutMs) {
                std::cerr << "ERROR: Timeout after " << timeoutMs << "ms" << std::endl;
                std::cerr << "  Active: " << active << std::endl;
                std::cerr << "  Waiting: " << waiting << std::endl;
                std::cerr << "  WithWork: " << withWork << std::endl;
                std::cerr << "  Coordinating: " << (coordinating ? "yes" : "no") << std::endl;

                // Emergency cleanup
                if (waiting > 0) {
                    std::cerr << "Forcing barrier release..." << std::endl;
                    m_resultsReady.store(true, std::memory_order_release);
                    m_syncCV.notify_all();
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));

                    // Reset everything
                    m_workersWaiting.store(0, std::memory_order_release);
                    m_workersWithWork.store(0, std::memory_order_release);
                    m_resultsReady.store(false, std::memory_order_release);
                }

                return false;
            }

            if (elapsed % 1000 < 20) {
                std::cout << "Still waiting... (Active: " << active
                    << ", Waiting: " << waiting
                    << ", WithWork: " << withWork << ")" << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
};

// ============================================================================
// IMPLEMENTATION OF MCTS::run
// ==================================================== ========================

template<typename GameTag>
void MCTS<GameTag>::run(uint32_t numSimulations, MCTSThreadPool<GameTag>& pool) {
    pool.executeMCTS(this, numSimulations);
}