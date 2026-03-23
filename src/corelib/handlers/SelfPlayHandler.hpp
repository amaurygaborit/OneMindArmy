#pragma once

#include <chrono>
#include <optional>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <span>
#include <algorithm>
#include <filesystem>
#include <string>
#include <csignal>
#include <atomic>
#include <sstream>

#include "../interfaces/IHandler.hpp"
#include "../model/ReplayBuffer.hpp"
#include "../model/StateEncoder.hpp"

// Global atomic for graceful shutdown
inline std::atomic<bool> g_keepRunning{ true };

inline void sigintHandler(int) {
    std::cout << "\n\n[System] SIGINT detected. Finalizing writes and shutting down...\n";
    g_keepRunning.store(false, std::memory_order_release);
}

namespace Core
{
    /**
     * @brief High-performance Self-Play handler.
     * Manages asynchronous game generation, MCTS tree orchestration, and training data serialization.
     * Optimized for multi-core CPUs and high-throughput GPUs.
     */
    template<ValidGameTraits GT>
    class SelfPlayHandler : public IHandler<GT>
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        struct GameContext {
            std::array<TreeSearch<GT>*, Defs::kNumPlayers> trees;
            State                  currentState;
            AlignedVec<Action>     actionHistory;
            std::vector<uint64_t>  hashHistory;
            ReplayBuffer<GT>       replayBuffer;
            uint32_t               turnCount = 0;
            bool                   isOfficial = false; // Contributes to the iteration quota
        };

        BackendConfig  m_backendCfg;
        TrainingConfig m_trainingCfg;
        std::string    m_datasetPath = "";

        void specificSetup(const YAML::Node& config) override {
            m_backendCfg.load(config, "train");
            m_trainingCfg.load(config, "train");

            const std::string gameName = config["name"].as<std::string>();
            const std::string dataFolder = "data/" + gameName;
            std::filesystem::create_directories(dataFolder);

            std::ostringstream oss;
            oss << dataFolder << "/iteration_" << std::setw(4) << std::setfill('0')
                << m_trainingCfg.currentIteration << ".bin";
            m_datasetPath = oss.str();
        }

        void resetGame(GameContext& g) {
            g.replayBuffer.clear();
            g.actionHistory.clear();
            g.hashHistory.clear();
            g.turnCount = 0;
            g.isOfficial = false;

            this->m_engine->getInitialState(0, g.currentState);
            g.hashHistory.push_back(g.currentState.hash());

            for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                g.trees[p]->startSearch(g.currentState, g.hashHistory);
        }

        /**
         * @brief Renders a dynamic ANSI dashboard to the terminal.
         * Provides real-time insights into MCTS health and ThreadPool saturation.
         */
        void printDashboard(uint32_t written, uint32_t target, uint32_t played, uint64_t samples,
            uint64_t moves, double sec, const GameContext& sampleGame) const
        {
            const double speed = moves / std::max(sec, 1e-6);
            const double filterRatio = (written > 0) ? static_cast<double>(played) / written : 1.0;

            // Move cursor up 11 lines to overwrite the previous block
            static bool initialized = false;
            if (initialized) std::cout << "\033[11F";
            initialized = true;

            auto* currentTree = sampleGame.trees[this->m_engine->getCurrentPlayer(sampleGame.currentState)];

            std::cout << "\033[K" << "======================= SELF-PLAY DASHBOARD =======================\n";
            std::cout << "\033[K" << " [PROGRESS] Games: " << written << " / " << target
                << " | Samples: " << samples << " | Filter: x" << std::fixed << std::setprecision(1) << filterRatio << "\n";
            std::cout << "\033[K" << " [METRICS]  Speed: " << std::setprecision(1) << speed << " m/s"
                << " | Elapsed: " << (int)sec / 3600 << "h " << ((int)sec % 3600) / 60 << "m " << (int)sec % 60 << "s\n";

            std::cout << "\033[K" << "-------------------------- TREE STATS -----------------------------\n";
            std::cout << "\033[K" << " Q-Value: " << std::setw(6) << std::setprecision(3) << currentTree->getRootValue()
                << " | Mem: " << std::setw(3) << (int)(currentTree->getMemoryUsage() * 100) << "%"
                << " | Sims: " << currentTree->getSimulationCount() << "\n";

            std::cout << "\033[K" << "----------------------- THREADPOOL STATE --------------------------\n";
            std::cout << "\033[K" << " Queue Ready: " << std::setw(4) << this->m_threadPool->getReadyQueueSize()
                << " | Eval: " << std::setw(4) << this->m_threadPool->getEvalQueueSize()
                << " | Back: " << std::setw(4) << this->m_threadPool->getBackpropQueueSize() << "\n";
            std::cout << "\033[K" << " Batch Size : " << std::setw(4) << m_backendCfg.inferenceBatchSize
                << " | Free Contexts: " << this->m_threadPool->getFreeEventCount() << "\n";
            std::cout << "\033[K" << "===================================================================\n" << std::flush;
        }

    public:
        SelfPlayHandler() = default;
        virtual ~SelfPlayHandler() = default;

        void execute() override {
            std::signal(SIGINT, sigintHandler);
            const uint32_t target = m_trainingCfg.gamesPerIteration;

            // Allocate game contexts based on parallel configuration
            std::vector<GameContext> games(m_backendCfg.numParallelGames);
            size_t treeAllocIdx = 0;
            for (auto& g : games) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                    g.trees[p] = this->m_treeSearch[treeAllocIdx++].get();
                g.actionHistory.reserve(Defs::kMaxHistory * 2);
                resetGame(g);
            }

            // Designate games that contribute to the iteration dataset
            for (size_t i = 0; i < games.size() && i < static_cast<size_t>(target); ++i)
                games[i].isOfficial = true;

            std::ofstream outFile(m_datasetPath, std::ios::binary | std::ios::app);
            if (!outFile.is_open()) throw std::runtime_error("Critical: Failed to open dataset file.");

            uint32_t gamesWritten = 0, gamesPlayed = 0;
            uint64_t movesWritten = 0, totalGpuMoves = 0, totalSamples = 0;
            auto startTime = std::chrono::high_resolution_clock::now();

            // Initial line buffer for the dashboard
            std::cout << "\n\n\n\n\n\n\n\n\n\n\n";

            while (gamesWritten < target && g_keepRunning.load(std::memory_order_acquire)) {
                std::vector<TreeSearch<GT>*> activeTrees;
                for (auto& g : games) {
                    uint32_t cp = this->m_engine->getCurrentPlayer(g.currentState);
                    activeTrees.push_back(g.trees[cp]);
                }

                // Orchestrate multi-threaded search across all active games
                this->m_threadPool->executeMultipleTrees(activeTrees, this->m_engineCfg.numSimulations);

                for (auto& g : games) {
                    const uint32_t cp = this->m_engine->getCurrentPlayer(g.currentState);
                    TreeSearch<GT>* activeTree = g.trees[cp];

                    if (g.isOfficial) {
                        State povState = g.currentState;
                        this->m_engine->changeStatePov(cp, povState);

                        const size_t histSize = std::min<size_t>(g.actionHistory.size(), Defs::kMaxHistory);
                        StaticVec<Action, Defs::kMaxHistory> povHistory;

                        for (size_t i = g.actionHistory.size() - histSize; i < g.actionHistory.size(); ++i) {
                            Action a = g.actionHistory[i];
                            this->m_engine->changeActionPov(cp, a);
                            povHistory.push_back(a);
                        }

                        std::array<float, Defs::kNNInputSize> encoded;
                        StateEncoder<GT>::encode(povState, povHistory, encoded);

                        g.replayBuffer.recordTurn(
                            encoded,
                            activeTree->getRootPolicy(),
                            activeTree->getRootLegalMovesMask(),
                            cp
                        );
                    }

                    // Move selection logic (Temperature-based exploration)
                    float temp = (g.turnCount < this->m_engineCfg.temperatureDropTurn) ? 1.0f : 0.0f;
                    const Action action = activeTree->selectMove(temp);

                    g.turnCount++;
                    this->m_engine->applyAction(action, g.currentState);
                    g.hashHistory.push_back(g.currentState.hash());
                    g.actionHistory.push_back(action);

                    for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                        g.trees[p]->advanceRoot(action, g.currentState);

                    totalGpuMoves++;

                    // Check for terminal conditions or resignation
                    auto outcome = this->m_engine->getGameResult(g.currentState, g.hashHistory);
                    if (!outcome && g.turnCount > this->m_engineCfg.resignMinPly) {
                        if (activeTree->getRootValue() < this->m_engineCfg.resignThreshold)
                            outcome = this->m_engine->buildResignResult(cp);
                    }

                    if (outcome) {
                        if (g.isOfficial && gamesWritten < target) {
                            gamesPlayed++;
                            size_t samplesInGame = g.replayBuffer.size();
                            if (g.replayBuffer.flushToFile(*outcome, outFile, m_trainingCfg.drawSampleRate)) {
                                movesWritten += g.turnCount;
                                totalSamples += samplesInGame;
                                gamesWritten++;
                            }
                        }
                        resetGame(g);
                        g.isOfficial = (gamesWritten < target);
                    }
                }

                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - startTime).count();
                printDashboard(gamesWritten, target, gamesPlayed, totalSamples, totalGpuMoves, elapsed, games[0]);
            }

            // --- FINAL SESSION SUMMARY ---
            double totalSeconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();

            std::cout << "\n" << std::string(60, '=') << "\n";
            std::cout << " [SESSION SUMMARY]\n";
            std::cout << "  - Total Samples Generated : " << totalSamples << "\n";
            std::cout << "  - Games Saved to Disk     : " << gamesWritten << "\n";
            std::cout << "  - Data Filtering Ratio    : " << std::fixed << std::setprecision(2)
                << (static_cast<double>(gamesPlayed) / std::max(1u, gamesWritten)) << "x\n";
            std::cout << "  - Mean Game Length        : " << (gamesWritten > 0 ? (double)movesWritten / gamesWritten : 0.0) << " plies\n";
            std::cout << "  - Average Throughput      : " << std::fixed << std::setprecision(1) << (totalGpuMoves / totalSeconds) << " nodes/s\n";
            std::cout << "  - Execution Time          : " << (int)totalSeconds / 3600 << "h " << ((int)totalSeconds % 3600) / 60 << "m\n";
            std::cout << std::string(60, '=') << std::endl;

        }
    };
}