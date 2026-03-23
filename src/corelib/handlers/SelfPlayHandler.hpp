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

// Atomic flag for safe interruption (Ctrl+C)
inline std::atomic<bool> g_keepRunning{ true };

inline void sigintHandler(int) {
    std::cout << "\n\n[System] SIGINT caught. Finalizing buffers and exiting...\n";
    g_keepRunning.store(false, std::memory_order_release);
}

namespace Core
{
    /**
     * @brief High-Throughput Self-Play Handler.
     * * Orchestrates parallel game generation using MCTS and Neural Network inference.
     * Manages data serialization into binary datasets for training.
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

            std::cout << "[Info] Dataset target: " << m_datasetPath << std::endl;
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
         * @brief Renders a clean, fixed-position ANSI dashboard.
         */
        void printDashboard(uint32_t written, uint32_t target, uint32_t played, uint64_t totalMoves,
            double elapsedSec, const GameContext& sampleGame) const
        {
            const double mps = totalMoves / std::max(elapsedSec, 1e-6);
            const double progress = (static_cast<double>(written) / target) * 100.0;

            // ANSI: Move cursor up 11 lines to overwrite
            static bool firstDraw = true;
            if (!firstDraw) std::cout << "\033[11F";
            firstDraw = false;

            auto* currentTree = sampleGame.trees[this->m_engine->getCurrentPlayer(sampleGame.currentState)];

            std::cout << "\033[K" << "======================== ENGINE SELF-PLAY ========================\n";
            std::cout << "\033[K" << " [PROGRESS]  " << std::setw(6) << written << " / " << target
                << " games (" << std::fixed << std::setprecision(1) << progress << "%)\n";

            std::cout << "\033[K" << " [THROUGHPUT] " << std::setw(6) << (int)mps << " m/s | "
                << "Elapsed: " << (int)elapsedSec / 3600 << "h " << ((int)elapsedSec % 3600) / 60 << "m "
                << (int)elapsedSec % 60 << "s\n";

            std::cout << "\033[K" << "-------------------------- TREE METRICS --------------------------\n";
            std::cout << "\033[K" << " Root Q: " << std::fixed << std::setw(6) << std::setprecision(3) << currentTree->getRootValue()
                << " | SimCount: " << std::setw(6) << currentTree->getSimulationCount()
                << " | Mem: " << std::setw(3) << (int)(currentTree->getMemoryUsage() * 100) << "%\n";

            std::cout << "\033[K" << "------------------------- PIPELINE STATE -------------------------\n";
            std::cout << "\033[K" << " Queue Ready: " << std::setw(4) << this->m_threadPool->getReadyQueueSize()
                << " | Eval: " << std::setw(4) << this->m_threadPool->getEvalQueueSize()
                << " | Back: " << std::setw(4) << this->m_threadPool->getBackpropQueueSize() << "\n";

            std::cout << "\033[K" << " Inference  : " << std::setw(4) << m_backendCfg.inferenceBatchSize
                << " (Batch) | Free Contexts: " << this->m_threadPool->getFreeEventCount() << "\n";
            std::cout << "\033[K" << "==================================================================\n" << std::flush;
        }

    public:
        SelfPlayHandler() = default;
        virtual ~SelfPlayHandler() = default;

        void execute() override {
            std::signal(SIGINT, sigintHandler);
            const uint32_t target = m_trainingCfg.gamesPerIteration;

            // Initialization
            std::vector<GameContext> games(m_backendCfg.numParallelGames);
            size_t treeAllocIdx = 0;
            for (auto& g : games) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                    g.trees[p] = this->m_treeSearch[treeAllocIdx++].get();
                g.actionHistory.reserve(Defs::kMaxHistory * 2);
                resetGame(g);
            }

            // Flag games contributing to the dataset
            for (size_t i = 0; i < games.size() && i < static_cast<size_t>(target); ++i)
                games[i].isOfficial = true;

            std::ofstream outFile(m_datasetPath, std::ios::binary | std::ios::app);
            if (!outFile.is_open()) throw std::runtime_error("IO Error: Could not open dataset for writing.");

            uint32_t gamesWritten = 0, gamesPlayed = 0;
            uint64_t totalMoves = 0, totalSamples = 0, totalPliesWritten = 0;
            auto startTime = std::chrono::high_resolution_clock::now();

            // Print initial padding for dashboard
            std::cout << std::string(11, '\n');

            while (gamesWritten < target && g_keepRunning.load(std::memory_order_acquire)) {

                // 1. Batch Tree Search
                std::vector<TreeSearch<GT>*> activeTrees;
                for (auto& g : games) {
                    activeTrees.push_back(g.trees[this->m_engine->getCurrentPlayer(g.currentState)]);
                }
                this->m_threadPool->executeMultipleTrees(activeTrees, this->m_engineCfg.numSimulations);

                // 2. State Advancement
                for (auto& g : games) {
                    const uint32_t cp = this->m_engine->getCurrentPlayer(g.currentState);
                    TreeSearch<GT>* activeTree = g.trees[cp];

                    if (g.isOfficial) {
                        // Data Encoding for Replay Buffer
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

                        g.replayBuffer.recordTurn(encoded, activeTree->getRootPolicy(), activeTree->getRootLegalMovesMask(), cp);
                    }

                    // Select and apply action
                    float temp = (g.turnCount < this->m_engineCfg.temperatureDropTurn) ? 1.0f : 0.0f;
                    const Action action = activeTree->selectMove(temp);

                    g.turnCount++;
                    totalMoves++;
                    this->m_engine->applyAction(action, g.currentState);
                    g.hashHistory.push_back(g.currentState.hash());
                    g.actionHistory.push_back(action);

                    for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                        g.trees[p]->advanceRoot(action, g.currentState);

                    // 3. Termination Check
                    auto outcome = this->m_engine->getGameResult(g.currentState, g.hashHistory);

                    // Resignation Logic
                    if (!outcome && g.turnCount > this->m_engineCfg.resignMinPly) {
                        if (activeTree->getRootValue() < this->m_engineCfg.resignThreshold)
                            outcome = this->m_engine->buildResignResult(cp);
                    }

                    if (outcome) {
                        if (g.isOfficial && gamesWritten < target) {
                            gamesPlayed++;
                            size_t samples = g.replayBuffer.size();
                            if (g.replayBuffer.flushToFile(*outcome, outFile, m_trainingCfg.drawSampleRate)) {
                                totalPliesWritten += g.turnCount;
                                totalSamples += samples;
                                gamesWritten++;
                            }
                        }
                        resetGame(g);
                        g.isOfficial = (gamesWritten < target);
                    }
                }

                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - startTime).count();
                printDashboard(gamesWritten, target, gamesPlayed, totalMoves, elapsed, games[0]);
            }

            // Summary report
            double totalTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
            std::cout << "\n" << std::string(60, '=') << "\n";
            std::cout << " [COMPLETED] Self-Play Session Finished\n";
            std::cout << "  - Total Samples     : " << totalSamples << "\n";
            std::cout << "  - Games Saved       : " << gamesWritten << "\n";
            std::cout << "  - Avg Game Length   : " << (gamesWritten > 0 ? (double)totalPliesWritten / gamesWritten : 0.0) << " plies\n";
            std::cout << "  - Global Throughput : " << std::fixed << std::setprecision(1) << (totalMoves / totalTime) << " m/s\n";
            std::cout << std::string(60, '=') << std::endl;
        }
    };
}