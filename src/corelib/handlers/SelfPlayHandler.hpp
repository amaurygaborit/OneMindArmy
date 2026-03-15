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

#include "../interfaces/IHandler.hpp"
#include "../model/ReplayBuffer.hpp"
#include "../model/StateEncoder.hpp"

// --- GESTIONNAIRE DE SIGNAL GLOBAL ---
inline std::atomic<bool> g_keepRunning{ true };

inline void sigintHandler(int signal) {
    std::cout << "\n\n[System] Signal CTRL+C détecté ! "
        << "Fermeture propre en cours... (Veuillez patienter pour éviter la corruption des données)\n";
    g_keepRunning.store(false, std::memory_order_release);
}

namespace Core
{
    template<ValidGameTraits GT>
    class SelfPlayHandler : public IHandler<GT>
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        struct GameContext {
            std::array<TreeSearch<GT>*, Defs::kNumPlayers> trees;
            State currentState;
            AlignedVec<Action> actionHistory;
            std::vector<uint64_t> hashHistory;
            ReplayBuffer<GT> replayBuffer;
            uint32_t turnCount = 0;
        };

        BackendConfig  m_backendCfg;
        TrainingConfig m_trainingCfg;
        std::string m_datasetPath = "";

        void specificSetup(const YAML::Node& config) override
        {
            m_backendCfg.load(config, "train");
            m_trainingCfg.load(config, "train");

            if (!config["name"]) {
                throw std::runtime_error("Config Error: Missing 'name' field in YAML.");
            }
            std::string gameName = config["name"].as<std::string>();

            std::string dataFolder = "data/" + gameName;
            std::filesystem::create_directories(dataFolder);
            m_datasetPath = dataFolder + "/" + gameName + "_training_data.bin";
        }

        void resetGame(GameContext& g)
        {
            g.replayBuffer.clear();
            g.actionHistory.clear();
            g.hashHistory.clear();
            g.turnCount = 0;

            this->m_engine->getInitialState(0, g.currentState);
            g.hashHistory.push_back(g.currentState.hash());

            for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                g.trees[p]->startSearch(g.currentState, g.hashHistory);
            }
        }

    public:
        SelfPlayHandler() = default;
        virtual ~SelfPlayHandler() = default;

        void execute() override
        {
            std::signal(SIGINT, sigintHandler);

            std::cout << "[SelfPlay] Starting generation: " << m_trainingCfg.gamesPerIteration << " games\n";
            std::cout << "[SelfPlay] Parallelism: " << m_backendCfg.numParallelGames << " games | Batch: " << m_backendCfg.maxBatchSize << "\n";

            std::vector<GameContext> games(m_backendCfg.numParallelGames);
            size_t treeAllocatorIndex = 0;
            for (auto& g : games) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                    g.trees[p] = this->m_treeSearch[treeAllocatorIndex++].get();
                }
                g.actionHistory.reserve(Defs::kMaxHistory * 2);
                resetGame(g);
            }

            std::ofstream outFile(m_datasetPath, std::ios::binary | std::ios::app);
            if (!outFile.is_open()) {
                throw std::runtime_error("Fatal Error: Could not open dataset file for writing: " + m_datasetPath);
            }

            uint32_t gamesFinished = 0;
            uint64_t totalMovesPlayed = 0; // Passage en uint64 pour plus de sécurité sur de longs runs
            auto startTime = std::chrono::high_resolution_clock::now();

            while (gamesFinished < m_trainingCfg.gamesPerIteration && g_keepRunning.load(std::memory_order_acquire))
            {
                std::vector<TreeSearch<GT>*> activeTrees;
                activeTrees.reserve(m_backendCfg.numParallelGames);
                for (auto& g : games) {
                    uint32_t currentPlayer = this->m_engine->getCurrentPlayer(g.currentState);
                    activeTrees.push_back(g.trees[currentPlayer]);
                }

                this->m_threadPool->executeMultipleTrees(activeTrees, this->m_engineCfg.numSimulations);

                for (auto& g : games)
                {
                    if (gamesFinished >= m_trainingCfg.gamesPerIteration) break;

                    uint32_t currentPlayer = this->m_engine->getCurrentPlayer(g.currentState);
                    TreeSearch<GT>* activeTree = g.trees[currentPlayer];

                    // 1. Record Sample
                    State povState = g.currentState;
                    this->m_engine->changeStatePov(currentPlayer, povState);

                    size_t histSize = std::min<size_t>(g.actionHistory.size(), Defs::kMaxHistory);
                    AlignedVec<Action> povHistory(reserve_only, histSize);
                    for (size_t i = g.actionHistory.size() - histSize; i < g.actionHistory.size(); ++i) {
                        Action a = g.actionHistory[i];
                        this->m_engine->changeActionPov(currentPlayer, a);
                        povHistory.push_back(a);
                    }

                    auto encodedInput = StateEncoder<GT>::encode(povState, povHistory);
                    g.replayBuffer.recordTurn(
                        encodedInput,
                        activeTree->getRootPolicy(),
                        activeTree->getRootLegalMovesMask()
                    );

                    // 2. Play Move
                    g.turnCount++;
                    float temp = (g.turnCount < this->m_engineCfg.temperatureDrop) ? 1.0f : 0.1f;
                    Action selectedAction = activeTree->selectMove(temp);

                    this->m_engine->applyAction(selectedAction, g.currentState);
                    g.hashHistory.push_back(g.currentState.hash());
                    g.actionHistory.push_back(selectedAction);

                    for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                        g.trees[p]->advanceRoot(selectedAction, g.currentState);
                    }

                    totalMovesPlayed++;

                    // 3. Check Endgame
                    std::optional<GameResult> outcome = this->m_engine->getGameResult(g.currentState, g.hashHistory);

                    if (outcome.has_value())
                    {
                        g.replayBuffer.flushToFile(outcome.value(), outFile);
                        gamesFinished++;

                        // Mise à jour de l'affichage toutes les 10 parties
                        if (gamesFinished % 10 == 0 || gamesFinished == m_trainingCfg.gamesPerIteration)
                        {
                            auto now = std::chrono::high_resolution_clock::now();
                            double elapsedSec = std::chrono::duration<double>(now - startTime).count();
                            double movesPerSec = totalMovesPlayed / (elapsedSec > 0 ? elapsedSec : 1.0);

                            // \r permet de réécrire sur la même ligne, \n pour fixer l'étape toutes les 10 parties
                            std::cout << "[SelfPlay] Progress: " << std::setw(6) << gamesFinished << " / " << m_trainingCfg.gamesPerIteration
                                << " games | Total Moves: " << std::setw(8) << totalMovesPlayed
                                << " | Speed: " << std::fixed << std::setprecision(1) << movesPerSec << " m/s" << std::endl;
                        }

                        resetGame(g);
                    }
                }
            }

            outFile.close();
            auto endTime = std::chrono::high_resolution_clock::now();
            double totalSec = std::chrono::duration<double>(endTime - startTime).count();

            std::cout << "\n[SelfPlay] Finished!" << "\n - Games: " << gamesFinished
                << "\n - Total Moves: " << totalMovesPlayed
                << "\n - Time: " << std::fixed << std::setprecision(1) << totalSec << "s"
                << "\n - Avg Speed: " << totalMovesPlayed / (totalSec > 0 ? totalSec : 1.0) << " m/s\n";
        }
    };
}