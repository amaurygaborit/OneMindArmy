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
        uint32_t m_currentIteration = 0;

        void specificSetup(const YAML::Node& config) override
        {
            m_backendCfg.load(config, "train");
            m_trainingCfg.load(config, "train");

            if (!config["name"]) {
                throw std::runtime_error("Config Error: Missing 'name' field in YAML.");
            }
            std::string gameName = config["name"].as<std::string>();

            // Extraction de l'itération courante (Gérée et injectée par Python)
            if (config["training"] && config["training"]["currentIteration"]) {
                m_currentIteration = config["training"]["currentIteration"].as<uint32_t>();
            }

            std::string dataFolder = "data/" + gameName;
            std::filesystem::create_directories(dataFolder);

            // NOUVEAU: Formatage dynamique du fichier par itération (ex: iteration_0042.bin)
            std::ostringstream oss;
            oss << dataFolder << "/iteration_" << std::setw(4) << std::setfill('0') << m_currentIteration << ".bin";
            m_datasetPath = oss.str();
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
            std::cout << "[SelfPlay] Iteration File: " << m_datasetPath << "\n";
            // On utilise la nouvelle nomenclature inferenceBatchSize
            std::cout << "[SelfPlay] Parallelism: " << m_backendCfg.numParallelGames << " games | Batch: " << m_backendCfg.inferenceBatchSize << "\n";

            std::vector<GameContext> games(m_backendCfg.numParallelGames);
            size_t treeAllocatorIndex = 0;
            for (auto& g : games) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                    g.trees[p] = this->m_treeSearch[treeAllocatorIndex++].get();
                }
                g.actionHistory.reserve(Defs::kMaxHistory * 2);
                resetGame(g);
            }

            // Ouverture du fichier en append. Si le fichier n'existe pas, il sera créé.
            std::ofstream outFile(m_datasetPath, std::ios::binary | std::ios::app);
            if (!outFile.is_open()) {
                throw std::runtime_error("Fatal Error: Could not open dataset file for writing: " + m_datasetPath);
            }

            uint32_t gamesFinished = 0;
            uint64_t totalMovesPlayed = 0;
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

                    // Envoi vers le ReplayBuffer avec le bit-packing !
                    g.replayBuffer.recordTurn(
                        encodedInput,
                        activeTree->getRootPolicy(),
                        activeTree->getRootLegalMovesMask() // Retourne bien le array<bool>
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

                            std::cout << "[SelfPlay] Progress: " << std::setw(6) << gamesFinished << " / " << m_trainingCfg.gamesPerIteration
                                << " games | Total Moves: " << std::setw(8) << totalMovesPlayed
                                << " | Speed: " << std::fixed << std::setprecision(1) << movesPerSec << " m/s\r" << std::flush;
                        }

                        resetGame(g);
                    }
                }
            }

            outFile.close();

            // Calcul du temps total
            auto endTime = std::chrono::high_resolution_clock::now();
            double totalSec = std::chrono::duration<double>(endTime - startTime).count();

            int hours = static_cast<int>(totalSec) / 3600;
            int minutes = (static_cast<int>(totalSec) % 3600) / 60;
            int seconds = static_cast<int>(totalSec) % 60;

            std::cout << "\n\n[SelfPlay] Finished!"
                << "\n - Games: " << gamesFinished
                << "\n - Total Moves: " << totalMovesPlayed
                << "\n - Total Time: " << std::setfill('0') << std::setw(2) << hours << "h "
                << std::setw(2) << minutes << "m "
                << std::setw(2) << seconds << "s"
                << "\n - Avg Speed: " << std::fixed << std::setprecision(1) << (totalMovesPlayed / (totalSec > 0 ? totalSec : 1.0)) << " moves/s\n";
        }
    };
}