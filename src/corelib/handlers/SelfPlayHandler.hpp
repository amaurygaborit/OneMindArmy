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
            State                    currentState;
            AlignedVec<Action>       actionHistory;
            std::vector<uint64_t>    hashHistory;
            ReplayBuffer<GT>         replayBuffer;
            uint32_t                 turnCount    = 0;

            // FIX: chaque partie sait si elle a déjà été comptée
            // (i.e. si elle fait partie des gamesPerIteration parties officielles)
            bool                     isOfficial   = false;
        };

        BackendConfig  m_backendCfg;
        TrainingConfig m_trainingCfg;
        std::string    m_datasetPath      = "";
        uint32_t       m_currentIteration = 0;

        void specificSetup(const YAML::Node& config) override
        {
            m_backendCfg.load(config, "train");
            m_trainingCfg.load(config, "train");

            if (!config["name"]) {
                throw std::runtime_error("Config Error: Missing 'name' field in YAML.");
            }
            std::string gameName = config["name"].as<std::string>();

            // Itération courante injectée par Python
            if (config["training"] && config["training"]["currentIteration"]) {
                m_currentIteration = config["training"]["currentIteration"].as<uint32_t>();
            }

            std::string dataFolder = "data/" + gameName;
            std::filesystem::create_directories(dataFolder);

            std::ostringstream oss;
            oss << dataFolder << "/iteration_"
                << std::setw(4) << std::setfill('0') << m_currentIteration << ".bin";
            m_datasetPath = oss.str();
        }

        void resetGame(GameContext& g)
        {
            g.replayBuffer.clear();
            g.actionHistory.clear();
            g.hashHistory.clear();
            g.turnCount  = 0;
            g.isOfficial = false;

            this->m_engine->getInitialState(0, g.currentState);
            g.hashHistory.push_back(g.currentState.hash());

            for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                g.trees[p]->startSearch(g.currentState, g.hashHistory);
            }
        }

        // ------------------------------------------------------------------
        // Détecte une résignation après un coup.
        // Retourne un GameResult peuplé si la résignation est déclenchée,
        // std::nullopt sinon.
        // ------------------------------------------------------------------
        std::optional<GameResult> checkResign(const GameContext& g, uint32_t currentPlayer) const
        {
            // Résignation désactivée si le seuil est hors de [-1, 0]
            if (this->m_engineCfg.resignThreshold < -1.0f || this->m_engineCfg.resignThreshold >= 0.0f)
                return std::nullopt;

            // Pas encore assez de coups joués
            if (g.turnCount < this->m_engineCfg.resignMinPly)
                return std::nullopt;

            // La valeur root est du point de vue du joueur qui VIENT de jouer,
            // donc on interroge son arbre.
            float rootValue = g.trees[currentPlayer]->getRootValue();

            if (rootValue < this->m_engineCfg.resignThreshold)
            {
                // Le joueur courant résigne : il perd (-1), l'autre gagne (+1)
                // On construit le tableau de résultats de façon game-agnostic.
                std::array<float, Defs::kNumPlayers> results{};
                for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                    results[p] = (p == currentPlayer) ? -1.0f : 1.0f;
                }
                return GameResult{ results, 7 };
            }

            return std::nullopt;
        }

    public:
        SelfPlayHandler()          = default;
        virtual ~SelfPlayHandler() = default;

        void execute() override
        {
            std::signal(SIGINT, sigintHandler);

            const uint32_t target = m_trainingCfg.gamesPerIteration;

            std::cout << "[SelfPlay] Starting generation: " << target << " games\n";
            std::cout << "[SelfPlay] Iteration File    : " << m_datasetPath << "\n";
            std::cout << "[SelfPlay] Parallelism       : "
                      << m_backendCfg.numParallelGames << " games | Batch: "
                      << m_backendCfg.inferenceBatchSize << "\n";

            if (this->m_engineCfg.resignThreshold >= -1.0f && this->m_engineCfg.resignThreshold < 0.0f) {
                std::cout << "[SelfPlay] Resign            : threshold="
                          << this->m_engineCfg.resignThreshold << "  minPly=" << this->m_engineCfg.resignMinPly << "\n";
            } else {
                std::cout << "[SelfPlay] Resign            : disabled\n";
            }

            // ------------------------------------------------------------------
            // Allocation des contextes de parties
            // ------------------------------------------------------------------
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
                throw std::runtime_error(
                    "Fatal Error: Could not open dataset file for writing: " + m_datasetPath);
            }

            // FIX: compteurs séparés
            uint32_t gamesFinished  = 0;  // parties officielles terminées (≤ target)
            uint64_t officialMoves  = 0;  // coups des parties officielles uniquement
            uint64_t totalMoves     = 0;  // tous les coups (pour le débit brut GPU)

            auto startTime = std::chrono::high_resolution_clock::now();

            while (gamesFinished < target &&
                   g_keepRunning.load(std::memory_order_acquire))
            {
                // ----------------------------------------------------------
                // 1. Lancement des simulations MCTS pour toutes les parties
                //    actives (qu'elles soient officielles ou non — on les
                //    laisse tourner pour remplir le batch GPU efficacement)
                // ----------------------------------------------------------
                std::vector<TreeSearch<GT>*> activeTrees;
                activeTrees.reserve(m_backendCfg.numParallelGames);
                for (auto& g : games) {
                    uint32_t cp = this->m_engine->getCurrentPlayer(g.currentState);
                    activeTrees.push_back(g.trees[cp]);
                }
                this->m_threadPool->executeMultipleTrees(
                    activeTrees, this->m_engineCfg.numSimulations);

                // ----------------------------------------------------------
                // 2. Traitement de chaque partie
                // ----------------------------------------------------------
                for (auto& g : games)
                {
                    // On ne s'arrête plus de traiter les parties non-officielles :
                    // elles doivent continuer à tourner pour garder le batch GPU plein.
                    // On ne flush PAS leur résultat sur disque.

                    uint32_t currentPlayer = this->m_engine->getCurrentPlayer(g.currentState);
                    TreeSearch<GT>* activeTree = g.trees[currentPlayer];

                    // --- Enregistrement du sample (seulement si officielle) ---
                    if (g.isOfficial)
                    {
                        State povState = g.currentState;
                        this->m_engine->changeStatePov(currentPlayer, povState);

                        size_t histSize = std::min<size_t>(
                            g.actionHistory.size(), Defs::kMaxHistory);

                        AlignedVec<Action> povHistory(reserve_only, histSize);
                        for (size_t i = g.actionHistory.size() - histSize;
                             i < g.actionHistory.size(); ++i)
                        {
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
                    }

                    // --- Sélection et application du coup ---
                    g.turnCount++;
                    float temp = (g.turnCount < this->m_engineCfg.temperatureDrop)
                                 ? 1.0f : 0.1f;
                    Action selectedAction = activeTree->selectMove(temp);

                    this->m_engine->applyAction(selectedAction, g.currentState);
                    g.hashHistory.push_back(g.currentState.hash());
                    g.actionHistory.push_back(selectedAction);

                    for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                        g.trees[p]->advanceRoot(selectedAction, g.currentState);
                    }

                    // FIX: on compte TOUS les coups pour le débit GPU...
                    totalMoves++;
                    // ...mais seulement les coups des parties officielles pour les stats
                    if (g.isOfficial) officialMoves++;

                    // --- Détection de fin de partie ---
                    std::optional<GameResult> outcome =
                        this->m_engine->getGameResult(g.currentState, g.hashHistory);

                    // Résignation (seulement si pas déjà terminé par une règle)
                    if (!outcome.has_value()) {
                        outcome = checkResign(g, currentPlayer);
                    }

                    if (outcome.has_value())
                    {
                        if (g.isOfficial)
                        {
                            // Partie officielle → flush sur disque
                            g.replayBuffer.flushToFile(outcome.value(), outFile);
                            gamesFinished++;

                            if (gamesFinished % 10 == 0 || gamesFinished == target)
                            {
                                auto now = std::chrono::high_resolution_clock::now();
                                double elapsed = std::chrono::duration<double>(
                                    now - startTime).count();
                                double movesPerSec = totalMoves /
                                    (elapsed > 0.0 ? elapsed : 1.0);
                                double avgLen = (gamesFinished > 0)
                                    ? static_cast<double>(officialMoves) / gamesFinished
                                    : 0.0;

                                std::cout
                                    << "[SelfPlay] "
                                    << std::setw(6) << gamesFinished << " / " << target
                                    << " games | AvgLen: "
                                    << std::fixed << std::setprecision(1) << avgLen
                                    << " ply | Speed: "
                                    << std::fixed << std::setprecision(1) << movesPerSec
                                    << " m/s\r" << std::flush;
                            }
                        }
                        // Qu'elle soit officielle ou non, on remet la partie à zéro.
                        // La prochaine partie qui démarre sera officielle seulement
                        // si le quota n'est pas encore atteint.
                        resetGame(g);

                        // FIX: la nouvelle partie est officielle seulement si on n'a
                        // pas encore atteint le quota. On vérifie APRÈS l'incrément
                        // de gamesFinished pour éviter une partie de trop.
                        g.isOfficial = (gamesFinished < target);
                    }
                }
            }

            outFile.close();

            auto endTime   = std::chrono::high_resolution_clock::now();
            double totalSec = std::chrono::duration<double>(endTime - startTime).count();

            int hours   = static_cast<int>(totalSec) / 3600;
            int minutes = (static_cast<int>(totalSec) % 3600) / 60;
            int seconds = static_cast<int>(totalSec) % 60;

            double avgLen = (gamesFinished > 0)
                ? static_cast<double>(officialMoves) / gamesFinished : 0.0;

            std::cout << "\n\n[SelfPlay] Finished!"
                << "\n  Games (official)  : " << gamesFinished
                << "\n  Official Moves    : " << officialMoves
                << "\n  Avg Game Length   : " << std::fixed << std::setprecision(1)
                                              << avgLen << " ply"
                << "\n  Total Moves (GPU) : " << totalMoves
                << "\n  Total Time        : "
                    << std::setfill('0') << std::setw(2) << hours   << "h "
                    << std::setw(2)      << minutes                  << "m "
                    << std::setw(2)      << seconds                  << "s"
                << "\n  Avg Speed (GPU)   : "
                    << std::fixed << std::setprecision(1)
                    << (totalMoves / (totalSec > 0.0 ? totalSec : 1.0)) << " moves/s\n";
        }
    };
}