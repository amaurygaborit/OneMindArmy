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

inline std::atomic<bool> g_keepRunning{ true };

inline void sigintHandler(int signal) {
    std::cout << "\n\n[System] Signal CTRL+C détecté ! Fermeture propre en cours...\n";
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
            uint32_t                 turnCount = 0;
            bool                     isOfficial = false;
        };

        BackendConfig  m_backendCfg;
        TrainingConfig m_trainingCfg;
        std::string    m_datasetPath = "";

        void specificSetup(const YAML::Node& config) override
        {
            m_backendCfg.load(config, "train");
            m_trainingCfg.load(config, "train");

            if (!config["name"])
                throw std::runtime_error("Config Error: Missing 'name' field in YAML.");

            std::string gameName = config["name"].as<std::string>();
            std::string dataFolder = "data/" + gameName;
            std::filesystem::create_directories(dataFolder);

            std::ostringstream oss;
            oss << dataFolder << "/iteration_"
                << std::setw(4) << std::setfill('0') << m_trainingCfg.currentIteration << ".bin";
            m_datasetPath = oss.str();
        }

        void resetGame(GameContext& g)
        {
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

        // ------------------------------------------------------------------
        // Résignation — game-agnostic.
        //
        // FIX : suppression du format WDL (kNumPlayers * 3) — migration
        // partielle qui causait un buffer overread dans ThreadPool::loopInference
        // (copy_n de 6 éléments depuis un array de 2) → crash silencieux du
        // thread d'inférence → executeMultipleTrees bloqué indéfiniment.
        //
        // On délègue la construction du GameResult à buildResignResult()
        // défini dans le moteur — format scalaire simple par joueur [-1, +1].
        // ------------------------------------------------------------------
        std::optional<GameResult> checkResign(const GameContext& g,
            uint32_t           currentPlayer) const
        {
            const float    threshold = this->m_engineCfg.resignThreshold;
            const uint32_t minPly = this->m_engineCfg.resignMinPly;

            if (threshold <= -1.0f || threshold >= 0.0f)
                return std::nullopt;

            if (g.turnCount < minPly)
                return std::nullopt;

            if (g.trees[currentPlayer]->getRootValue() < threshold)
                return this->m_engine->buildResignResult(currentPlayer);

            return std::nullopt;
        }

    public:
        SelfPlayHandler() = default;
        virtual ~SelfPlayHandler() = default;

        void execute() override
        {
            std::signal(SIGINT, sigintHandler);

            const uint32_t target = m_trainingCfg.gamesPerIteration;

            std::cout << "[SelfPlay] Starting  : " << target << " games\n"
                << "[SelfPlay] File       : " << m_datasetPath << "\n"
                << "[SelfPlay] Parallelism: " << m_backendCfg.numParallelGames
                << " games | Batch: " << m_backendCfg.inferenceBatchSize << "\n";

            const float thr = this->m_engineCfg.resignThreshold;
            if (thr > -1.0f && thr < 0.0f)
                std::cout << "[SelfPlay] Resign     : threshold=" << thr
                << "  minPly=" << this->m_engineCfg.resignMinPly << "\n";
            else
                std::cout << "[SelfPlay] Resign     : disabled\n";

            std::vector<GameContext> games(m_backendCfg.numParallelGames);
            size_t treeAllocIdx = 0;
            for (auto& g : games) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                    g.trees[p] = this->m_treeSearch[treeAllocIdx++].get();
                g.actionHistory.reserve(Defs::kMaxHistory * 2);
                resetGame(g);
            }

            std::ofstream outFile(m_datasetPath, std::ios::binary | std::ios::app);
            if (!outFile.is_open())
                throw std::runtime_error("Fatal: Cannot open dataset file: " + m_datasetPath);

            uint32_t gamesFinished = 0;
            uint64_t officialMoves = 0;
            uint64_t totalMoves = 0;
            auto     startTime = std::chrono::high_resolution_clock::now();

            while (gamesFinished < target &&
                g_keepRunning.load(std::memory_order_acquire))
            {
                std::vector<TreeSearch<GT>*> activeTrees;
                activeTrees.reserve(m_backendCfg.numParallelGames);
                for (auto& g : games) {
                    uint32_t cp = this->m_engine->getCurrentPlayer(g.currentState);
                    activeTrees.push_back(g.trees[cp]);
                }
                this->m_threadPool->executeMultipleTrees(
                    activeTrees, this->m_engineCfg.numSimulations);

                for (auto& g : games)
                {
                    uint32_t        cp = this->m_engine->getCurrentPlayer(g.currentState);
                    TreeSearch<GT>* activeTree = g.trees[cp];

                    if (g.isOfficial)
                    {
                        State povState = g.currentState;
                        this->m_engine->changeStatePov(cp, povState);

                        size_t histSize = std::min<size_t>(
                            g.actionHistory.size(), Defs::kMaxHistory);
                        AlignedVec<Action> povHistory(reserve_only, histSize);
                        for (size_t i = g.actionHistory.size() - histSize;
                            i < g.actionHistory.size(); ++i)
                        {
                            Action a = g.actionHistory[i];
                            this->m_engine->changeActionPov(cp, a);
                            povHistory.push_back(a);
                        }

                        g.replayBuffer.recordTurn(
                            StateEncoder<GT>::encode(povState, povHistory),
                            activeTree->getRootPolicy(),
                            activeTree->getRootLegalMovesMask()
                        );
                    }

                    g.turnCount++;
                    float  temp = (g.turnCount < this->m_engineCfg.temperatureDrop) ? 1.0f : 0.1f;
                    Action action = activeTree->selectMove(temp);

                    this->m_engine->applyAction(action, g.currentState);
                    g.hashHistory.push_back(g.currentState.hash());
                    g.actionHistory.push_back(action);

                    for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                        g.trees[p]->advanceRoot(action, g.currentState);

                    totalMoves++;
                    if (g.isOfficial) officialMoves++;

                    std::optional<GameResult> outcome =
                        this->m_engine->getGameResult(g.currentState, g.hashHistory);

                    if (!outcome.has_value())
                        outcome = checkResign(g, cp);

                    if (outcome.has_value())
                    {
                        if (g.isOfficial)
                        {
                            g.replayBuffer.flushToFile(outcome.value(), outFile);
                            gamesFinished++;

                            if (gamesFinished % 10 == 0 || gamesFinished == target)
                            {
                                auto   now = std::chrono::high_resolution_clock::now();
                                double elapsed = std::chrono::duration<double>(now - startTime).count();
                                double avgLen = gamesFinished > 0
                                    ? static_cast<double>(officialMoves) / gamesFinished : 0.0;

                                std::cout << "[SelfPlay] " << std::setw(6) << gamesFinished
                                    << " / " << target << " games | AvgLen: "
                                    << std::fixed << std::setprecision(1) << avgLen
                                    << " ply | Speed: "
                                    << (totalMoves / std::max(elapsed, 1e-6))
                                    << " m/s\r" << std::flush;
                            }
                        }
                        resetGame(g);
                        g.isOfficial = (gamesFinished < target);
                    }
                }
            }

            outFile.close();

            auto   end = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(end - startTime).count();
            int    hh = static_cast<int>(sec) / 3600;
            int    mm = (static_cast<int>(sec) % 3600) / 60;
            int    ss = static_cast<int>(sec) % 60;
            double avg = gamesFinished > 0
                ? static_cast<double>(officialMoves) / gamesFinished : 0.0;

            std::cout << "\n\n[SelfPlay] Finished!"
                << "\n  Games (official)  : " << gamesFinished
                << "\n  Avg Game Length   : " << std::fixed << std::setprecision(1) << avg << " ply"
                << "\n  Total Moves (GPU) : " << totalMoves
                << "\n  Total Time        : "
                << std::setfill('0') << std::setw(2) << hh << "h "
                << std::setw(2) << mm << "m "
                << std::setw(2) << ss << "s"
                << "\n  Avg Speed (GPU)   : "
                << (totalMoves / std::max(sec, 1e-6)) << " moves/s\n";
        }
    };
}