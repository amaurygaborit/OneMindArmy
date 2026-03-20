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

inline void sigintHandler(int /*signal*/) {
    std::cout << "\n\n[System] CTRL+C detected. Graceful shutdown in progress...\n";
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
            State                 currentState;
            AlignedVec<Action>    actionHistory;
            std::vector<uint64_t> hashHistory;
            ReplayBuffer<GT>      replayBuffer;
            uint32_t              turnCount = 0;
            bool                  isOfficial = false;
        };

        BackendConfig  m_backendCfg;
        TrainingConfig m_trainingCfg;
        std::string    m_datasetPath = "";
        float          m_drawSampleRate = 1.0f;

        void specificSetup(const YAML::Node& config) override
        {
            m_backendCfg.load(config, "train");
            m_trainingCfg.load(config, "train");

            if (!config["name"])
                throw std::runtime_error("Config Error: Missing 'name' field in YAML.");

            const std::string gameName = config["name"].as<std::string>();
            const std::string dataFolder = "data/" + gameName;
            std::filesystem::create_directories(dataFolder);

            std::ostringstream oss;
            oss << dataFolder << "/iteration_"
                << std::setw(4) << std::setfill('0') << m_trainingCfg.currentIteration << ".bin";
            m_datasetPath = oss.str();

            if (config["training"] && config["training"]["drawSampleRate"])
                m_drawSampleRate = config["training"]["drawSampleRate"].as<float>();
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

        std::optional<GameResult> checkResign(const GameContext& g,
            uint32_t           currentPlayer) const
        {
            const float    threshold = this->m_engineCfg.resignThreshold;
            const uint32_t minPly = this->m_engineCfg.resignMinPly;

            if (threshold <= -1.0f || threshold >= 0.0f) return std::nullopt;
            if (g.turnCount < minPly)                    return std::nullopt;

            if (g.trees[currentPlayer]->getRootValue() < threshold)
                return this->m_engine->buildResignResult(currentPlayer);

            return std::nullopt;
        }

        // ----------------------------------------------------------------
        // Progress display
        //
        // Counters:
        //   gamesWritten  : games actually stored in the replay buffer
        //                   → loop stop condition, shown as progress
        //   gamesPlayed   : all games played including filtered draws
        //                   → shown for transparency
        //   writtenMoves  : total plies of written games → AvgLen
        //   gpuMoves      : all plies while gamesWritten < target → Speed
        // ----------------------------------------------------------------
        void printProgress(uint32_t gamesWritten, uint32_t target,
            uint32_t gamesPlayed,
            uint64_t writtenMoves, uint64_t gpuMoves,
            double   elapsedSec) const
        {
            const double avgLen = (gamesWritten > 0)
                ? static_cast<double>(writtenMoves) / gamesWritten : 0.0;
            const double speed = gpuMoves / std::max(elapsedSec, 1e-6);

            // Filter rate: how many games were played per written game
            const double filterRatio = (gamesWritten > 0)
                ? static_cast<double>(gamesPlayed) / gamesWritten : 1.0;

            std::cout
                << "[SelfPlay] " << std::setw(6) << gamesWritten
                << " / " << target
                << " written (" << std::fixed << std::setprecision(0)
                << gamesPlayed << " played"
                << ", x" << std::setprecision(1) << filterRatio << " filter)"
                << " | AvgLen: " << std::fixed << std::setprecision(1) << avgLen << " ply"
                << " | Speed: " << std::fixed << std::setprecision(1) << speed << " m/s"
                << "\r" << std::flush;
        }

    public:
        SelfPlayHandler() = default;
        virtual ~SelfPlayHandler() = default;

        void execute() override
        {
            std::signal(SIGINT, sigintHandler);

            const uint32_t target = m_trainingCfg.gamesPerIteration;

            std::cout
                << "[SelfPlay] Starting     : " << target << " games to write\n"
                << "[SelfPlay] File          : " << m_datasetPath << "\n"
                << "[SelfPlay] Parallelism   : " << m_backendCfg.numParallelGames
                << " games | Batch: " << m_backendCfg.inferenceBatchSize << "\n"
                << "[SelfPlay] Exploration   : Gumbel-Top-K (k="
                << this->m_engineCfg.gumbelK << ")\n"
                << "[SelfPlay] Policy target : "
                << (this->m_engineCfg.gumbelSigma > 0.0f
                    ? "Improved Q (sigma=" + std::to_string(this->m_engineCfg.gumbelSigma) + ")"
                    : "Visit counts (classic)")
                << "\n";

            const float thr = this->m_engineCfg.resignThreshold;
            if (thr > -1.0f && thr < 0.0f)
                std::cout << "[SelfPlay] Resign        : threshold=" << thr
                << "  minPly=" << this->m_engineCfg.resignMinPly << "\n";
            else
                std::cout << "[SelfPlay] Resign        : disabled\n";

            if (m_drawSampleRate < 1.0f)
                std::cout << "[SelfPlay] Draw filter   : " << m_drawSampleRate * 100.f
                << "% kept (iteration will be ~"
                << std::fixed << std::setprecision(1) << (1.0f / m_drawSampleRate)
                << "x longer)\n";
            else
                std::cout << "[SelfPlay] Draw filter   : disabled\n";

            // ---- Allocate game contexts ----
            std::vector<GameContext> games(m_backendCfg.numParallelGames);
            size_t treeAllocIdx = 0;
            for (auto& g : games) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                    g.trees[p] = this->m_treeSearch[treeAllocIdx++].get();
                g.actionHistory.reserve(Defs::kMaxHistory * 2);
                resetGame(g);
            }

            // Mark initial batch as official
            for (size_t i = 0; i < games.size() && i < static_cast<size_t>(target); ++i)
                games[i].isOfficial = true;

            std::ofstream outFile(m_datasetPath, std::ios::binary | std::ios::app);
            if (!outFile.is_open())
                throw std::runtime_error("Fatal: Cannot open dataset file: " + m_datasetPath);

            // ----------------------------------------------------------------
            // Counters
            //
            // gamesWritten  : games actually stored on disk
            //                 → LOOP STOP CONDITION (gamesWritten >= target)
            //                 → shown as progress numerator
            //
            // gamesPlayed   : all completed games (written + filtered draws)
            //                 → shown for transparency
            //
            // writtenMoves  : plies of written games only → correct AvgLen
            //
            // gpuMoves      : all plies played while gamesWritten < target
            //                 → denominator for Speed (true GPU throughput)
            // ----------------------------------------------------------------
            uint32_t gamesWritten = 0;
            uint32_t gamesPlayed = 0;
            uint64_t writtenMoves = 0;
            uint64_t gpuMoves = 0;

            auto startTime = std::chrono::high_resolution_clock::now();

            while (gamesWritten < target &&
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
                    const uint32_t  cp = this->m_engine->getCurrentPlayer(g.currentState);
                    TreeSearch<GT>* activeTree = g.trees[cp];

                    if (g.isOfficial)
                    {
                        State povState = g.currentState;
                        this->m_engine->changeStatePov(cp, povState);

                        const size_t histSize = std::min<size_t>(
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

                    const Action action = activeTree->selectMove(1.0f);

                    g.turnCount++;
                    this->m_engine->applyAction(action, g.currentState);
                    g.hashHistory.push_back(g.currentState.hash());
                    g.actionHistory.push_back(action);

                    for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                        g.trees[p]->advanceRoot(action, g.currentState);

                    gpuMoves++;

                    std::optional<GameResult> outcome =
                        this->m_engine->getGameResult(g.currentState, g.hashHistory);

                    if (!outcome.has_value())
                        outcome = checkResign(g, cp);

                    if (outcome.has_value())
                    {
                        if (g.isOfficial && gamesWritten < target)
                        {
                            gamesPlayed++;

                            const bool written = g.replayBuffer.flushToFile(
                                outcome.value(), outFile, m_drawSampleRate);

                            if (written) {
                                writtenMoves += g.turnCount;
                                gamesWritten++;
                            }

                            if (gamesWritten % 10 == 0 || gamesWritten == target)
                            {
                                auto   now = std::chrono::high_resolution_clock::now();
                                double elapsed = std::chrono::duration<double>(
                                    now - startTime).count();
                                printProgress(gamesWritten, target, gamesPlayed,
                                    writtenMoves, gpuMoves, elapsed);
                            }
                        }

                        resetGame(g);
                        // Official if we still need more written games
                        g.isOfficial = (gamesWritten < target);
                    }
                }
            }

            outFile.close();

            auto   end = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(end - startTime).count();
            int    hh = static_cast<int>(sec) / 3600;
            int    mm = (static_cast<int>(sec) % 3600) / 60;
            int    ss = static_cast<int>(sec) % 60;

            const double avgLen = (gamesWritten > 0)
                ? static_cast<double>(writtenMoves) / gamesWritten : 0.0;
            const double filterRate = (gamesWritten > 0)
                ? static_cast<double>(gamesPlayed) / gamesWritten : 1.0;

            std::cout << "\n\n[SelfPlay] Finished!"
                << "\n  Games written     : " << gamesWritten
                << "\n  Games played      : " << gamesPlayed
                << "\n  Filter ratio      : x" << std::fixed << std::setprecision(2) << filterRate
                << " (" << std::setprecision(1) << (100.0 / filterRate) << "% kept)"
                << "\n  Avg Game Length   : " << std::fixed << std::setprecision(1)
                << avgLen << " ply"
                << "\n  GPU Moves (total) : " << gpuMoves
                << "\n  Avg Speed         : " << std::fixed << std::setprecision(1)
                << (gpuMoves / std::max(sec, 1e-6)) << " moves/s"
                << "\n  Total Time        : "
                << std::setfill('0') << std::setw(2) << hh << "h "
                << std::setw(2) << mm << "m "
                << std::setw(2) << ss << "s\n";
        }
    };
}