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
#include <cmath>
#include <cstdio>

#include "../interfaces/IHandler.hpp"
#include "../model/ReplayBuffer.hpp"
#include "../model/StateEncoder.hpp"

// Graceful shutdown via SIGINT allows flushing accumulated game data to disk 
// before terminating, preventing dataset corruption.
inline std::atomic<bool> g_keepRunning{ true };
inline void sigintHandler(int) {
    g_keepRunning.store(false, std::memory_order_release);
}

namespace Core
{
    // ============================================================================
    // SELF-PLAY HANDLER
    // Orchestrates parallel game generation. 
    //
    // Architecture:
    // The main thread strictly handles game state progression and binary I/O.
    // All heavy lifting (MCTS traversal, NN batching, TensorRT inference) is 
    // offloaded transparently to the ThreadPool. This separation prevents the 
    // control loop from bottlenecking GPU throughput.
    // ============================================================================
    template<ValidGameTraits GT>
    class SelfPlayHandler : public IHandler<GT>
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        struct GameContext
        {
            // Each player requires an independent root node to ensure tree reuse 
            // remains valid across sequential turns in self-play.
            std::array<TreeSearch<GT>*, Defs::kNumPlayers> trees;

            State                 currentState;
            AlignedVec<Action>    actionHistory;
            std::vector<uint64_t> hashHistory;
            ReplayBuffer<GT>      replayBuffer;

            uint32_t turnCount = 0;
            bool     isOfficial = false; // Prevents over-generation past target quota
        };

        // Captures MCTS metrics immediately after search completes.
        // Required because tree progression (advanceRoot) wipes root metrics 
        // before the dashboard has a chance to render them.
        struct DashSnap
        {
            float    rootQ = 0.0f;
            uint32_t sims = 0;
            int      memPct = 0;
        };

        struct DashboardState
        {
            uint32_t gamesWritten = 0;
            uint32_t gamesEnded = 0;
            uint64_t totalMoves = 0;
            uint64_t totalSamples = 0;
            uint64_t totalPlies = 0;
            bool firstDraw = true;
        };

        BackendConfig  m_backendCfg;
        TrainingConfig m_trainingCfg;
        std::string    m_datasetPath;

        static constexpr int kBoxWidth = 60;
        static constexpr int kDashLines = 14;

        void specificSetup(const YAML::Node& config) override
        {
            std::cout << "[SelfPlayHandler] Setup initialized.\n";

            m_backendCfg.load(config, "train");
            m_trainingCfg.load(config, "train");

            const std::string gameName = config["name"].as<std::string>();
            const std::string dataFolder = "data/" + gameName;
            std::filesystem::create_directories(dataFolder);

            std::ostringstream oss;
            oss << dataFolder << "/iteration_"
                << std::setw(4) << std::setfill('0')
                << m_trainingCfg.currentIteration << ".bin";
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

        // --- Terminal Dashboard Formatting Helpers ---

        static std::string progressBar(double ratio, int width = 20)
        {
            ratio = std::clamp(ratio, 0.0, 1.0);
            const int filled = static_cast<int>(std::round(ratio * width));
            std::string bar;
            bar.reserve(width + 2);
            bar += '[';
            for (int i = 0; i < width; ++i) bar += (i < filled) ? '#' : '-';
            bar += ']';
            return bar;
        }

        static std::string fmtDuration(double seconds)
        {
            const auto s = static_cast<uint64_t>(std::max(seconds, 0.0));
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%uh %02um %02us",
                static_cast<unsigned>(s / 3600),
                static_cast<unsigned>((s % 3600) / 60),
                static_cast<unsigned>(s % 60));
            return buf;
        }

        static std::string fmtSI(double v)
        {
            char buf[32];
            if (v >= 1e6) std::snprintf(buf, sizeof(buf), "%.2fM", v * 1e-6);
            else if (v >= 1e3) std::snprintf(buf, sizeof(buf), "%.2fk", v * 1e-3);
            else               std::snprintf(buf, sizeof(buf), "%.0f", v);
            return buf;
        }

        static std::string boxRow(const std::string& content)
        {
            constexpr int kContentWidth = kBoxWidth - 2;
            std::string line;
            line.reserve(kBoxWidth + 8);
            line += "║ ";
            line += content;
            const int pad = kContentWidth - static_cast<int>(content.size());
            if (pad > 0) line.append(static_cast<size_t>(pad), ' ');
            line += " ║\n";
            return line;
        }

        static std::string boxRuler(const std::string& title = "")
        {
            static const std::string HL = "═";
            std::string out;
            out.reserve(kBoxWidth * 3 + 8);
            out += "╠";
            if (title.empty()) {
                for (int i = 0; i < kBoxWidth; ++i) out += HL;
            }
            else {
                const std::string label = " " + title + " ";
                const int fill = kBoxWidth - static_cast<int>(label.size());
                const int left = fill / 2;
                const int right = fill - left;
                for (int i = 0; i < left; ++i) out += HL;
                out += label;
                for (int i = 0; i < right; ++i) out += HL;
            }
            out += "╣\n";
            return out;
        }

        // Renders dashboard in-place using ANSI escape codes to prevent flickering.
        void printDashboard(DashboardState& d, uint32_t target,
            double elapsed, const DashSnap& snap) const
        {
            static const std::string HL = "═";
            static const std::string CL = "\033[K";

            if (!d.firstDraw)
                std::cout << "\033[" << kDashLines << "F";
            d.firstDraw = false;

            const double safeElapsed = std::max(elapsed, 1e-6);
            const double ratio = target > 0
                ? static_cast<double>(d.gamesWritten) / target : 0.0;
            const double gps = d.gamesWritten / safeElapsed;
            const double mps = d.totalMoves / safeElapsed;
            const double avgLen = d.gamesWritten > 0
                ? static_cast<double>(d.totalPlies) / d.gamesWritten : 0.0;
            const double eta = (gps > 1e-9 && d.gamesWritten < target)
                ? (target - d.gamesWritten) / gps : 0.0;

            const float    rootQ = snap.rootQ;
            const uint32_t sims = snap.sims;
            const int      memPct = snap.memPct;

            std::ostringstream o;
            char buf[128];

            {
                const std::string title = " SELF-PLAY ENGINE ";
                const int fill = kBoxWidth - static_cast<int>(title.size());
                const int left = fill / 2;
                const int right = fill - left;
                o << CL << "╔";
                for (int i = 0; i < left; ++i) o << HL;
                o << title;
                for (int i = 0; i < right; ++i) o << HL;
                o << "╗\n";
            }

            {
                constexpr int kMaxPath = kBoxWidth - 12;
                std::string path = m_datasetPath;
                if (static_cast<int>(path.size()) > kMaxPath)
                    path = "..." + path.substr(path.size() - static_cast<size_t>(kMaxPath - 3));
                std::snprintf(buf, sizeof(buf), "Dataset : %s", path.c_str());
                o << CL << boxRow(buf);
            }

            o << CL << boxRuler("PROGRESS");

            {
                std::snprintf(buf, sizeof(buf),
                    "Games   : %5u / %-5u %s %5.1f%%",
                    d.gamesWritten, target,
                    progressBar(ratio, 16).c_str(),
                    ratio * 100.0);
                o << CL << boxRow(buf);
            }

            {
                std::snprintf(buf, sizeof(buf),
                    "Samples : %-8s  |  Avg length : %5.1f plies",
                    fmtSI(static_cast<double>(d.totalSamples)).c_str(), avgLen);
                o << CL << boxRow(buf);
            }

            o << CL << boxRuler("THROUGHPUT");

            {
                std::snprintf(buf, sizeof(buf),
                    "Games/s : %7.2f  |  Moves/s : %-10s",
                    gps, fmtSI(mps).c_str());
                o << CL << boxRow(buf);
            }

            {
                std::snprintf(buf, sizeof(buf),
                    "Elapsed : %-14s  ETA : %s",
                    fmtDuration(elapsed).c_str(),
                    d.gamesWritten < target ? fmtDuration(eta).c_str() : "done!  ");
                o << CL << boxRow(buf);
            }

            o << CL << boxRuler("SEARCH");

            {
                std::snprintf(buf, sizeof(buf),
                    "Root Q  : %+.4f  |  Sims : %5u  |  Mem : %3d%%",
                    rootQ, sims, memPct);
                o << CL << boxRow(buf);
            }

            o << CL << boxRuler("PIPELINE");

            {
                std::snprintf(buf, sizeof(buf),
                    "Threads  Search:%-3u  Infer:%-3u  Backprop:%-3u",
                    m_backendCfg.numSearchThreads,
                    m_backendCfg.numInferenceThreads,
                    m_backendCfg.numBackpropThreads);
                o << CL << boxRow(buf);
            }

            {
                std::snprintf(buf, sizeof(buf),
                    "Batch   : %4u   |  Parallel games : %4u",
                    m_backendCfg.inferenceBatchSize,
                    m_backendCfg.numParallelGames);
                o << CL << boxRow(buf);
            }

            {
                o << CL << "╚";
                for (int i = 0; i < kBoxWidth; ++i) o << HL;
                o << "╝\n";
            }

            std::cout << o.str() << std::flush;
        }

        static void printSummary(const DashboardState& d, uint32_t target,
            double totalTime, const std::string& datasetPath)
        {
            static const std::string HL = "═";
            auto srow = [](const std::string& content) { std::cout << boxRow(content); };

            char buf[128];
            std::cout << '\n';

            std::cout << "╔";
            for (int i = 0; i < kBoxWidth; ++i) std::cout << HL;
            std::cout << "╗\n";

            srow(" SELF-PLAY SESSION COMPLETE");
            std::cout << boxRuler();

            std::snprintf(buf, sizeof(buf), "Dataset    : %s", datasetPath.c_str());
            srow(buf);

            std::snprintf(buf, sizeof(buf),
                "Games      : %u written  /  %u ended  /  %u target",
                d.gamesWritten, d.gamesEnded, target);
            srow(buf);

            std::snprintf(buf, sizeof(buf),
                "Samples    : %s positions",
                fmtSI(static_cast<double>(d.totalSamples)).c_str());
            srow(buf);

            const double avgLen = d.gamesWritten > 0
                ? static_cast<double>(d.totalPlies) / d.gamesWritten : 0.0;
            std::snprintf(buf, sizeof(buf),
                "Avg length : %.1f plies / game", avgLen);
            srow(buf);

            const double throughput = d.totalMoves / std::max(totalTime, 1e-6);
            std::snprintf(buf, sizeof(buf),
                "Throughput : %s moves/s  |  Duration : %s",
                fmtSI(throughput).c_str(),
                fmtDuration(totalTime).c_str());
            srow(buf);

            std::cout << "╚";
            for (int i = 0; i < kBoxWidth; ++i) std::cout << HL;
            std::cout << "╝\n" << std::flush;
        }

    public:
        SelfPlayHandler() = default;
        ~SelfPlayHandler() = default;

        void execute() override
        {
            std::signal(SIGINT, sigintHandler);
            const uint32_t target = m_trainingCfg.gamesPerIteration;

            std::vector<GameContext> games(m_backendCfg.numParallelGames);
            size_t treeAllocIdx = 0;
            for (auto& g : games) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                    g.trees[p] = this->m_treeSearch[treeAllocIdx++].get();

                g.actionHistory.reserve(Defs::kMaxHistory * 2);
                resetGame(g);
            }

            for (size_t i = 0; i < games.size() && i < static_cast<size_t>(target); ++i)
                games[i].isOfficial = true;

            std::ofstream outFile(m_datasetPath, std::ios::binary | std::ios::app);
            if (!outFile.is_open())
                throw std::runtime_error(
                    "[SelfPlayHandler] Cannot open dataset for writing: " + m_datasetPath);

            // Pre-allocate active structures to prevent heap thrashing inside the hot loop.
            std::vector<TreeSearch<GT>*> activeTrees;
            activeTrees.reserve(m_backendCfg.numParallelGames);

            DashboardState dash;
            const auto startTime = std::chrono::high_resolution_clock::now();

            std::cout << std::string(kDashLines, '\n');

            while (dash.gamesWritten < target
                && g_keepRunning.load(std::memory_order_acquire))
            {
                activeTrees.clear();
                for (const auto& g : games)
                    activeTrees.push_back(
                        g.trees[this->m_engine->getCurrentPlayer(g.currentState)]);

                this->m_threadPool->executeMultipleTrees(
                    activeTrees, this->m_engineCfg.numSimulations);

                DashSnap snap;
                if (!activeTrees.empty()) {
                    auto* t = activeTrees[0];
                    snap.rootQ = t->getRootValue();
                    snap.sims = t->getSimulationCount();
                    snap.memPct = static_cast<int>(t->getMemoryUsage() * 100.0f);
                }

                for (auto& g : games)
                {
                    const uint32_t cp = this->m_engine->getCurrentPlayer(g.currentState);
                    TreeSearch<GT>* activeTree = g.trees[cp];

                    if (g.isOfficial)
                    {
                        State povState = g.currentState;
                        this->m_engine->changeStatePov(cp, povState);

                        const size_t histSize = std::min<size_t>(
                            g.actionHistory.size(), Defs::kMaxHistory);

                        StaticVec<Action, Defs::kMaxHistory> povHistory;
                        for (size_t i = g.actionHistory.size() - histSize;
                            i < g.actionHistory.size(); ++i)
                        {
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
                            cp);
                    }

                    const float temperature =
                        (g.turnCount < this->m_engineCfg.temperatureDropTurn)
                        ? 1.0f : 0.0f;

                    const Action action = activeTree->selectMove(temperature);
                    const float resignQ = activeTree->getRootValue();

                    g.turnCount++;
                    dash.totalMoves++;

                    this->m_engine->applyAction(action, g.currentState);
                    g.hashHistory.push_back(g.currentState.hash());
                    g.actionHistory.push_back(action);

                    for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                        g.trees[p]->advanceRoot(action, g.currentState);

                    auto outcome = this->m_engine->getGameResult(
                        g.currentState, g.hashHistory);

                    if (!outcome
                        && g.turnCount > this->m_engineCfg.resignMinPly
                        && resignQ < this->m_engineCfg.resignThreshold)
                    {
                        outcome = this->m_engine->buildResignResult(cp);
                    }

                    if (outcome)
                    {
                        if (g.isOfficial && dash.gamesWritten < target)
                        {
                            dash.gamesEnded++;
                            const size_t samples = g.replayBuffer.size();

                            if (g.replayBuffer.flushToFile(
                                *outcome,
                                outFile,
                                m_trainingCfg.drawSampleRate,
                                m_trainingCfg.drawScore))
                            {
                                dash.totalPlies += g.turnCount;
                                dash.totalSamples += samples;
                                dash.gamesWritten++;
                            }
                        }

                        resetGame(g);
                        g.isOfficial = (dash.gamesWritten < target);
                    }
                }

                const double elapsed = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - startTime).count();
                printDashboard(dash, target, elapsed, snap);
            }

            if (!g_keepRunning.load(std::memory_order_acquire))
                std::cout << "\n[System] Interrupted — flushing buffers...\n";

            const double totalTime = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startTime).count();

            printSummary(dash, target, totalTime, m_datasetPath);
        }
    };
}