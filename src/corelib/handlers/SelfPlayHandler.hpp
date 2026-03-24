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

// ─── Signal handling ──────────────────────────────────────────────────────────
// g_keepRunning is set to false when SIGINT is received.
// The signal handler itself does nothing except flip the flag —
// all I/O and cleanup happens on the main thread after the loop exits.
// NOTE: std::cout is NOT async-signal-safe; do not call it from the handler.
inline std::atomic<bool> g_keepRunning{ true };
inline void sigintHandler(int) {
    g_keepRunning.store(false, std::memory_order_release);
}

namespace Core
{
    /**
     * @brief High-Throughput Self-Play Handler
     *
     * Orchestrates parallel game generation using MCTS + NN inference and
     * serialises training samples into binary datasets.
     *
     * Threading model
     * ───────────────
     * The main loop (execute()) is single-threaded: it advances every game
     * by one ply and dispatches MCTS work to the ThreadPool.  The heavy
     * computation (tree search + inference + backprop) runs entirely inside
     * the ThreadPool worker threads and is invisible to this class.
     *
     * Data flow per ply
     * ─────────────────
     *   1. Collect the active tree for each parallel game.
     *   2. Call executeMultipleTrees() — blocks until all trees reach numSims.
     *   3. For official games, encode the current state and record a sample.
     *   4. Select and apply the chosen action; advance every tree root.
     *   5. Check for terminal state / resignation; flush buffer if game ended.
     *   6. Refresh the terminal dashboard.
     */
    template<ValidGameTraits GT>
    class SelfPlayHandler : public IHandler<GT>
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        // ─────────────────────────────────────────────────────────────────────
        // GameContext — everything required to drive one parallel game.
        // ─────────────────────────────────────────────────────────────────────
        struct GameContext
        {
            /// One independent search tree per player.
            /// In self-play, each player maintains its own root node so that
            /// tree reuse across moves is valid regardless of whose turn it is.
            std::array<TreeSearch<GT>*, Defs::kNumPlayers> trees;

            State                 currentState;
            AlignedVec<Action>    actionHistory;   ///< All plies since game start
            std::vector<uint64_t> hashHistory;     ///< Zobrist hashes for repetition detection
            ReplayBuffer<GT>      replayBuffer;    ///< Accumulates samples until game end

            uint32_t turnCount = 0;
            /// True → this game's outcome will be committed to the dataset file.
            bool     isOfficial = false;
        };

        // ─────────────────────────────────────────────────────────────────────
        // DashSnap — point-in-time snapshot of the MCTS metrics.
        //
        // WHY A SEPARATE STRUCT?
        // The dashboard is rendered AFTER advanceRoot() has been called for
        // every parallel game.  advanceRoot() unconditionally calls
        // resetCounters() (tree-reuse path) or startSearch() (full-reset
        // path), both of which zero m_simulationsFinished and m_nodeCount.
        // Reading getRootValue() / getSimulationCount() / getMemoryUsage()
        // after that point always returns 0.
        //
        // The fix: call captureSnap() RIGHT AFTER executeMultipleTrees()
        // returns — before any advanceRoot() or resetGame() — and stash
        // the values here.
        // ─────────────────────────────────────────────────────────────────────
        struct DashSnap
        {
            float    rootQ = 0.0f;
            uint32_t sims = 0;
            int      memPct = 0;
        };

        // ─────────────────────────────────────────────────────────────────────
        // DashboardState — live metrics displayed in the terminal.
        // Kept as a plain struct (no atomics) because it is only accessed
        // from the single-threaded main loop.
        // ─────────────────────────────────────────────────────────────────────
        struct DashboardState
        {
            uint32_t gamesWritten = 0; ///< Games whose samples were committed to disk
            uint32_t gamesEnded = 0; ///< Official games that reached a terminal state
            ///  (includes draws filtered by drawSampleRate)
            uint64_t totalMoves = 0; ///< All plies across every parallel game
            uint64_t totalSamples = 0; ///< Total training positions written
            uint64_t totalPlies = 0; ///< Cumulative game length for written games
            ///  (used to compute average game length)
            bool firstDraw = true;     ///< True before the first dashboard render
        };

        // ─────────────────────────────────────────────────────────────────────
        // Configuration
        // ─────────────────────────────────────────────────────────────────────
        BackendConfig  m_backendCfg;
        TrainingConfig m_trainingCfg;
        std::string    m_datasetPath;

        // ─────────────────────────────────────────────────────────────────────
        // Dashboard geometry
        // kBoxWidth  — inner width of the box in terminal columns.
        // kDashLines — MUST equal the exact number of lines printed by
        //              printDashboard() so that cursor-up repositioning works.
        //              Update this constant whenever lines are added/removed.
        // ─────────────────────────────────────────────────────────────────────
        static constexpr int kBoxWidth = 60;
        static constexpr int kDashLines = 14;

        // ═════════════════════════════════════════════════════════════════════
        // IHandler interface
        // ═════════════════════════════════════════════════════════════════════

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

        // ═════════════════════════════════════════════════════════════════════
        // Game management
        // ═════════════════════════════════════════════════════════════════════

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

        // ═════════════════════════════════════════════════════════════════════
        // Dashboard — static helpers
        // ═════════════════════════════════════════════════════════════════════

        /// ASCII progress bar: "[################----]"
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

        /// Format a duration in seconds as "Xh XXm XXs".
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

        /// Format a large value with an SI suffix (k, M).
        static std::string fmtSI(double v)
        {
            char buf[32];
            if (v >= 1e6) std::snprintf(buf, sizeof(buf), "%.2fM", v * 1e-6);
            else if (v >= 1e3) std::snprintf(buf, sizeof(buf), "%.2fk", v * 1e-3);
            else               std::snprintf(buf, sizeof(buf), "%.0f", v);
            return buf;
        }

        /// Build a full box content row: "║ <content padded to kBoxWidth-2> ║\n"
        /// Content must be pure ASCII (size() == rendered column count).
        static std::string boxRow(const std::string& content)
        {
            // kBoxWidth - 2 = content area (1 space of padding on each side)
            constexpr int kContentWidth = kBoxWidth - 2;
            std::string line;
            line.reserve(kBoxWidth + 8); // +8 for multi-byte border chars + newline
            line += "║ ";
            line += content;
            const int pad = kContentWidth - static_cast<int>(content.size());
            if (pad > 0) line.append(static_cast<size_t>(pad), ' ');
            line += " ║\n";
            return line;
        }

        /// Build a section separator: "╠══ TITLE ══╣\n"
        /// Pass an empty title for a plain horizontal rule.
        static std::string boxRuler(const std::string& title = "")
        {
            static const std::string HL = "═"; // U+2550, 3 bytes, 1 column
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

        // ═════════════════════════════════════════════════════════════════════
        // Dashboard — main render
        // ═════════════════════════════════════════════════════════════════════

        /**
         * @brief Renders the live dashboard in-place using ANSI escape codes.
         *
         * The method uses "\033[NF" to move the cursor N lines up, then
         * overwrites every line with "\033[K" (clear to end of line) before
         * writing new content.  This eliminates flicker caused by partial
         * redraws.
         *
         * Line count: must equal kDashLines exactly.
         *
         * @param d       Live metrics (firstDraw flag is modified here)
         * @param target  Total games requested for this iteration
         * @param elapsed Wall-clock seconds since session start
         * @param sample  Any running game (used to read one tree's metrics)
         */
        void printDashboard(DashboardState& d, uint32_t target,
            double elapsed, const DashSnap& snap) const
        {
            static const std::string HL = "═";
            static const std::string CL = "\033[K"; // clear to end of line

            // Reposition cursor at the first dashboard line on subsequent renders
            if (!d.firstDraw)
                std::cout << "\033[" << kDashLines << "F";
            d.firstDraw = false;

            // ── Compute derived metrics ───────────────────────────────────
            const double safeElapsed = std::max(elapsed, 1e-6);
            const double ratio = target > 0
                ? static_cast<double>(d.gamesWritten) / target : 0.0;
            const double gps = d.gamesWritten / safeElapsed;
            const double mps = d.totalMoves / safeElapsed;
            const double avgLen = d.gamesWritten > 0
                ? static_cast<double>(d.totalPlies) / d.gamesWritten : 0.0;
            const double eta = (gps > 1e-9 && d.gamesWritten < target)
                ? (target - d.gamesWritten) / gps : 0.0;

            // Use the pre-captured snapshot — values are valid because they
            // were read before advanceRoot() called resetCounters().
            const float    rootQ = snap.rootQ;
            const uint32_t sims = snap.sims;
            const int      memPct = snap.memPct;

            // ── Build the frame into a single string (minimises write() calls) ──
            std::ostringstream o;
            char buf[128];

            // Line 1 ── ╔══ HEADER ══╗
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

            // Line 2 ── Dataset path (truncated from the left if necessary)
            {
                constexpr int kMaxPath = kBoxWidth - 12; // "Dataset : " = 10 chars
                std::string path = m_datasetPath;
                if (static_cast<int>(path.size()) > kMaxPath)
                    path = "..." + path.substr(path.size() - static_cast<size_t>(kMaxPath - 3));
                std::snprintf(buf, sizeof(buf), "Dataset : %s", path.c_str());
                o << CL << boxRow(buf);
            }

            // Line 3 ── ╠══ PROGRESS ══╣
            o << CL << boxRuler("PROGRESS");

            // Line 4 ── Games progress bar
            {
                std::snprintf(buf, sizeof(buf),
                    "Games   : %5u / %-5u %s %5.1f%%",
                    d.gamesWritten, target,
                    progressBar(ratio, 16).c_str(),
                    ratio * 100.0);
                o << CL << boxRow(buf);
            }

            // Line 5 ── Samples and average game length
            {
                std::snprintf(buf, sizeof(buf),
                    "Samples : %-8s  |  Avg length : %5.1f plies",
                    fmtSI(static_cast<double>(d.totalSamples)).c_str(), avgLen);
                o << CL << boxRow(buf);
            }

            // Line 6 ── ╠══ THROUGHPUT ══╣
            o << CL << boxRuler("THROUGHPUT");

            // Line 7 ── Games/s and Moves/s
            {
                std::snprintf(buf, sizeof(buf),
                    "Games/s : %7.2f  |  Moves/s : %-10s",
                    gps, fmtSI(mps).c_str());
                o << CL << boxRow(buf);
            }

            // Line 8 ── Elapsed time and ETA
            {
                std::snprintf(buf, sizeof(buf),
                    "Elapsed : %-14s  ETA : %s",
                    fmtDuration(elapsed).c_str(),
                    d.gamesWritten < target ? fmtDuration(eta).c_str() : "done!  ");
                o << CL << boxRow(buf);
            }

            // Line 9 ── ╠══ SEARCH ══╣
            o << CL << boxRuler("SEARCH");

            // Line 10 ── Root Q-value, simulation count, memory pressure
            {
                std::snprintf(buf, sizeof(buf),
                    "Root Q  : %+.4f  |  Sims : %5u  |  Mem : %3d%%",
                    rootQ, sims, memPct);
                o << CL << boxRow(buf);
            }

            // Line 11 ── ╠══ PIPELINE ══╣
            o << CL << boxRuler("PIPELINE");

            // NOTE: The ThreadPool queues are always empty when sampled here
            // because executeMultipleTrees() is blocking — it only returns once
            // m_pendingTasks == 0, meaning every event has already been drained
            // from every queue.  Showing "0 0 0" would be misleading.
            // We display the static thread-pool configuration instead, which is
            // the actually useful information for tuning purposes.

            // Line 12 ── Thread counts
            {
                std::snprintf(buf, sizeof(buf),
                    "Threads  Search:%-3u  Infer:%-3u  Backprop:%-3u",
                    m_backendCfg.numSearchThreads,
                    m_backendCfg.numInferenceThreads,
                    m_backendCfg.numBackpropThreads);
                o << CL << boxRow(buf);
            }

            // Line 13 ── Batch size and parallel games
            {
                std::snprintf(buf, sizeof(buf),
                    "Batch   : %4u   |  Parallel games : %4u",
                    m_backendCfg.inferenceBatchSize,
                    m_backendCfg.numParallelGames);
                o << CL << boxRow(buf);
            }

            // Line 14 ── ╚══════════╝
            {
                o << CL << "╚";
                for (int i = 0; i < kBoxWidth; ++i) o << HL;
                o << "╝\n";
            }

            std::cout << o.str() << std::flush;
        }

        // ═════════════════════════════════════════════════════════════════════
        // Session summary — printed once, after the loop exits
        // ═════════════════════════════════════════════════════════════════════

        static void printSummary(const DashboardState& d, uint32_t target,
            double totalTime, const std::string& datasetPath)
        {
            static const std::string HL = "═";

            auto srow = [](const std::string& content) {
                std::cout << boxRow(content);
                };

            char buf[128];
            std::cout << '\n';

            // Top border
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

            // Bottom border
            std::cout << "╚";
            for (int i = 0; i < kBoxWidth; ++i) std::cout << HL;
            std::cout << "╝\n" << std::flush;
        }

    public:
        SelfPlayHandler() = default;
        ~SelfPlayHandler() = default;

        // ═════════════════════════════════════════════════════════════════════
        // Main entry point
        // ═════════════════════════════════════════════════════════════════════

        void execute() override
        {
            std::signal(SIGINT, sigintHandler);
            const uint32_t target = m_trainingCfg.gamesPerIteration;

            // ── Initialise game contexts ──────────────────────────────────
            std::vector<GameContext> games(m_backendCfg.numParallelGames);
            size_t treeAllocIdx = 0;
            for (auto& g : games) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                    g.trees[p] = this->m_treeSearch[treeAllocIdx++].get();
                // Reserve enough space to avoid rehashing during a long game
                g.actionHistory.reserve(Defs::kMaxHistory * 2);
                resetGame(g);
            }

            // Mark the first N contexts as "official" (contributing to the quota)
            for (size_t i = 0; i < games.size() && i < static_cast<size_t>(target); ++i)
                games[i].isOfficial = true;

            // ── Open output file ──────────────────────────────────────────
            std::ofstream outFile(m_datasetPath, std::ios::binary | std::ios::app);
            if (!outFile.is_open())
                throw std::runtime_error(
                    "[SelfPlayHandler] Cannot open dataset for writing: " + m_datasetPath);

            // ── Pre-allocate hot-path buffers ─────────────────────────────
            // FIX: allocating activeTrees inside the loop caused one heap
            // allocation per iteration.  Reserve once here instead.
            std::vector<TreeSearch<GT>*> activeTrees;
            activeTrees.reserve(m_backendCfg.numParallelGames);

            DashboardState dash;
            const auto startTime = std::chrono::high_resolution_clock::now();

            // Reserve vertical space so the first render doesn't scroll the terminal
            std::cout << std::string(kDashLines, '\n');

            // ═════════════════════════════════════════════════════════════
            // Main loop — one iteration == one ply across every parallel game
            // ═════════════════════════════════════════════════════════════
            while (dash.gamesWritten < target
                && g_keepRunning.load(std::memory_order_acquire))
            {
                // ── Step 1 : collect the active tree for each game ────────
                // Each game may have a different current player, so we cannot
                // cache this mapping across iterations.
                activeTrees.clear();
                for (const auto& g : games)
                    activeTrees.push_back(
                        g.trees[this->m_engine->getCurrentPlayer(g.currentState)]);

                // ── Step 2 : run MCTS (blocking) ──────────────────────────
                this->m_threadPool->executeMultipleTrees(
                    activeTrees, this->m_engineCfg.numSimulations);

                // ── Capture tree metrics BEFORE any advanceRoot() call ────
                // advanceRoot() calls resetCounters() (tree-reuse path) or
                // startSearch() (full-reset path) — both zero
                // m_simulationsFinished and m_nodeCount immediately.
                // activeTrees[0] is the tree that just finished for games[0]
                // this iteration; its counters are still valid right here.
                DashSnap snap;
                if (!activeTrees.empty()) {
                    auto* t = activeTrees[0];
                    snap.rootQ = t->getRootValue();
                    snap.sims = t->getSimulationCount();
                    snap.memPct = static_cast<int>(t->getMemoryUsage() * 100.0f);
                }

                // ── Step 3 : advance every game by one ply ────────────────
                for (auto& g : games)
                {
                    const uint32_t cp = this->m_engine->getCurrentPlayer(g.currentState);
                    TreeSearch<GT>* activeTree = g.trees[cp];

                    // ── 3a. Record a training sample (official games only) ─
                    if (g.isOfficial)
                    {
                        // Build the POV-rotated state and history for the
                        // current player before the action is applied
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

                    // ── 3b. Select and apply the chosen action ────────────
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

                    // Advance every player's tree root to the new state
                    for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                        g.trees[p]->advanceRoot(action, g.currentState);

                    // ── 3c. Check for game termination ────────────────────
                    auto outcome = this->m_engine->getGameResult(
                        g.currentState, g.hashHistory);

                    // Resignation: only eligible after the minimum ply
                    // threshold, and only when the active player's Q is below
                    // the configured threshold from their own perspective
                    if (!outcome
                        && g.turnCount > this->m_engineCfg.resignMinPly
                        && resignQ < this->m_engineCfg.resignThreshold)
                    {
                        outcome = this->m_engine->buildResignResult(cp);
                    }

                    // ── 3d. Handle game end ───────────────────────────────
                    if (outcome)
                    {
                        if (g.isOfficial && dash.gamesWritten < target)
                        {
                            // gamesEnded counts every official terminal state,
                            // whereas gamesWritten is only incremented when
                            // flushToFile() actually writes data (draws may be
                            // discarded by drawSampleRate)
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
                        // Only flag the new game as official if we still need more
                        g.isOfficial = (dash.gamesWritten < target);
                    }
                }

                // ── Step 4 : refresh the dashboard ───────────────────────
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - startTime).count();
                printDashboard(dash, target, elapsed, snap);
            }

            // ── Print interrupt notice if the loop was interrupted ────────
            if (!g_keepRunning.load(std::memory_order_acquire))
                std::cout << "\n[System] Interrupted — flushing buffers...\n";

            // ── Final summary ─────────────────────────────────────────────
            const double totalTime = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startTime).count();

            printSummary(dash, target, totalTime, m_datasetPath);
        }
    };
}