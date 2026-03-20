#pragma once
#include <atomic>
#include <random>
#include <cstring>
#include <numeric>
#include <limits>
#include <vector>
#include <cmath>
#include <span>
#include <array>

#include "../bootstrap/GameConfig.hpp"
#include "../interfaces/IEngine.hpp"
#include "../util/PovUtils.hpp"
#include "SearchStrategy.hpp"
#include "StateEncoder.hpp"

namespace Core
{
    // ========================================================================
    // ATOMIC WRAPPER
    // ========================================================================
    template<typename T>
    struct AtomicVal {
        std::atomic<T> val;
        AtomicVal(T v = 0) : val(v) {}
        AtomicVal(const AtomicVal& o) {
            val.store(o.val.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        AtomicVal& operator=(const AtomicVal& o) {
            val.store(o.val.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };

    // ========================================================================
    // NODE EVENT
    //
    // WDL LAYOUT: for each player p, 3 consecutive floats:
    //   [p*3+0] = P(Win), [p*3+1] = P(Draw), [p*3+2] = P(Loss)
    //
    //   nnWDL   : filled by GPU inference
    //   trueWDL : filled from GameResult::wdl for terminal nodes
    //
    // Scalar for PUCT/backprop: W - L = [p*3+0] - [p*3+2]  ∈ [-1, +1]
    // ========================================================================
    template<ValidGameTraits GT>
    struct NodeEvent
    {
        USING_GAME_TYPES(GT);

        uint32_t             leafNodeIdx = 0;
        AlignedVec<uint32_t> path;
        AlignedVec<Action>   pathActions;
        AlignedVec<uint64_t> pathHashes;

        std::array<float, Defs::kNNInputSize>    nnInput{};
        std::array<float, Defs::kNumPlayers * 3> nnWDL{};
        std::array<float, Defs::kNumPlayers * 3> trueWDL{};
        std::array<float, Defs::kActionSpace>    policy{};

        ActionList validActions;

        bool isTerminal = false;
        bool collision = false;

        explicit NodeEvent(uint32_t maxDepth)
            : path(reserve_only, maxDepth)
            , pathActions(reserve_only, maxDepth)
            , pathHashes(reserve_only, maxDepth)
        {
        }

        void reset()
        {
            path.clear();
            pathActions.clear();
            pathHashes.clear();
            validActions.clear();

            nnInput.fill(0.0f);
            nnWDL.fill(0.0f);
            trueWDL.fill(0.0f);
            policy.fill(0.0f);

            leafNodeIdx = 0;
            isTerminal = false;
            collision = false;
        }

        // W - L ∈ [-1, +1] for player p.
        [[nodiscard]] float scalarValue(size_t p, bool fromNN) const noexcept
        {
            const auto& wdl = fromNN ? nnWDL : trueWDL;
            return wdl[p * 3 + 0] - wdl[p * 3 + 2];
        }
    };

    // ========================================================================
    // MONTE CARLO TREE SEARCH (Lock-Free Asynchronous)
    //
    // Gumbel extensions (compatible with the async parallel pipeline):
    //
    //  1. applyGumbelTopK() is called instead of applyDirichletNoise()
    //     at the root. Controlled by gumbelK in EngineConfig (0 = disabled).
    //
    //  2. getRootPolicy() uses computeImprovedPolicy() to produce a better
    //     training target via completed Q-values. Controlled by gumbelSigma
    //     in EngineConfig (0 = raw visit counts, classic AlphaZero).
    // ========================================================================
    template<ValidGameTraits GT>
    class TreeSearch
    {
    private:
        USING_GAME_TYPES(GT);
        using Event = NodeEvent<GT>;
        using Strategy = StrategyPUCT;
        using EdgeData = typename Strategy::EdgeData;

        static constexpr uint8_t FLAG_NONE = 0x00;
        static constexpr uint8_t FLAG_EXPANDING = 0x01;
        static constexpr uint8_t FLAG_EXPANDED = 0x02;
        static constexpr uint8_t FLAG_TERMINAL = 0x04;

        const EngineConfig           m_config;
        std::shared_ptr<IEngine<GT>> m_engine;

        AlignedVec<AtomicVal<uint8_t>>  m_nodeFlags;
        AlignedVec<AtomicVal<uint16_t>> m_nodeNumChildren;
        AlignedVec<uint32_t>            m_nodeFirstChild;
        AlignedVec<EdgeData>            m_nodeEdges;
        AlignedVec<float>               m_nodePrior;
        AlignedVec<Action>              m_nodeAction;

        State                    m_rootState;
        uint32_t                 m_rootIdx = UINT32_MAX;
        AlignedVec<Action>       m_realHistory;
        std::vector<uint64_t>    m_realHashHistory;

        std::atomic<uint32_t>    m_nodeCount{ 0 };
        std::atomic<uint32_t>    m_simulationsLaunched{ 0 };
        std::atomic<uint32_t>    m_simulationsFinished{ 0 };

        // ----------------------------------------------------------------
        uint32_t allocNodes(uint32_t count)
        {
            uint32_t idx = m_nodeCount.fetch_add(count, std::memory_order_relaxed);
            return (idx + count > m_config.maxNodes) ? UINT32_MAX : idx;
        }

        // ----------------------------------------------------------------
        void prepareNodeInput(Event& ctx, const State& leafState)
        {
            uint32_t viewer = m_engine->getCurrentPlayer(leafState);

            State povState = leafState;
            m_engine->changeStatePov(viewer, povState);

            const size_t totalNeeded = Defs::kMaxHistory;
            const size_t realCount = m_realHistory.size();
            const size_t pathCount = ctx.pathActions.size();
            const size_t totalAvailable = realCount + pathCount;
            size_t startIdx = (totalAvailable > totalNeeded) ? (totalAvailable - totalNeeded) : 0;
            size_t processed = 0;

            AlignedVec<Action> povHistory(reserve_only, totalNeeded);

            for (size_t i = 0; i < realCount; ++i) {
                if (processed >= startIdx || (realCount - i + pathCount) <= totalNeeded) {
                    Action a = m_realHistory[i];
                    m_engine->changeActionPov(viewer, a);
                    povHistory.push_back(a);
                }
                ++processed;
            }
            for (size_t i = 0; i < pathCount; ++i) {
                Action a = ctx.pathActions[i];
                m_engine->changeActionPov(viewer, a);
                povHistory.push_back(a);
            }

            ctx.nnInput = StateEncoder<GT>::encode(povState, povHistory);
        }

        // ----------------------------------------------------------------
        void buildFullHashHistory(const AlignedVec<uint64_t>& pathHashes,
            std::vector<uint64_t>& out) const
        {
            out.clear();
            out.insert(out.end(), m_realHashHistory.begin(), m_realHashHistory.end());
            out.insert(out.end(), pathHashes.begin(), pathHashes.end());
        }

        // ----------------------------------------------------------------
        // Root exploration — Gumbel-Top-K if gumbelK > 0, Dirichlet otherwise.
        // Only called on the root node (leaf == m_rootIdx in backprop).
        // ----------------------------------------------------------------
        void applyRootExploration(uint32_t startIdx, uint32_t nChildren)
        {
            if (nChildren == 0) return;

            const uint32_t k = m_config.gumbelK;

            if (k > 0) {
                // ---- Gumbel-Top-K ----
                // More principled than Dirichlet: exploration is proportional
                // to the policy prior rather than uniform noise.
                Strategy::applyGumbelTopK(
                    startIdx, nChildren,
                    m_nodePrior.data(),   // priors modified in-place
                    k);
            }
            else if (m_config.dirichletEpsilon > 0.0f) {
                // ---- Classic Dirichlet (fallback when gumbelK = 0) ----
                thread_local std::mt19937 rng{ std::random_device{}() };
                std::gamma_distribution<float> gamma(m_config.dirichletAlpha, 1.0f);

                float sum = 0.0f;
                std::vector<float> noise(nChildren);
                for (uint32_t i = 0; i < nChildren; ++i) {
                    noise[i] = gamma(rng);
                    sum += noise[i];
                }

                if (sum > 1e-9f) {
                    const float invSum = 1.0f / sum;
                    const float eps = m_config.dirichletEpsilon;
                    for (uint32_t i = 0; i < nChildren; ++i) {
                        float n = noise[i] * invSum;
                        float p = m_nodePrior[startIdx + i];
                        m_nodePrior[startIdx + i] = (1.0f - eps) * p + eps * n;
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        static void copyWDLFromResult(const GameResult& src,
            std::array<float, Defs::kNumPlayers * 3>& dst) noexcept
        {
            dst = src.wdl;
        }

    public:
        TreeSearch(std::shared_ptr<IEngine<GT>> engine, const EngineConfig& cfg)
            : m_config(cfg), m_engine(engine)
            , m_nodeFlags(reserve_only, cfg.maxNodes)
            , m_nodeNumChildren(reserve_only, cfg.maxNodes)
            , m_nodeFirstChild(reserve_only, cfg.maxNodes)
            , m_nodeEdges(reserve_only, cfg.maxNodes)
            , m_nodePrior(reserve_only, cfg.maxNodes)
            , m_nodeAction(reserve_only, cfg.maxNodes)
            , m_realHistory(reserve_only, Defs::kMaxHistory * 2 + 512)
        {
            m_realHashHistory.reserve(Defs::kMaxHistory * 2 + 512);
            for (uint32_t i = 0; i < cfg.maxNodes; ++i) {
                m_nodeFlags.emplace_back(FLAG_NONE);
                m_nodeNumChildren.emplace_back(0);
                m_nodeFirstChild.emplace_back(0);
                m_nodeEdges.emplace_back();
                m_nodePrior.emplace_back(0.0f);
                m_nodeAction.emplace_back();
            }
        }

        void resetCounters()
        {
            m_simulationsLaunched.store(0, std::memory_order_relaxed);
            m_simulationsFinished.store(0, std::memory_order_relaxed);
        }

        uint32_t incrementLaunched()
        {
            return m_simulationsLaunched.fetch_add(1, std::memory_order_relaxed) + 1;
        }

        [[nodiscard]] uint32_t getLaunchedCount()   const { return m_simulationsLaunched.load(std::memory_order_relaxed); }
        [[nodiscard]] uint32_t getSimulationCount() const { return m_simulationsFinished.load(std::memory_order_relaxed); }

        void startSearch(const State& rootState, std::span<const uint64_t> currentHistory)
        {
            m_nodeCount.store(0, std::memory_order_relaxed);
            resetCounters();
            m_realHashHistory.assign(currentHistory.begin(), currentHistory.end());
            m_realHistory.clear();

            m_rootIdx = allocNodes(1);
            if (m_rootIdx != UINT32_MAX) {
                m_nodeFlags[m_rootIdx].val.store(FLAG_NONE, std::memory_order_relaxed);
                m_nodeNumChildren[m_rootIdx].val.store(0, std::memory_order_relaxed);
            }
            m_rootState = rootState;
        }

        // ================================================================
        // PHASE 1: GATHER
        // ================================================================
        bool gather(Event& ctx)
        {
            ctx.reset();
            if (m_rootIdx == UINT32_MAX) return false;

            uint32_t currIdx = m_rootIdx;
            State    currState = m_rootState;
            ctx.path.push_back(currIdx);
            uint32_t depth = 0;

            std::vector<uint64_t> fullHash;
            fullHash.reserve(m_realHashHistory.size() + m_config.maxDepth);

            while (true)
            {
                uint8_t flags = m_nodeFlags[currIdx].val.load(std::memory_order_acquire);

                if (flags & FLAG_TERMINAL) {
                    ctx.isTerminal = true;
                    ctx.leafNodeIdx = currIdx;
                    buildFullHashHistory(ctx.pathHashes, fullHash);
                    if (auto outcome = m_engine->getGameResult(currState, fullHash))
                        copyWDLFromResult(*outcome, ctx.trueWDL);
                    else
                        ctx.trueWDL.fill(0.0f);
                    return false;
                }

                if (!(flags & FLAG_EXPANDED)) {
                    uint8_t expected = FLAG_NONE;
                    if (m_nodeFlags[currIdx].val.compare_exchange_strong(
                        expected, FLAG_EXPANDING, std::memory_order_acquire))
                    {
                        ctx.leafNodeIdx = currIdx;
                        buildFullHashHistory(ctx.pathHashes, fullHash);

                        if (auto outcome = m_engine->getGameResult(currState, fullHash)) {
                            ctx.isTerminal = true;
                            copyWDLFromResult(*outcome, ctx.trueWDL);
                            m_nodeFlags[currIdx].val.store(
                                FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                            return false;
                        }

                        ctx.isTerminal = false;
                        prepareNodeInput(ctx, currState);
                        ctx.validActions = m_engine->getValidActions(currState, fullHash);
                        return true;
                    }
                    else {
                        if (expected & FLAG_EXPANDED) continue;
                        ctx.collision = true;
                        ctx.isTerminal = true;
                        ctx.leafNodeIdx = currIdx;
                        ctx.trueWDL.fill(0.0f);
                        return false;
                    }
                }

                uint32_t nChildren = m_nodeNumChildren[currIdx].val.load(std::memory_order_relaxed);
                if (nChildren == 0) {
                    ctx.isTerminal = true;
                    ctx.leafNodeIdx = currIdx;
                    ctx.trueWDL.fill(0.0f);
                    return false;
                }

                uint32_t firstChild = m_nodeFirstChild[currIdx];
                uint32_t bestChild = firstChild;
                float    bestScore = -std::numeric_limits<float>::max();
                uint32_t parentVisits = std::max(1u, static_cast<uint32_t>(
                    Strategy::getPolicyMetric(m_nodeEdges[currIdx])));

                for (uint32_t i = 0; i < nChildren; ++i) {
                    uint32_t cIdx = firstChild + i;
                    float    score = Strategy::computeScore(
                        m_nodeEdges[cIdx], parentVisits,
                        m_nodePrior[cIdx], m_config.cPUCT, m_config.fpuValue);
                    if (score > bestScore) { bestScore = score; bestChild = cIdx; }
                }

                Strategy::applyVirtualLoss(m_nodeEdges[bestChild], m_config.virtualLoss);
                m_engine->applyAction(m_nodeAction[bestChild], currState);

                ctx.pathHashes.push_back(currState.hash());
                ctx.pathActions.push_back(m_nodeAction[bestChild]);
                currIdx = bestChild;
                ctx.path.push_back(currIdx);

                if (++depth >= m_config.maxDepth) {
                    ctx.isTerminal = true;
                    ctx.leafNodeIdx = currIdx;
                    ctx.trueWDL.fill(0.0f);
                    return false;
                }
            }
        }

        // ================================================================
        // PHASE 2: BACKPROPAGATION
        // ================================================================
        void backprop(const Event& ctx)
        {
            const uint32_t leaf = ctx.leafNodeIdx;

            if (!ctx.collision) {
                uint8_t flags = m_nodeFlags[leaf].val.load(std::memory_order_relaxed);

                if (flags & FLAG_EXPANDING) {
                    if (ctx.validActions.empty() || ctx.isTerminal) {
                        m_nodeFlags[leaf].val.store(
                            FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                    }
                    else {
                        const uint32_t nChildren = static_cast<uint32_t>(ctx.validActions.size());
                        const uint32_t startIdx = allocNodes(nChildren);

                        if (startIdx != UINT32_MAX) {
                            for (uint32_t i = 0; i < nChildren; ++i) {
                                m_nodeAction[startIdx + i] = ctx.validActions[i];
                                uint32_t aId = m_engine->actionToIdx(ctx.validActions[i]);
                                m_nodePrior[startIdx + i] =
                                    (aId < Defs::kActionSpace) ? ctx.policy[aId] : 0.0f;
                                m_nodeFlags[startIdx + i].val.store(FLAG_NONE, std::memory_order_relaxed);
                            }

                            // Root exploration: Gumbel-Top-K or Dirichlet
                            if (leaf == m_rootIdx)
                                applyRootExploration(startIdx, nChildren);

                            m_nodeFirstChild[leaf] = startIdx;
                            m_nodeNumChildren[leaf].val.store(
                                static_cast<uint16_t>(nChildren), std::memory_order_relaxed);
                            m_nodeFlags[leaf].val.store(FLAG_EXPANDED, std::memory_order_release);
                        }
                        else {
                            m_nodeFlags[leaf].val.store(
                                FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                        }
                    }
                }
            }

            // W - L scalar per player
            std::array<float, Defs::kNumPlayers> scalars{};
            for (size_t p = 0; p < Defs::kNumPlayers; ++p)
                scalars[p] = ctx.scalarValue(p, /*fromNN=*/!ctx.isTerminal);

            for (int i = static_cast<int>(ctx.path.size()) - 1; i >= 1; --i) {
                const uint32_t nodeIdx = ctx.path[i];
                const uint32_t playerWhoMoved = ctx.pathActions[i - 1].ownerId();

                Strategy::removeVirtualLoss(m_nodeEdges[nodeIdx], m_config.virtualLoss);

                if (!ctx.collision && playerWhoMoved < Defs::kNumPlayers)
                    Strategy::update(m_nodeEdges[nodeIdx], scalars[playerWhoMoved]);
            }

            if (ctx.collision)
                m_simulationsLaunched.fetch_sub(1, std::memory_order_relaxed);
            else
                m_simulationsFinished.fetch_add(1, std::memory_order_release);
        }

        // ================================================================
        // UTILITIES
        // ================================================================
        void advanceRoot(const Action& actionPlayed, const State& newState)
        {
            m_realHistory.push_back(actionPlayed);
            m_realHashHistory.push_back(newState.hash());

            bool reused = false;
            if (m_config.reuseTree && m_rootIdx != UINT32_MAX) {
                uint8_t flags = m_nodeFlags[m_rootIdx].val.load(std::memory_order_acquire);
                if (flags & FLAG_EXPANDED) {
                    uint32_t start = m_nodeFirstChild[m_rootIdx];
                    uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
                    for (uint32_t i = 0; i < num; ++i) {
                        if (m_nodeAction[start + i] == actionPlayed) {
                            m_rootIdx = start + i;
                            m_rootState = newState;
                            resetCounters();
                            reused = true;
                            // Re-apply root exploration on the new root
                            uint32_t ns = m_nodeFirstChild[m_rootIdx];
                            uint32_t nn = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
                            if (nn > 0) applyRootExploration(ns, nn);
                            break;
                        }
                    }
                }
            }
            if (!reused) startSearch(newState, m_realHashHistory);
        }

        [[nodiscard]] Action selectMove(float temperature)
        {
            if (m_rootIdx == UINT32_MAX) return Action{};
            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            if (num == 0) return Action{};

            uint32_t start = m_nodeFirstChild[m_rootIdx];
            std::vector<double> weights(num);
            double sum = 0.0;

            for (uint32_t i = 0; i < num; ++i) {
                double count = static_cast<double>(Strategy::getPolicyMetric(m_nodeEdges[start + i]));
                double w = (temperature < 1e-3f) ? count : std::pow(count, 1.0 / temperature);
                weights[i] = w;
                sum += w;
            }

            if (temperature < 1e-3f) {
                uint32_t best = 0;
                for (uint32_t i = 1; i < num; ++i)
                    if (weights[i] > weights[best]) best = i;
                return m_nodeAction[start + best];
            }

            thread_local std::mt19937 gen{ std::random_device{}() };
            std::uniform_real_distribution<double> dist(0.0, sum);
            double val = dist(gen), run = 0.0;
            for (uint32_t i = 0; i < num; ++i) {
                run += weights[i];
                if (run >= val) return m_nodeAction[start + i];
            }
            return m_nodeAction[start + num - 1];
        }

        // ----------------------------------------------------------------
        // getRootValue — weighted mean Q (W-L) over root children.
        // Used by SelfPlayHandler::checkResign().
        // ----------------------------------------------------------------
        [[nodiscard]] float getRootValue() const
        {
            if (m_rootIdx == UINT32_MAX) return 0.0f;
            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            if (num == 0) return 0.0f;

            uint32_t start = m_nodeFirstChild[m_rootIdx];
            float    weightedSum = 0.0f;
            float    totalVisits = 0.0f;

            for (uint32_t i = 0; i < num; ++i) {
                float v = static_cast<float>(Strategy::getPolicyMetric(m_nodeEdges[start + i]));
                float q = Strategy::getQ(m_nodeEdges[start + i]);
                weightedSum += v * q;
                totalVisits += v;
            }
            return (totalVisits < 1.0f) ? 0.0f : (weightedSum / totalVisits);
        }

        // ----------------------------------------------------------------
        // getRootPolicy
        //
        // If gumbelSigma > 0: returns the IMPROVED policy target computed
        // from completed Q-values (Gumbel MuZero policy improvement step).
        // This produces a significantly better training signal per simulation.
        //
        // If gumbelSigma == 0: falls back to raw visit counts (classic AlphaZero).
        // ----------------------------------------------------------------
        [[nodiscard]] std::array<float, Defs::kActionSpace> getRootPolicy() const
        {
            std::array<float, Defs::kActionSpace> pol{};
            pol.fill(0.0f);

            if (m_rootIdx == UINT32_MAX) return pol;
            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            if (num == 0) return pol;

            uint32_t start = m_nodeFirstChild[m_rootIdx];

            if (m_config.gumbelSigma > 0.0f) {
                // ---- Gumbel improved policy target ----
                Strategy::computeImprovedPolicy(
                    start, num,
                    m_nodePrior.data(),
                    m_nodeEdges.data(),
                    m_config.gumbelSigma,
                    [this, start](uint32_t offset) -> uint32_t {
                        return m_engine->actionToIdx(m_nodeAction[start + offset]);
                    },
                    Defs::kActionSpace,
                    pol.data()
                );
            }
            else {
                // ---- Classic visit-count policy target ----
                for (uint32_t i = 0; i < num; ++i) {
                    float    v = static_cast<float>(Strategy::getPolicyMetric(m_nodeEdges[start + i]));
                    uint32_t id = m_engine->actionToIdx(m_nodeAction[start + i]);
                    if (id < Defs::kActionSpace) pol[id] += v;
                }
            }

            return pol;
        }

        [[nodiscard]] std::array<bool, Defs::kActionSpace> getRootLegalMovesMask() const
        {
            std::array<bool, Defs::kActionSpace> mask{};
            mask.fill(false);
            if (m_rootIdx == UINT32_MAX) return mask;
            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            uint32_t start = m_nodeFirstChild[m_rootIdx];
            for (uint32_t i = 0; i < num; ++i) {
                uint32_t id = m_engine->actionToIdx(m_nodeAction[start + i]);
                if (id < Defs::kActionSpace) mask[id] = true;
            }
            return mask;
        }

        [[nodiscard]] float getMemoryUsage() const
        {
            return static_cast<float>(m_nodeCount.load(std::memory_order_relaxed))
                / static_cast<float>(m_config.maxNodes);
        }
    };
}