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
    template<typename T>
    struct AtomicVal {
        std::atomic<T> val;
        AtomicVal(T v = 0) : val(v) {}
        AtomicVal(const AtomicVal& o) { val.store(o.val.load(std::memory_order_relaxed), std::memory_order_relaxed); }
        AtomicVal& operator=(const AtomicVal& o) { val.store(o.val.load(std::memory_order_relaxed), std::memory_order_relaxed); return *this; }
    };

    template<ValidGameTraits GT>
    struct NodeEvent
    {
        USING_GAME_TYPES(GT);

        uint32_t             leafNodeIdx = 0;
        uint32_t             leafViewer = UINT32_MAX;
        AlignedVec<uint32_t> path;
        AlignedVec<Action>   pathActions;
        AlignedVec<uint64_t> pathHashes;

        AlignedVec<uint64_t> fullHashBuffer;

        std::array<float, Defs::kNNInputSize>    nnInput{};
        std::array<float, Defs::kNumPlayers * 3> nnWDL{};
        std::array<float, Defs::kNumPlayers * 3> trueWDL{};
        std::array<float, Defs::kActionSpace>    policy{};

        ActionList validActions;

        bool isTerminal = false;
        bool collision = false;

        explicit NodeEvent(uint32_t maxDepth)
            : path(reserve_only, maxDepth + 1)
            , pathActions(reserve_only, maxDepth)
            , pathHashes(reserve_only, maxDepth)
            , fullHashBuffer(reserve_only, 2048)
        {
        }

        void reset()
        {
            path.clear();
            pathActions.clear();
            pathHashes.clear();
            validActions.clear();
            fullHashBuffer.clear(); // Clear the content, but capacity remains untouched.

            nnInput.fill(0.0f);
            nnWDL.fill(0.0f);
            trueWDL.fill(0.0f);
            policy.fill(0.0f);

            leafNodeIdx = 0;
            isTerminal = false;
            collision = false;
        }

        [[nodiscard]] float scalarValue(size_t p, bool fromNN) const noexcept {
            const auto& wdl = fromNN ? nnWDL : trueWDL;
            return wdl[p * 3 + 0] - wdl[p * 3 + 2];
        }
    };

    template<ValidGameTraits GT>
    class TreeSearch
    {
    private:
        USING_GAME_TYPES(GT);
        using Event = NodeEvent<GT>;
        using Strategy = StrategyPUCT<GT>; // Updated to match templated Strategy
        using EdgeData = typename Strategy::EdgeData;

        static constexpr uint8_t FLAG_NONE = 0x00;
        static constexpr uint8_t FLAG_EXPANDING = 0x01;
        static constexpr uint8_t FLAG_EXPANDED = 0x02;
        static constexpr uint8_t FLAG_TERMINAL = 0x04;
        static constexpr uint8_t FLAG_GUMBEL_APPLIED = 0x08;

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

        uint32_t allocNodes(uint32_t count) {
            uint32_t idx = m_nodeCount.fetch_add(count, std::memory_order_relaxed);
            return (idx + count > m_config.maxNodes) ? UINT32_MAX : idx;
        }

        void prepareNodeInput(Event& ctx, const State& leafState) {
            uint32_t viewer = m_engine->getCurrentPlayer(leafState);
            ctx.leafViewer = viewer;
            State povState = leafState;
            m_engine->changeStatePov(viewer, povState);

            const size_t totalNeeded = Defs::kMaxHistory;
            const size_t realCount = m_realHistory.size();
            const size_t pathCount = ctx.pathActions.size();

            StaticVec<Action, Defs::kMaxHistory> povHistory;

            // Calculate exactly how many actions to pull from each buffer
            const size_t pathTake = std::min(pathCount, totalNeeded);
            const size_t realTake = std::min(realCount, totalNeeded - pathTake);

            // 1. Fill from the actual game history first
            for (size_t i = realCount - realTake; i < realCount; ++i) {
                Action a = m_realHistory[i];
                m_engine->changeActionPov(viewer, a);
                povHistory.push_back(a);
            }

            // 2. Fill the rest from the current MCTS simulation path
            for (size_t i = pathCount - pathTake; i < pathCount; ++i) {
                Action a = ctx.pathActions[i];
                m_engine->changeActionPov(viewer, a);
                povHistory.push_back(a);
            }

            ctx.nnInput = StateEncoder<GT>::encode(povState, povHistory);
        }

        void buildFullHashHistory(const AlignedVec<uint64_t>& pathHashes, AlignedVec<uint64_t>& out) const {
            out.clear();
            out.insert(out.end(), m_realHashHistory.begin(), m_realHashHistory.end());
            out.insert(out.end(), pathHashes.begin(), pathHashes.end());
        }

        void applyRootExploration(uint32_t nodeIdx) {
            if (m_config.gumbelK == 0) return;
            uint8_t old_flags = m_nodeFlags[nodeIdx].val.fetch_or(FLAG_GUMBEL_APPLIED, std::memory_order_acq_rel);
            if (old_flags & FLAG_GUMBEL_APPLIED) return;

            uint32_t nChildren = m_nodeNumChildren[nodeIdx].val.load(std::memory_order_relaxed);
            if (nChildren == 0) return;

            uint32_t startIdx = m_nodeFirstChild[nodeIdx];
            Strategy::applyGumbelTopK(startIdx, nChildren, m_nodePrior.data(), m_config.gumbelK);
        }

        static void copyWDLFromResult(const GameResult& src, std::array<float, Defs::kNumPlayers * 3>& dst) noexcept {
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

        void resetCounters() {
            m_simulationsLaunched.store(0, std::memory_order_relaxed);
            m_simulationsFinished.store(0, std::memory_order_relaxed);
        }

        uint32_t incrementLaunched() { return m_simulationsLaunched.fetch_add(1, std::memory_order_relaxed) + 1; }
        [[nodiscard]] uint32_t getLaunchedCount()   const { return m_simulationsLaunched.load(std::memory_order_relaxed); }
        [[nodiscard]] uint32_t getSimulationCount() const { return m_simulationsFinished.load(std::memory_order_relaxed); }

        void startSearch(const State& rootState, std::span<const uint64_t> currentHistory) {
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

        bool gather(Event& ctx) {
            ctx.reset();
            if (m_rootIdx == UINT32_MAX) return false;

            uint32_t currIdx = m_rootIdx;
            State    currState = m_rootState;
            ctx.path.push_back(currIdx);
            uint32_t depth = 0;

            while (true) {
                uint8_t flags = m_nodeFlags[currIdx].val.load(std::memory_order_acquire);

                if (flags & FLAG_TERMINAL) {
                    ctx.isTerminal = true;
                    ctx.leafNodeIdx = currIdx;
                    buildFullHashHistory(ctx.pathHashes, ctx.fullHashBuffer);
                    if (auto outcome = m_engine->getGameResult(currState, ctx.fullHashBuffer))
                        copyWDLFromResult(*outcome, ctx.trueWDL);
                    else
                        ctx.trueWDL.fill(0.0f);
                    return false;
                }

                if (!(flags & FLAG_EXPANDED)) {
                    uint8_t expected = FLAG_NONE;
                    if (m_nodeFlags[currIdx].val.compare_exchange_strong(expected, FLAG_EXPANDING, std::memory_order_acquire)) {
                        ctx.leafNodeIdx = currIdx;
                        buildFullHashHistory(ctx.pathHashes, ctx.fullHashBuffer);

                        if (auto outcome = m_engine->getGameResult(currState, ctx.fullHashBuffer)) {
                            ctx.isTerminal = true;
                            copyWDLFromResult(*outcome, ctx.trueWDL);
                            m_nodeFlags[currIdx].val.store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                            return false;
                        }

                        ctx.isTerminal = false;
                        prepareNodeInput(ctx, currState);
                        ctx.validActions = m_engine->getValidActions(currState, ctx.fullHashBuffer);
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
                uint32_t parentVisits = std::max(1u, static_cast<uint32_t>(Strategy::getPolicyMetric(m_nodeEdges[currIdx])));

                for (uint32_t i = 0; i < nChildren; ++i) {
                    uint32_t cIdx = firstChild + i;
                    float    score = Strategy::computeScore(m_nodeEdges[cIdx], parentVisits, m_nodePrior[cIdx], m_config.cPUCT, m_config.fpuValue);
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

        void backprop(const Event& ctx) {
            const uint32_t leaf = ctx.leafNodeIdx;

            if (!ctx.collision) {
                uint8_t flags = m_nodeFlags[leaf].val.load(std::memory_order_relaxed);

                if (flags & FLAG_EXPANDING) {
                    if (ctx.validActions.empty() || ctx.isTerminal) {
                        m_nodeFlags[leaf].val.store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                    }
                    else {
                        const uint32_t nChildren = static_cast<uint32_t>(ctx.validActions.size());
                        const uint32_t startIdx = allocNodes(nChildren);

                        if (startIdx != UINT32_MAX) {
                            for (uint32_t i = 0; i < nChildren; ++i) {
                                m_nodeAction[startIdx + i] = ctx.validActions[i];
                                uint32_t aId = m_engine->actionToIdx(ctx.validActions[i]);
                                m_nodePrior[startIdx + i] = (aId < Defs::kActionSpace) ? ctx.policy[aId] : 0.0f;
                                m_nodeFlags[startIdx + i].val.store(FLAG_NONE, std::memory_order_relaxed);
                                m_nodeEdges[startIdx + i].visitCount.store(0, std::memory_order_relaxed);
                                m_nodeEdges[startIdx + i].totalValue.store(0.0f, std::memory_order_relaxed);
                                m_nodeNumChildren[startIdx + i].val.store(0, std::memory_order_relaxed);                            
                            }
                            m_nodeFirstChild[leaf] = startIdx;
                            m_nodeNumChildren[leaf].val.store(static_cast<uint16_t>(nChildren), std::memory_order_relaxed);

                            if (leaf == m_rootIdx) applyRootExploration(leaf);

                            m_nodeFlags[leaf].val.fetch_or(FLAG_EXPANDED, std::memory_order_release);
                        }
                        else {
                            m_nodeNumChildren[leaf].val.store(0, std::memory_order_relaxed);
                            m_nodeFlags[leaf].val.store(FLAG_EXPANDED, std::memory_order_release);
                        }
                    }
                }
            }

            std::array<float, Defs::kNumPlayers> scalars{};
            if (ctx.isTerminal) {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p) scalars[p] = ctx.scalarValue(p, false);
            }
            else {
                for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                    uint32_t absPlayer = (ctx.leafViewer + p) % Defs::kNumPlayers;
                    scalars[absPlayer] = ctx.scalarValue(p, true);
                }
            }

            for (int i = static_cast<int>(ctx.path.size()) - 1; i >= 1; --i) {
                const uint32_t nodeIdx = ctx.path[i];
                const uint32_t playerWhoMoved = ctx.pathActions[i - 1].ownerId();

                Strategy::removeVirtualLoss(m_nodeEdges[nodeIdx], m_config.virtualLoss);

                if (!ctx.collision && playerWhoMoved < Defs::kNumPlayers)
                    Strategy::update(m_nodeEdges[nodeIdx], scalars[playerWhoMoved]);
            }

            if (ctx.collision) m_simulationsLaunched.fetch_sub(1, std::memory_order_relaxed);
            else m_simulationsFinished.fetch_add(1, std::memory_order_release);
        }

        void advanceRoot(const Action& actionPlayed, const State& newState) {
            m_realHistory.push_back(actionPlayed);
            m_realHashHistory.push_back(newState.hash());

            bool reused = false;
            if (m_config.reuseTree && m_rootIdx != UINT32_MAX &&
                m_nodeCount.load(std::memory_order_relaxed) < (m_config.maxNodes * 0.8f))
            {
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
                            applyRootExploration(m_rootIdx);
                            break;
                        }
                    }
                }
            }
            if (!reused) startSearch(newState, m_realHashHistory);
        }

        [[nodiscard]] Action selectMove(float temperature) {
            if (m_rootIdx == UINT32_MAX) return Action{};
            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            if (num == 0 || num > Defs::kMaxValidActions) return Action{};

            uint32_t start = m_nodeFirstChild[m_rootIdx];

            // OPTIMISATION : Allocation statique sur la pile pour les poids
            std::array<double, Defs::kMaxValidActions> weights;
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

        [[nodiscard]] float getRootValue() const {
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

        [[nodiscard]] std::array<float, Defs::kActionSpace> getRootPolicy() const {
            std::array<float, Defs::kActionSpace> pol{};
            pol.fill(0.0f);

            if (m_rootIdx == UINT32_MAX) return pol;
            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            if (num == 0) return pol;
            uint32_t start = m_nodeFirstChild[m_rootIdx];

            if (m_config.gumbelCScale > 0.0f) {
                Strategy::computeImprovedPolicy(
                    start, num, m_nodePrior.data(), m_nodeEdges.data(),
                    m_config.gumbelCVisit, m_config.gumbelCScale,
                    [this, start](uint32_t offset) -> uint32_t { return m_engine->actionToIdx(m_nodeAction[start + offset]); },
                    Defs::kActionSpace, pol.data());
            }
            else {
                for (uint32_t i = 0; i < num; ++i) {
                    float    v = static_cast<float>(Strategy::getPolicyMetric(m_nodeEdges[start + i]));
                    uint32_t id = m_engine->actionToIdx(m_nodeAction[start + i]);
                    if (id < Defs::kActionSpace) pol[id] += v;
                }
            }
            return pol;
        }

        [[nodiscard]] std::array<bool, Defs::kActionSpace> getRootLegalMovesMask() const {
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

        [[nodiscard]] float getMemoryUsage() const {
            return static_cast<float>(m_nodeCount.load(std::memory_order_relaxed)) / static_cast<float>(m_config.maxNodes);
        }
    };
}