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
    // Allows primitive types to be used atomically inside STL containers.
    // ========================================================================
    template<typename T>
    struct AtomicVal {
        std::atomic<T> val;
        AtomicVal(T v = 0) : val(v) {}
        AtomicVal(const AtomicVal& o) { val.store(o.val.load(std::memory_order_relaxed), std::memory_order_relaxed); }
        AtomicVal& operator=(const AtomicVal& o) { val.store(o.val.load(std::memory_order_relaxed), std::memory_order_relaxed); return *this; }
    };

    // ========================================================================
    // NODE EVENT (The Context passed between ThreadPool and MCTS)
    // ========================================================================
    template<ValidGameTraits GT>
    struct NodeEvent
    {
        USING_GAME_TYPES(GT);

        uint32_t leafNodeIdx;

        AlignedVec<uint32_t> path;
        AlignedVec<Action>   pathActions;
        AlignedVec<uint64_t> pathHashes;

        // The pre-encoded, zero-copy Neural Network input array
        std::array<float, Defs::kNNInputSize> nnInput{};

        bool isTerminal;
        bool collision;

        ActionList validActions;

        // Neural Network outputs mapping (Safely zero-initialized)
        std::array<float, Defs::kActionSpace> policy{};
        GameResult values{};

        explicit NodeEvent(uint32_t maxDepth)
            : leafNodeIdx(0)
            , path(reserve_only, maxDepth)
            , pathActions(reserve_only, maxDepth)
            , pathHashes(reserve_only, maxDepth)
            , isTerminal(false)
            , collision(false)
        {
        }

        void reset() {
            path.clear();
            pathActions.clear();
            pathHashes.clear();
            validActions.clear();

            policy.fill(0.0f);
            values.fill(0.0f);

            isTerminal = false;
            collision = false;
            leafNodeIdx = 0;
        }
    };

    // ========================================================================
    // MONTE CARLO TREE SEARCH (Lock-Free Asynchronous Engine)
    // ========================================================================
    template<ValidGameTraits GT>
    class TreeSearch
    {
    private:
        USING_GAME_TYPES(GT);
        using Event = NodeEvent<GT>;
        using Strategy = StrategyPUCT;
        using EdgeData = typename Strategy::EdgeData;

        // Node Lifecycle Flags
        static constexpr uint8_t FLAG_NONE = 0x00;
        static constexpr uint8_t FLAG_EXPANDING = 0x01;
        static constexpr uint8_t FLAG_EXPANDED = 0x02;
        static constexpr uint8_t FLAG_TERMINAL = 0x04;

        const EngineConfig m_config;
        std::shared_ptr<IEngine<GT>> m_engine;

        // Flattened tree structure optimized for contiguous cache locality
        AlignedVec<AtomicVal<uint8_t>>  m_nodeFlags;
        AlignedVec<AtomicVal<uint16_t>> m_nodeNumChildren;
        AlignedVec<uint32_t> m_nodeFirstChild;
        AlignedVec<EdgeData> m_nodeEdges;
        AlignedVec<float>    m_nodePrior;
        AlignedVec<Action>   m_nodeAction;

        State m_rootState;
        uint32_t m_rootIdx = UINT32_MAX;
        AlignedVec<Action> m_realHistory;
        std::vector<uint64_t> m_realHashHistory;

        std::atomic<uint32_t> m_nodeCount{ 0 };

        // Multi-threading synchronization counters
        std::atomic<uint32_t> m_simulationsLaunched{ 0 };
        std::atomic<uint32_t> m_simulationsFinished{ 0 };

        // --------------------------------------------------------------------
        // Internal Allocator
        // --------------------------------------------------------------------
        uint32_t allocNodes(uint32_t count) {
            uint32_t idx = m_nodeCount.fetch_add(count, std::memory_order_relaxed);
            return (idx + count > m_config.maxNodes) ? UINT32_MAX : idx;
        }

        // --------------------------------------------------------------------
        // Network Payload Preparation & POV Alignment
        // --------------------------------------------------------------------
        void prepareNodeInput(Event& ctx, const State& leafState) {
            uint32_t viewer = m_engine->getCurrentPlayer(leafState);

            // 1. Shift the state perspective
            State povState = leafState;
            m_engine->changeStatePov(viewer, povState);

            // 2. Extract the sliding window of history
            size_t totalNeeded = Defs::kMaxHistory;
            size_t realCount = m_realHistory.size();
            size_t pathCount = ctx.pathActions.size();
            size_t totalAvailable = realCount + pathCount;

            size_t startIdx = (totalAvailable > totalNeeded) ? (totalAvailable - totalNeeded) : 0;
            size_t processed = 0;

            AlignedVec<Action> povHistory(reserve_only, totalNeeded);

            for (size_t i = 0; i < realCount; ++i) {
                if (processed >= startIdx || (realCount - i + pathCount) <= totalNeeded) {
                    Action a = m_realHistory[i];
                    m_engine->changeActionPov(viewer, a);
                    povHistory.push_back(a);
                }
                processed++;
            }
            for (size_t i = 0; i < pathCount; ++i) {
                Action a = ctx.pathActions[i];
                m_engine->changeActionPov(viewer, a);
                povHistory.push_back(a);
            }

            // 3. Immediately encode to float array for TensorRT
            ctx.nnInput = StateEncoder<GT>::encode(povState, povHistory);
        }

        void buildFullHashHistory(const AlignedVec<uint64_t>& pathHashes, std::vector<uint64_t>& outFullHistory) const {
            outFullHistory.clear();
            outFullHistory.insert(outFullHistory.end(), m_realHashHistory.begin(), m_realHashHistory.end());
            outFullHistory.insert(outFullHistory.end(), pathHashes.begin(), pathHashes.end());
        }

        // --------------------------------------------------------------------
        // Exploration (Dirichlet Noise Injection)
        // --------------------------------------------------------------------
        void applyDirichletNoise(uint32_t startIdx, uint32_t nChildren) {
            thread_local std::mt19937 rng(std::random_device{}());
            std::gamma_distribution<float> gamma(m_config.dirichletAlpha, 1.0f);

            float sum = 0.0f;
            std::vector<float> noise(nChildren);
            for (uint32_t i = 0; i < nChildren; ++i) {
                noise[i] = gamma(rng);
                sum += noise[i];
            }

            float epsilon = m_config.dirichletEpsilon;
            if (sum > 1e-9f) {
                float invSum = 1.0f / sum;
                for (uint32_t i = 0; i < nChildren; ++i) {
                    float n = noise[i] * invSum;
                    float p = m_nodePrior[startIdx + i];
                    m_nodePrior[startIdx + i] = (1.0f - epsilon) * p + epsilon * n;
                }
            }
        }

    public:
        TreeSearch(std::shared_ptr<IEngine<GT>> engine, const EngineConfig& cfg)
            : m_config(cfg), m_engine(engine)
            , m_nodeFlags(reserve_only, cfg.maxNodes), m_nodeNumChildren(reserve_only, cfg.maxNodes)
            , m_nodeFirstChild(reserve_only, cfg.maxNodes), m_nodeEdges(reserve_only, cfg.maxNodes)
            , m_nodePrior(reserve_only, cfg.maxNodes), m_nodeAction(reserve_only, cfg.maxNodes)
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

        uint32_t incrementLaunched() {
            return m_simulationsLaunched.fetch_add(1, std::memory_order_relaxed) + 1;
        }

        [[nodiscard]] uint32_t getLaunchedCount() const {
            return m_simulationsLaunched.load(std::memory_order_relaxed);
        }

        [[nodiscard]] uint32_t getSimulationCount() const {
            return m_simulationsFinished.load(std::memory_order_relaxed);
        }

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

        // --------------------------------------------------------------------
        // PHASE 1: GATHER (Traverse tree, apply Virtual Loss, prepare Neural input)
        // --------------------------------------------------------------------
        bool gather(Event& ctx) {
            ctx.reset();
            if (m_rootIdx == UINT32_MAX) return false;

            uint32_t currIdx = m_rootIdx;
            State currState = m_rootState;

            ctx.path.push_back(currIdx);
            uint32_t depth = 0;

            std::vector<uint64_t> fullHashHistory;
            fullHashHistory.reserve(m_realHashHistory.size() + m_config.maxDepth);

            while (true) {
                uint8_t flags = m_nodeFlags[currIdx].val.load(std::memory_order_acquire);

                if (flags & FLAG_TERMINAL) {
                    ctx.isTerminal = true;
                    ctx.leafNodeIdx = currIdx;

                    buildFullHashHistory(ctx.pathHashes, fullHashHistory);
                    if (auto outcome = m_engine->getGameResult(currState, fullHashHistory)) {
                        ctx.values = *outcome;
                    }
                    else {
                        ctx.values.fill(0.0f);
                    }
                    return false;
                }

                if (!(flags & FLAG_EXPANDED)) {
                    uint8_t expected = FLAG_NONE;

                    // Attempt to lock the node for expansion
                    if (m_nodeFlags[currIdx].val.compare_exchange_strong(expected, FLAG_EXPANDING, std::memory_order_acquire)) {
                        ctx.leafNodeIdx = currIdx;
                        buildFullHashHistory(ctx.pathHashes, fullHashHistory);

                        if (auto outcome = m_engine->getGameResult(currState, fullHashHistory)) {
                            ctx.isTerminal = true;
                            ctx.values = *outcome;
                            m_nodeFlags[currIdx].val.store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                            return false;
                        }

                        ctx.isTerminal = false;
                        prepareNodeInput(ctx, currState);
                        ctx.validActions = m_engine->getValidActions(currState, fullHashHistory);

                        return true; // Node needs GPU evaluation
                    }
                    else {
                        // Another thread just finished expanding it
                        if (expected & FLAG_EXPANDED) continue;

                        // Collision: Prevent deadlock by throwing the collision flag
                        ctx.collision = true;
                        ctx.isTerminal = true;
                        ctx.leafNodeIdx = currIdx;
                        ctx.values.fill(0.0f);
                        return false;
                    }
                }

                uint32_t nChildren = m_nodeNumChildren[currIdx].val.load(std::memory_order_relaxed);

                if (nChildren == 0) {
                    ctx.isTerminal = true;
                    ctx.leafNodeIdx = currIdx;
                    ctx.values.fill(0.0f);
                    return false;
                }

                uint32_t firstChild = m_nodeFirstChild[currIdx];
                uint32_t bestChild = firstChild;
                float bestScore = -std::numeric_limits<float>::max();

                uint32_t parentVisits = std::max(1u, static_cast<uint32_t>(Strategy::getPolicyMetric(m_nodeEdges[currIdx])));

                for (uint32_t i = 0; i < nChildren; ++i) {
                    uint32_t cIdx = firstChild + i;
                    float score = Strategy::computeScore(m_nodeEdges[cIdx], parentVisits, m_nodePrior[cIdx], m_config.cPUCT);
                    if (score > bestScore) {
                        bestScore = score;
                        bestChild = cIdx;
                    }
                }

                const Action& action = m_nodeAction[bestChild];

                // Multi-threading core mechanic: Dynamic Penalty
                Strategy::applyVirtualLoss(m_nodeEdges[bestChild], m_config.virtualLoss);

                m_engine->applyAction(action, currState);

                ctx.pathHashes.push_back(currState.hash());
                ctx.pathActions.push_back(action);

                currIdx = bestChild;
                ctx.path.push_back(currIdx);

                if (++depth >= m_config.maxDepth) {
                    ctx.isTerminal = true;
                    ctx.leafNodeIdx = currIdx;
                    ctx.values.fill(0.0f);
                    return false;
                }
            }
        }

        // --------------------------------------------------------------------
        // PHASE 2: BACKPROPAGATION (Update Tree, Remove VL, Add Noise)
        // --------------------------------------------------------------------
        void backprop(const Event& ctx) {
            uint32_t leaf = ctx.leafNodeIdx;

            // Expand the node only if no collision occurred
            if (!ctx.collision) {
                uint8_t flags = m_nodeFlags[leaf].val.load(std::memory_order_relaxed);

                if (flags & FLAG_EXPANDING) {
                    if (ctx.validActions.empty() || ctx.isTerminal) {
                        m_nodeFlags[leaf].val.store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                    }
                    else {
                        uint32_t nChildren = static_cast<uint32_t>(ctx.validActions.size());
                        uint32_t startIdx = allocNodes(nChildren);

                        if (startIdx != UINT32_MAX) {
                            for (uint32_t i = 0; i < nChildren; ++i) {
                                m_nodeAction[startIdx + i] = ctx.validActions[i];

                                uint32_t actionId = m_engine->actionToIdx(ctx.validActions[i]);
                                m_nodePrior[startIdx + i] = (actionId < Defs::kActionSpace) ? ctx.policy[actionId] : 0.0f;
                                m_nodeFlags[startIdx + i].val.store(FLAG_NONE, std::memory_order_relaxed);
                            }

                            // Inject Exploration Noise if this is the Root Node
                            if (leaf == m_rootIdx && m_config.dirichletEpsilon > 0.0f) {
                                applyDirichletNoise(startIdx, nChildren);
                            }

                            m_nodeFirstChild[leaf] = startIdx;
                            m_nodeNumChildren[leaf].val.store(static_cast<uint16_t>(nChildren), std::memory_order_relaxed);
                            m_nodeFlags[leaf].val.store(FLAG_EXPANDED, std::memory_order_release);
                        }
                        else {
                            m_nodeFlags[leaf].val.store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                        }
                    }
                }
            }

            // Always backpropagate the values and resolve virtual losses
            for (int i = (int)ctx.path.size() - 1; i >= 1; --i) {
                uint32_t nodeIdx = ctx.path[i];
                uint32_t playerWhoMoved = ctx.pathActions[i - 1].ownerId();

                Strategy::removeVirtualLoss(m_nodeEdges[nodeIdx], m_config.virtualLoss);

                if (!ctx.collision && playerWhoMoved < Defs::kNumPlayers) {
                    float playerValue = ctx.values[playerWhoMoved];
                    Strategy::update(m_nodeEdges[nodeIdx], playerValue);
                }
            }

            // Gestion des compteurs de synchronisation
            if (ctx.collision) {
                m_simulationsLaunched.fetch_sub(1, std::memory_order_relaxed);
            }
            else {
                m_simulationsFinished.fetch_add(1, std::memory_order_release);
            }
        }

        // --------------------------------------------------------------------
        // UTILITIES
        // --------------------------------------------------------------------
        void advanceRoot(const Action& actionPlayed, const State& newState) {
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

                            if (m_config.dirichletEpsilon > 0.0f) {
                                uint32_t newStart = m_nodeFirstChild[m_rootIdx];
                                uint32_t newNum = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
                                if (newNum > 0) applyDirichletNoise(newStart, newNum);
                            }
                            break;
                        }
                    }
                }
            }

            if (!reused) {
                startSearch(newState, m_realHashHistory);
            }
        }

        [[nodiscard]] Action selectMove(float temperature) {
            if (m_rootIdx == UINT32_MAX) return Action{};
            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            if (num == 0) return Action{};

            uint32_t start = m_nodeFirstChild[m_rootIdx];
            double weights[Defs::kMaxValidActions];
            double sum = 0.0;

            for (uint32_t i = 0; i < num; ++i) {
                double count = static_cast<double>(Strategy::getPolicyMetric(m_nodeEdges[start + i]));
                double w = (temperature < 1e-3f) ? count : std::pow(count, 1.0 / temperature);
                weights[i] = w;
                sum += w;
            }

            if (temperature < 1e-3f) {
                double maxVal = -1.0;
                uint32_t bestIdx = 0;
                for (uint32_t i = 0; i < num; ++i) {
                    if (weights[i] > maxVal) {
                        maxVal = weights[i];
                        bestIdx = i;
                    }
                }
                return m_nodeAction[start + bestIdx];
            }

            thread_local std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<double> dist(0.0, sum);

            double val = dist(gen);
            double run = 0.0;

            for (uint32_t i = 0; i < num; ++i) {
                run += weights[i];
                if (run >= val) return m_nodeAction[start + i];
            }
            return m_nodeAction[start + num - 1];
        }

        [[nodiscard]] std::array<float, Defs::kActionSpace> getRootPolicy() const {
            std::array<float, Defs::kActionSpace> pol{};
            if (m_rootIdx == UINT32_MAX) return pol;

            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            if (num == 0) return pol;

            uint32_t start = m_nodeFirstChild[m_rootIdx];

            for (uint32_t i = 0; i < num; ++i) {
                float visits = static_cast<float>(Strategy::getPolicyMetric(m_nodeEdges[start + i]));
                uint32_t actionId = m_engine->actionToIdx(m_nodeAction[start + i]);

                if (actionId < Defs::kActionSpace) {
                    pol[actionId] += visits;
                }
            }
            return pol;
        }

        [[nodiscard]] float getMemoryUsage() const {
            return static_cast<float>(m_nodeCount.load(std::memory_order_relaxed)) / static_cast<float>(m_config.maxNodes);
        }
    };
}