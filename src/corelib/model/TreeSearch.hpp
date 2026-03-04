#pragma once
#include "../interfaces/IEngine.hpp"
#include "../bootstrap/GameConfig.hpp"
#include "../util/PovUtils.hpp"
#include "SearchStrategie.hpp"
#include <atomic>
#include <random>
#include <cstring>
#include <numeric>
#include <limits>
#include <vector>
#include <cmath>
#include <span>
#include <array>

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

        uint32_t leafNodeIdx;

        AlignedVec<uint32_t> path;
        AlignedVec<Action>   pathActions;
        AlignedVec<uint64_t> pathHashes;

        State nnInputState;
        AlignedVec<Action> nnInputHistory;

        bool isTerminal;
        bool collision;

        ActionList validActions;

        // ====================================================================
        // OPTIMISATION : policy est un std::array
        // ====================================================================
        std::array<float, Defs::kActionSpace> policy{};
        GameResult values{};

        NodeEvent(uint16_t historySize, uint16_t maxDepth)
            : leafNodeIdx(0)
            , path(reserve_only, maxDepth)
            , pathActions(reserve_only, maxDepth)
            , pathHashes(reserve_only, maxDepth)
            , nnInputHistory(reserve_only, historySize)
            , isTerminal(false)
            , collision(false)
        {
        }

        void reset() {
            path.clear(); pathActions.clear(); pathHashes.clear();
            nnInputHistory.clear();
            validActions.clear();

            policy.fill(0.0f);
            values.fill(0.0f);

            isTerminal = false; collision = false; leafNodeIdx = 0;
        }
    };

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

        const TreeSearchConfig m_config;
        std::shared_ptr<IEngine<GT>> m_engine;

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
        std::atomic<uint32_t> m_simulationsFinished{ 0 };

        uint32_t allocNodes(uint32_t count) {
            uint32_t idx = m_nodeCount.fetch_add(count, std::memory_order_relaxed);
            return (idx + count > m_config.maxNodes) ? UINT32_MAX : idx;
        }

        void prepareNodeInput(Event& ctx, const State& leafState) {
            uint32_t viewer = m_engine->getCurrentPlayer(leafState);
            ctx.nnInputState = leafState;

            // Le changement de point de vue de l'état est géré par l'Engine
            m_engine->changeStatePov(viewer, ctx.nnInputState);

            ctx.nnInputHistory.clear();
            size_t totalNeeded = m_config.historySize;
            size_t realCount = m_realHistory.size();
            size_t pathCount = ctx.pathActions.size();
            size_t totalAvailable = realCount + pathCount;

            size_t startIdx = (totalAvailable > totalNeeded) ? (totalAvailable - totalNeeded) : 0;
            size_t processed = 0;

            for (size_t i = 0; i < realCount; ++i) {
                if (processed >= startIdx || (realCount - i + pathCount) <= totalNeeded) {
                    Action a = m_realHistory[i];
                    m_engine->changeActionPov(viewer, a);
                    ctx.nnInputHistory.push_back(a);
                }
                processed++;
            }
            for (size_t i = 0; i < pathCount; ++i) {
                Action a = ctx.pathActions[i];
                m_engine->changeActionPov(viewer, a);
                ctx.nnInputHistory.push_back(a);
            }
        }

        void buildFullHashHistory(const AlignedVec<uint64_t>& pathHashes, std::vector<uint64_t>& outFullHistory) const {
            outFullHistory.clear();
            outFullHistory.insert(outFullHistory.end(), m_realHashHistory.begin(), m_realHashHistory.end());
            outFullHistory.insert(outFullHistory.end(), pathHashes.begin(), pathHashes.end());
        }

    public:
        TreeSearch(std::shared_ptr<IEngine<GT>> engine, const TreeSearchConfig& cfg)
            : m_config(cfg), m_engine(engine)
            , m_nodeFlags(reserve_only, cfg.maxNodes), m_nodeNumChildren(reserve_only, cfg.maxNodes)
            , m_nodeFirstChild(reserve_only, cfg.maxNodes), m_nodeEdges(reserve_only, cfg.maxNodes)
            , m_nodePrior(reserve_only, cfg.maxNodes), m_nodeAction(reserve_only, cfg.maxNodes)
            , m_realHistory(reserve_only, cfg.historySize * 2)
        {
            m_realHashHistory.reserve(cfg.historySize * 2 + 256);

            for (uint32_t i = 0; i < cfg.maxNodes; ++i) {
                m_nodeFlags.emplace_back(FLAG_NONE); m_nodeNumChildren.emplace_back(0);
                m_nodeFirstChild.emplace_back(0); m_nodeEdges.emplace_back();
                m_nodePrior.emplace_back(0.0f); m_nodeAction.emplace_back();
            }
        }

        void startSearch(const State& rootState, std::span<const uint64_t> currentHistory) {
            m_nodeCount.store(0, std::memory_order_relaxed);
            m_simulationsFinished.store(0, std::memory_order_relaxed);

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
            State currState = m_rootState;

            ctx.path.push_back(currIdx);
            int depth = 0;

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

                    if (m_nodeFlags[currIdx].val.compare_exchange_strong(expected, FLAG_EXPANDING, std::memory_order_acquire)) {
                        ctx.leafNodeIdx = currIdx;

                        buildFullHashHistory(ctx.pathHashes, fullHashHistory);

                        if (auto outcome = m_engine->getGameResult(currState, fullHashHistory))
                        {
                            ctx.isTerminal = true;
                            ctx.values = *outcome;

                            m_nodeFlags[currIdx].val.store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                            return false;
                        }

                        ctx.isTerminal = false;
                        prepareNodeInput(ctx, currState);
                        ctx.validActions = m_engine->getValidActions(currState, fullHashHistory);

                        return true;
                    }
                    else {
                        if (expected & FLAG_EXPANDED) continue;
                        ctx.collision = true;
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

        void backprop(const Event& ctx) {
            if (ctx.collision) return;

            uint32_t leaf = ctx.leafNodeIdx;
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
                        m_nodeFirstChild[leaf] = startIdx;
                        m_nodeNumChildren[leaf].val.store(static_cast<uint16_t>(nChildren), std::memory_order_relaxed);
                        m_nodeFlags[leaf].val.store(FLAG_EXPANDED, std::memory_order_release);
                    }
                    else {
                        m_nodeFlags[leaf].val.store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
                    }
                }
            }

            for (int i = (int)ctx.path.size() - 1; i >= 1; --i) {
                uint32_t nodeIdx = ctx.path[i];
                uint32_t playerWhoMoved = ctx.pathActions[i - 1].ownerId();

                // 1. On retire IMPÉRATIVEMENT la perte virtuelle qu'on avait posée
                Strategy::removeVirtualLoss(m_nodeEdges[nodeIdx], m_config.virtualLoss);

                // 2. On applique la vraie valeur renvoyée par le réseau
                if (playerWhoMoved < Defs::kNumPlayers) {
                    float playerValue = ctx.values[playerWhoMoved];
                    Strategy::update(m_nodeEdges[nodeIdx], playerValue);
                }
            }
            m_simulationsFinished.fetch_add(1, std::memory_order_release);
        }

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
                            m_rootIdx = start + i; m_rootState = newState;
                            m_simulationsFinished.store(0, std::memory_order_relaxed);
                            reused = true; break;
                        }
                    }
                }
            }

            if (!reused) {
                startSearch(newState, m_realHashHistory);
            }
        }

        Action selectMove(float temperature) {
            if (m_rootIdx == UINT32_MAX) return Action{};
            uint32_t num = m_nodeNumChildren[m_rootIdx].val.load(std::memory_order_relaxed);
            if (num == 0) return Action{};

            uint32_t start = m_nodeFirstChild[m_rootIdx];
            double weights[Defs::kMaxValidActions];
            double sum = 0.0;

            for (uint32_t i = 0; i < num; ++i) {
                double count = static_cast<double>(Strategy::getPolicyMetric(m_nodeEdges[start + i]));
                double w = (temperature < 1e-3f) ? count : std::pow(count, 1.0 / temperature);
                weights[i] = w; sum += w;
            }

            if (temperature < 1e-3f) {
                double maxVal = -1.0; uint32_t bestIdx = 0;
                for (uint32_t i = 0; i < num; ++i) { if (weights[i] > maxVal) { maxVal = weights[i]; bestIdx = i; } }
                return m_nodeAction[start + bestIdx];
            }

            static thread_local std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<double> dist(0.0, sum);
            double val = dist(gen); double run = 0.0;
            for (uint32_t i = 0; i < num; ++i) {
                run += weights[i];
                if (run >= val) return m_nodeAction[start + i];
            }
            return m_nodeAction[start + num - 1];
        }

        uint32_t getSimulationCount() const { return m_simulationsFinished.load(std::memory_order_relaxed); }
        float getMemoryUsage() const { return static_cast<float>(m_nodeCount.load(std::memory_order_relaxed)) / static_cast<float>(m_config.maxNodes); }
    };
}