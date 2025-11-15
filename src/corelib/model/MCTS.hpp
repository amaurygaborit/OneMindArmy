#pragma once
#include "../interfaces/IEngine.hpp"
#include "NeuralNet.cuh"
#include "../util/AtomicOps.hpp"

#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <vector>
#include <atomic>
#include <mutex>
#include <cstring>

template<typename GameTag>
class MCTSThreadPool;

template<typename GameTag>
class MCTS
{
private:
    using GT = ITraits<GameTag>;
    using ObsState = typename ObsStateT<GameTag>;
    using Action = typename ActionT<GameTag>;
    using IdxState = typename IdxStateT<GameTag>;
    using IdxAction = typename IdxActionT<GameTag>;
    using IdxStateAction = typename IdxStateActionT<GameTag>;
    using ModelResults = typename ModelResults<GameTag>;

    static constexpr uint8_t kNumPlayers = GT::kNumPlayers;

    // ============================================================================
    // NODE STRUCTURE - Optimized for cache efficiency
    // ============================================================================
    struct Node
    {
        std::atomic<uint32_t> parentIdx{ UINT32_MAX };
        std::atomic<uint32_t> childOffset{ UINT32_MAX };
        std::atomic<uint16_t> childCount{ 0 };
        std::atomic<uint8_t> flags{ 0 };
        // Flags: [7]=expanding, [2]=terminal, [1]=pinned, [0]=expanded

        inline bool isExpanded() const noexcept {
            return flags.load(std::memory_order_acquire) & 0x01;
        }
        inline bool isPinned() const noexcept {
            return flags.load(std::memory_order_acquire) & 0x02;
        }
        inline bool isTerminal() const noexcept {
            return flags.load(std::memory_order_acquire) & 0x04;
        }
        inline bool isExpanding() const noexcept {
            return flags.load(std::memory_order_acquire) & 0x80;
        }

        inline void setExpanded() noexcept {
            flags.fetch_or(0x01, std::memory_order_release);
        }
        inline void setTerminal() noexcept {
            flags.fetch_or(0x04, std::memory_order_release);
        }
        inline void setPinned(bool v) noexcept {
            if (v) flags.fetch_or(0x02, std::memory_order_acquire);
            else flags.fetch_and(~0x02, std::memory_order_release);
        }

        // Returns true if we won the expansion lock
        inline bool tryLockExpansion() noexcept {
            uint8_t expected = 0;
            return flags.compare_exchange_strong(expected, 0x80,
                std::memory_order_acq_rel, std::memory_order_acquire);
        }

        inline void unlockExpansion() noexcept {
            flags.fetch_and(~0x80, std::memory_order_release);
        }

        inline void reset() noexcept {
            parentIdx.store(UINT32_MAX, std::memory_order_relaxed);
            childOffset.store(UINT32_MAX, std::memory_order_relaxed);
            childCount.store(0, std::memory_order_relaxed);
            flags.store(0, std::memory_order_relaxed);
        }
    };

    // ============================================================================
    // SIMULATION RESULT - What each thread stores for later backprop
    // ============================================================================
    struct SimulationResult
    {
        uint32_t leafNodeIdx;
        std::vector<std::pair<uint32_t, uint16_t>> pathCopy; // Immutable copy of path
        AlignedVec<IdxStateAction> historyCopy; // For inference
        bool isTerminal;
        AlignedVec<float> terminalValues; // Only if terminal

        SimulationResult() : leafNodeIdx(UINT32_MAX), isTerminal(false) {}
    };

    // ============================================================================
    // THREAD-LOCAL DATA - Each thread has its own workspace
    // ============================================================================
    struct ThreadLocalData
    {
        // Current simulation path (mutable during descent)
        std::vector<std::pair<uint32_t, uint16_t>> currentPath;

        // Local batch of completed simulations waiting for inference
        std::vector<SimulationResult> localBatch;
        uint32_t localBatchCapacity;

        // Temporary buffers for single simulation
        AlignedVec<Action> validActionsBuf;
        AlignedVec<float> policyBuf;
        AlignedVec<float> valuesBuf;
        AlignedVec<IdxStateAction> historyBuf;

        // Node recycling
        AlignedVec<uint32_t> localFreeNodes;
        static constexpr size_t kLocalCacheSize = 128;

        ThreadLocalData(size_t maxDepth, uint32_t batchCapacity, uint8_t historySize)
            : localBatchCapacity(batchCapacity)
            , validActionsBuf(reserve_only, GT::kMaxValidActions)
            , policyBuf(reserve_only, GT::kActionSpace, GT::kActionSpace)
            , valuesBuf(reserve_only, kNumPlayers, kNumPlayers)
            , historyBuf(reserve_only, historySize)
            , localFreeNodes(reserve_only, kLocalCacheSize)
        {
            currentPath.reserve(maxDepth);
            localBatch.reserve(batchCapacity);
        }

        void reset() {
            currentPath.clear();
            localBatch.clear();
        }
    };

    friend class MCTSThreadPool<GameTag>;

    // ============================================================================
    // MEMBER VARIABLES
    // ============================================================================

    std::shared_ptr<IEngine<GameTag>> m_engine;

    // Parameters
    const uint32_t m_maxNodes;
    const uint16_t m_maxChildren;
    const float m_cPUCT;
    const float m_virtualLoss;
    const uint16_t m_keepK;
    const uint8_t m_historySize;
    const uint16_t m_maxDepth;

    // Node storage (Structure of Arrays for cache efficiency)
    AlignedVec<Node> m_nodes;
    AlignedVec<ObsState> m_states;

    // Child data (grouped by slot for better cache locality)
    AlignedVec<std::atomic<uint32_t>> m_childNodeIdx;
    AlignedVec<std::atomic<Action>> m_childAction;
    AlignedVec<std::atomic<float>> m_childPrior;
    AlignedVec<std::atomic<uint32_t>> m_childN;
    AlignedVec<std::atomic<float>> m_childW; // [slot * kNumPlayers + player]

    // Root state
    std::atomic<uint32_t> m_rootIdx{ UINT32_MAX };

    // Root history (thread-safe: only modified between searches)
    std::mutex m_rootHistMtx;
    AlignedVec<IdxStateAction> m_rootIdxHist;
    AlignedVec<IdxStateAction> m_cachedRootHist; // Read-only during search

    // Free node pool (striped for reduced contention)
    static constexpr size_t kNumStripes = 8;
    struct FreeStripe {
        std::mutex mtx;
        AlignedVec<uint32_t> nodes;
        char padding[64]; // Cache line padding
    };
    std::array<FreeStripe, kNumStripes> m_freeStripes;

    // Search control
    std::atomic<uint32_t> m_simulationCount{ 0 };
    std::atomic<uint32_t> m_targetSimulations{ 0 };
    std::atomic<bool> m_searchActive{ false };

private:
    // ============================================================================
    // NODE ALLOCATION - Lock-free fast path with striped fallback
    // ============================================================================

    uint32_t allocNode(ThreadLocalData& tld) {
        // Fast path: use local cache
        if (!tld.localFreeNodes.empty()) {
            uint32_t idx = tld.localFreeNodes.back();
            tld.localFreeNodes.pop_back();
            m_nodes[idx].reset();
            return idx;
        }

        // Slow path: refill from global stripes
        size_t stripe = std::hash<std::thread::id>{}(std::this_thread::get_id()) % kNumStripes;

        for (size_t attempt = 0; attempt < kNumStripes; ++attempt) {
            auto& s = m_freeStripes[(stripe + attempt) % kNumStripes];
            std::lock_guard<std::mutex> lock(s.mtx);

            if (!s.nodes.empty()) {
                // Bulk refill for amortization
                size_t toTake = std::min<size_t>(64, s.nodes.size());
                for (size_t i = 1; i < toTake; ++i) {
                    tld.localFreeNodes.push_back(s.nodes.back());
                    s.nodes.pop_back();
                }
                uint32_t idx = s.nodes.back();
                s.nodes.pop_back();
                m_nodes[idx].reset();
                return idx;
            }
        }

        return UINT32_MAX; // OOM - need GC
    }

    void freeNodeGlobal(uint32_t idx) {
        if (idx == UINT32_MAX || idx >= m_maxNodes) return;

        // On ne peut pas connaître l'ID du thread, donc on choisit un stripe
        // (par exemple, basé sur l'idx)
        size_t stripe = idx % kNumStripes;
        auto& s = m_freeStripes[stripe];

        std::lock_guard<std::mutex> lock(s.mtx);
        s.nodes.push_back(idx);
    }

    void freeNode(uint32_t idx, ThreadLocalData& tld) {
        if (idx == UINT32_MAX || idx >= m_maxNodes) return;

        // Add to local cache
        if (tld.localFreeNodes.size() < ThreadLocalData::kLocalCacheSize) {
            tld.localFreeNodes.push_back(idx);
            return;
        }

        // Flush half to global stripe
        size_t stripe = std::hash<std::thread::id>{}(std::this_thread::get_id()) % kNumStripes;
        auto& s = m_freeStripes[stripe];

        std::lock_guard<std::mutex> lock(s.mtx);
        size_t toFlush = tld.localFreeNodes.size() / 2;
        s.nodes.insert(s.nodes.end(),
            tld.localFreeNodes.end() - toFlush,
            tld.localFreeNodes.end());
        tld.localFreeNodes.resize(tld.localFreeNodes.size() - toFlush);
        tld.localFreeNodes.push_back(idx);
    }

    // ============================================================================
    // TREE TRAVERSAL - Select child using UCB with virtual loss
    // ============================================================================

    uint32_t selectChild(uint32_t nodeIdx, const ObsState& nodeState, ThreadLocalData& tld) {
        Node& node = m_nodes[nodeIdx];
        uint32_t base = node.childOffset.load(std::memory_order_acquire);
        uint16_t count = node.childCount.load(std::memory_order_acquire);

        if (count == 0) {
            std::cerr << "ERROR: selectChild called on node with 0 children (node=" << nodeIdx << ")" << std::endl;
            return UINT32_MAX;
        }

        uint8_t curPlayer = m_engine->getCurrentPlayer(nodeState);

        // Compute sqrt(sum(N)) for UCB
        uint32_t sumN = 0;
        for (uint16_t i = 0; i < count; ++i) {
            sumN += m_childN[base + i].load(std::memory_order_acquire);
        }
        float sqrtSumN = std::sqrt(static_cast<float>(std::max(1u, sumN)));

        // Find best child via UCB
        float bestScore = -std::numeric_limits<float>::infinity();
        uint16_t bestSlot = UINT16_MAX;

        for (uint16_t slot = 0; slot < count; ++slot) {
            uint32_t slotIdx = base + slot;

            uint32_t childIdx = m_childNodeIdx[slotIdx].load(std::memory_order_acquire);
            if (childIdx == UINT32_MAX) {
                continue; // Skip invalid children
            }

            float N = static_cast<float>(m_childN[slotIdx].load(std::memory_order_acquire));
            float W = m_childW[slotIdx * kNumPlayers + curPlayer].load(std::memory_order_acquire);
            float Q = (N > 0.5f) ? (W / N) : 0.0f;
            float prior = m_childPrior[slotIdx].load(std::memory_order_acquire);
            float U = m_cPUCT * prior * sqrtSumN / (1.0f + N);
            float score = Q + U;

            if (score > bestScore) {
                bestScore = score;
                bestSlot = slot;
            }
        }

        if (bestSlot == UINT16_MAX) {
            std::cerr << "ERROR: No valid child found in selectChild (node=" << nodeIdx << ", count=" << count << ")" << std::endl;
            return UINT32_MAX;
        }

        // Record in path BEFORE applying virtual loss
        tld.currentPath.push_back({ nodeIdx, bestSlot });

        // Apply virtual loss (atomic operations)
        uint32_t slotIdx = base + bestSlot;
        m_childN[slotIdx].fetch_add(1u, std::memory_order_acq_rel);
        m_childW[slotIdx * kNumPlayers + curPlayer].fetch_sub(m_virtualLoss, std::memory_order_acq_rel);

        uint32_t selectedChild = m_childNodeIdx[slotIdx].load(std::memory_order_acquire);
        return selectedChild;
    }

    // ============================================================================
    // EXPANSION - Create children for a leaf node
    // ============================================================================

    bool expandNode(uint32_t nodeIdx, ObsState& nodeState, ThreadLocalData& tld) {
        Node& node = m_nodes[nodeIdx];

        // Check terminal
        tld.valuesBuf.clear();
        tld.valuesBuf.resize(kNumPlayers);
        if (m_engine->isTerminal(nodeState, tld.valuesBuf)) {
            node.setTerminal();
            node.setExpanded();
            return true; // Terminal
        }

        // Get valid actions
        tld.validActionsBuf.clear();
        m_engine->getValidActions(nodeState, tld.validActionsBuf);

        if (tld.validActionsBuf.empty()) {
            node.setTerminal();
            node.setExpanded();
            m_engine->isTerminal(nodeState, tld.valuesBuf);
            return true; // Terminal
        }

        uint16_t numActions = static_cast<uint16_t>(
            std::min<size_t>(tld.validActionsBuf.size(), m_maxChildren));
        uint32_t base = nodeIdx * m_maxChildren;

        // Allocate and initialize children
        uint16_t allocated = 0;
        for (uint16_t slot = 0; slot < numActions; ++slot) {
            uint32_t childIdx = allocNode(tld);

            if (childIdx == UINT32_MAX) {
                // OOM - mark remaining as invalid
                for (uint16_t i = slot; i < numActions; ++i) {
                    m_childNodeIdx[base + i].store(UINT32_MAX, std::memory_order_release);
                }
                break;
            }

            // Create child state
            ObsState childState = nodeState;
            m_engine->applyAction(tld.validActionsBuf[slot], childState);

            uint32_t slotIdx = base + slot;

            // Initialize child data BEFORE publishing index
            m_states[childIdx] = childState;
            m_nodes[childIdx].parentIdx.store(nodeIdx, std::memory_order_release);

            m_childAction[slotIdx].store(tld.validActionsBuf[slot], std::memory_order_release);

            float uniformPrior = 1.0f / numActions;
            m_childPrior[slotIdx].store(uniformPrior, std::memory_order_release);
            m_childN[slotIdx].store(0u, std::memory_order_release);

            for (uint8_t p = 0; p < kNumPlayers; ++p) {
                m_childW[slotIdx * kNumPlayers + p].store(0.0f, std::memory_order_release);
            }

            // Publish child index LAST (acts as memory barrier)
            m_childNodeIdx[slotIdx].store(childIdx, std::memory_order_release);
            allocated++;
        }

        node.childOffset.store(base, std::memory_order_release);
        node.childCount.store(allocated, std::memory_order_release);
        node.setExpanded();

        return false;
    }

    // ============================================================================
    // HISTORY BUILDING - Prepare input for neural network
    // ============================================================================

    void buildHistory(const ThreadLocalData& tld, AlignedVec<IdxStateAction>& outHistory) {
        outHistory.clear();

        size_t pathSize = tld.currentPath.size();
        size_t pathToUse = std::min<size_t>(pathSize, m_historySize);
        size_t rootToUse = std::min<size_t>(m_cachedRootHist.size(), m_historySize - pathToUse);

        // Add path actions (most recent first)
        for (size_t i = 0; i < pathToUse; ++i) {
            auto [nodeIdx, childSlot] = tld.currentPath[pathSize - 1 - i];

            if (nodeIdx >= m_maxNodes) {
                std::cerr << "ERROR: Invalid nodeIdx in buildHistory: " << nodeIdx << std::endl;
                continue; // Ignorer cette partie du chemin
            }

            Node& node = m_nodes[nodeIdx];
            // CRITICAL FIX: Validate node is still valid
            uint32_t base = node.childOffset.load(std::memory_order_acquire);
            if (base == UINT32_MAX || childSlot >= node.childCount.load(std::memory_order_acquire)) {
                std::cerr << "WARNING: Invalid child in buildHistory, skipping" << std::endl;
                continue;
            }
            // Make atomic copy of state to prevent races
            ObsState localState = m_states[nodeIdx];
            std::atomic_thread_fence(std::memory_order_acquire);

            uint32_t slotIdx = base + childSlot;

            IdxStateAction sa;
            m_engine->obsToIdx(localState, sa.state);

            Action tempAction = m_childAction[slotIdx].load(std::memory_order_acquire);
            m_engine->actionToIdx(tempAction, sa.lastAction);

            outHistory.push_back(sa);
        }

        // Add root history
        outHistory.insert(outHistory.end(),
            m_cachedRootHist.begin(),
            m_cachedRootHist.begin() + rootToUse);

        // Pad with nulls
        if (outHistory.size() < m_historySize) {
            IdxStateAction nullSA;
            std::memset(&nullSA, 0, sizeof(IdxStateAction));
            outHistory.resize(m_historySize, nullSA);
        }
    }

    // ============================================================================
    // BACKPROPAGATION - Update tree statistics
    // ============================================================================

    void backpropagate(const SimulationResult& sim, const AlignedVec<float>& values) {
        // No need to unpin leaf - we removed pinning

        // Walk path from leaf to root
        for (int i = static_cast<int>(sim.pathCopy.size()) - 1; i >= 0; --i) {
            auto [parentIdx, slot] = sim.pathCopy[i];

            if (parentIdx >= m_maxNodes) {
                std::cerr << "ERROR: Invalid parentIdx in backprop: " << parentIdx << std::endl;
                continue;
            }

            Node& parent = m_nodes[parentIdx];
            uint32_t base = parent.childOffset.load(std::memory_order_acquire);

            if (base == UINT32_MAX) {
                // On saute silencieusement. C'est normal si le reroot
                // a libéré ce nœud.
                continue;
            }

            uint32_t slotIdx = base + slot;

            // Revert virtual loss and add real value
            // We did: N += 1, W -= vl during selection
            // Now do: W += (value + vl) so net effect is W += value
            for (uint8_t p = 0; p < kNumPlayers; ++p) {
                float delta = values[p] + m_virtualLoss;
                m_childW[slotIdx * kNumPlayers + p].fetch_add(delta, std::memory_order_acq_rel);
            }
        }
    }

    // ============================================================================
    // GARBAGE COLLECTION - Prune subtrees to reclaim memory
    // ============================================================================

    void freeSubtree(uint32_t nodeIdx) {
        if (nodeIdx == UINT32_MAX || nodeIdx >= m_maxNodes) return;

        AlignedVec<uint32_t> stack;
        stack.reserve(256);
        stack.push_back(nodeIdx);

        while (!stack.empty()) {
            uint32_t idx = stack.back();
            stack.pop_back();

            Node& node = m_nodes[idx];
            uint32_t base = node.childOffset.load(std::memory_order_acquire);
            uint16_t count = node.childCount.load(std::memory_order_acquire);

            // Collect children
            for (uint16_t i = 0; i < count; ++i) {
                uint32_t childIdx = m_childNodeIdx[base + i].load(std::memory_order_acquire);
                if (childIdx != UINT32_MAX && childIdx < m_maxNodes) {
                    stack.push_back(childIdx);
                    m_childNodeIdx[base + i].store(UINT32_MAX, std::memory_order_release);
                }
            }

            node.reset();
            freeNodeGlobal(idx);
        }
    }

    void pruneRoot(ThreadLocalData& tld) {
        /*
        uint32_t rootIdx = m_rootIdx.load(std::memory_order_acquire);
        if (rootIdx == UINT32_MAX || rootIdx >= m_maxNodes) return;

        Node& root = m_nodes[rootIdx];
        uint16_t childCount = root.childCount.load(std::memory_order_acquire);
        if (childCount <= m_keepK) return;

        uint32_t base = root.childOffset.load(std::memory_order_acquire);

        // Build sorted list of children by visit count
        struct Candidate { uint16_t slot; uint32_t visits; };
        std::vector<Candidate> candidates;
        candidates.reserve(childCount);

        for (uint16_t slot = 0; slot < childCount; ++slot) {
            uint32_t childIdx = m_childNodeIdx[base + slot].load(std::memory_order_acquire);
            if (childIdx == UINT32_MAX || m_nodes[childIdx].isPinned()) continue;

            uint32_t visits = m_childN[base + slot].load(std::memory_order_acquire);
            candidates.push_back({ slot, visits });
        }

        if (candidates.size() <= m_keepK) return;

        // Partial sort to find top-K
        std::nth_element(candidates.begin(),
            candidates.begin() + m_keepK,
            candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.visits > b.visits; });

        // Free bottom candidates
        for (size_t i = m_keepK; i < candidates.size(); ++i) {
            uint16_t slot = candidates[i].slot;
            uint32_t childIdx = m_childNodeIdx[base + slot].load(std::memory_order_acquire);
            if (childIdx != UINT32_MAX) {
                freeSubtree(childIdx, tld);
            }
        }
        */
    }

public:
    // ============================================================================
    // CONSTRUCTOR
    // ============================================================================

    MCTS(std::shared_ptr<IEngine<GameTag>> engine,
        uint32_t maxNodes,
        uint8_t historySize,
        uint16_t maxDepth,
        float cPUCT,
        float virtualLoss,
        uint16_t keepK)
        : m_engine(std::move(engine))
        , m_maxNodes(maxNodes)
        , m_maxChildren(GT::kMaxValidActions)
        , m_cPUCT(cPUCT)
        , m_virtualLoss(virtualLoss)
        , m_keepK(keepK)
        , m_historySize(historySize)
        , m_maxDepth(maxDepth)
        , m_nodes(maxNodes)
        , m_states(maxNodes)
        , m_childNodeIdx(maxNodes* m_maxChildren)
        , m_childAction(maxNodes* m_maxChildren)
        , m_childPrior(maxNodes* m_maxChildren)
        , m_childN(maxNodes* m_maxChildren)
        , m_childW(maxNodes* m_maxChildren* kNumPlayers)
        , m_rootIdxHist(reserve_only, historySize)
    {
        // Initialize free node stripes
        size_t nodesPerStripe = (maxNodes + kNumStripes - 1) / kNumStripes;
        for (size_t s = 0; s < kNumStripes; ++s) {
            m_freeStripes[s].nodes.reserve(nodesPerStripe);
            size_t start = s * nodesPerStripe;
            size_t end = std::min<size_t>((s + 1) * nodesPerStripe, maxNodes);
            for (size_t i = start; i < end; ++i) {
                m_freeStripes[s].nodes.push_back(static_cast<uint32_t>(i));
            }
        }

        // Initialize child arrays
        for (size_t i = 0; i < m_childNodeIdx.size(); ++i) {
            m_childNodeIdx[i].store(UINT32_MAX, std::memory_order_relaxed);
            m_childPrior[i].store(0.0f, std::memory_order_relaxed);
            m_childN[i].store(0u, std::memory_order_relaxed);
            m_childAction[i].store(Action{}, std::memory_order_relaxed);
        }
        for (size_t i = 0; i < m_childW.size(); ++i) {
            m_childW[i].store(0.0f, std::memory_order_relaxed);
        }
    }

    ~MCTS() = default;
    MCTS(const MCTS&) = delete;
    MCTS& operator=(const MCTS&) = delete;

    // ============================================================================
    // PUBLIC INTERFACE
    // ============================================================================

    void cacheRootHistory() {
        std::lock_guard<std::mutex> lock(m_rootHistMtx);
        m_cachedRootHist = m_rootIdxHist;
    }

    void startSearch(const ObsState& rootState) {
        // Allocate root
        uint32_t newRoot = UINT32_MAX;

        // MODIFICATION : Essayer tous les stripes, pas seulement stripe 0
        for (size_t attempt = 0; attempt < kNumStripes; ++attempt) {
            auto& s = m_freeStripes[attempt];
            std::lock_guard<std::mutex> lock(s.mtx);
            if (!s.nodes.empty()) {
                newRoot = s.nodes.back();
                s.nodes.pop_back();
                break; // Nœud trouvé
            }
        }

        if (newRoot == UINT32_MAX) {
            // C'est une vraie erreur OOM
            throw std::runtime_error("No nodes available for root");
        }

        m_nodes[newRoot].reset();
        m_states[newRoot] = rootState;

        // CRITICAL: Expand root immediately before search starts
        // This prevents all threads from trying to expand it simultaneously
        auto tempTld = std::make_unique<ThreadLocalData>(m_maxDepth, 16, m_historySize);
        ObsState rootStateCopy = rootState;

        try {
            bool isTerminal = expandNode(newRoot, rootStateCopy, *tempTld);
            if (isTerminal) {
                std::cerr << "WARNING: Root state is terminal!" << std::endl;
            }

            std::cout << "Root expanded with "
                << m_nodes[newRoot].childCount.load(std::memory_order_acquire)
                << " children" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "ERROR expanding root: " << e.what() << std::endl;
            throw;
        }

        m_rootIdx.store(newRoot, std::memory_order_release);
        m_simulationCount.store(0u, std::memory_order_release);
    }

    void run(uint32_t numSimulations, MCTSThreadPool<GameTag>& pool);

    void bestActionFromRoot(Action& out) const {
        uint32_t rootIdx = m_rootIdx.load(std::memory_order_acquire);
        if (rootIdx == UINT32_MAX) {
            throw std::runtime_error("Root not set");
        }

        const Node& root = m_nodes[rootIdx];
        uint16_t count = root.childCount.load(std::memory_order_acquire);
        if (count == 0) {
            throw std::runtime_error("No children at root");
        }

        uint32_t base = root.childOffset.load(std::memory_order_acquire);
        uint32_t bestVisits = 0;
        uint16_t bestSlot = 0;

        for (uint16_t i = 0; i < count; ++i) {
            uint32_t visits = m_childN[base + i].load(std::memory_order_acquire);
            if (visits > bestVisits) {
                bestVisits = visits;
                bestSlot = i;
            }
        }

        out = m_childAction[base + bestSlot].load(std::memory_order_relaxed);

        // Debug output
        static constexpr const char* kSquares[64] = {
            "a1","b1","c1","d1","e1","f1","g1","h1",
            "a2","b2","c2","d2","e2","f2","g2","h2",
            "a3","b3","c3","d3","e3","f3","g3","h3",
            "a4","b4","c4","d4","e4","f4","g4","h4",
            "a5","b5","c5","d5","e5","f5","g5","h5",
            "a6","b6","c6","d6","e6","f6","g6","h6",
            "a7","b7","c7","d7","e7","f7","g7","h7",
            "a8","b8","c8","d8","e8","f8","g8","h8"
        };

        ObsState localState = m_states[rootIdx];
        uint8_t curPlayer = m_engine->getCurrentPlayer(localState);

        std::cout << "\n=== Root Statistics ===" << std::endl;
        for (uint16_t i = 0; i < count; ++i) {
            uint32_t slotIdx = base + i;
            Action a = m_childAction[slotIdx].load(std::memory_order_relaxed);
            uint32_t visits = m_childN[slotIdx].load(std::memory_order_acquire);
            float W = m_childW[slotIdx * kNumPlayers + curPlayer].load(std::memory_order_acquire);
            float prior = m_childPrior[slotIdx].load(std::memory_order_acquire);
            float Q = (visits > 0) ? (W / visits) : 0.0f;

            std::cout << kSquares[a.from()] << kSquares[a.to()] << kSquares[a.promo()]
                << " | N=" << visits << " | Q=" << Q
                << " | W=" << W << " | P=" << prior << std::endl;
        }
        std::cout << "======================\n" << std::endl;
    }

    void rerootByPlayedAction(const Action& played) {
        // CRITICAL: Search MUST be inactive before rerooting
        // Otherwise workers might be accessing nodes we're about to free
        std::atomic_thread_fence(std::memory_order_seq_cst);

        if (m_searchActive.load(std::memory_order_acquire)) {
            std::cerr << "CRITICAL ERROR: rerootByPlayedAction called during active search!" << std::endl;
            throw std::runtime_error("Cannot reroot during active search");
        }

        uint32_t oldRootIdx = m_rootIdx.load(std::memory_order_acquire);
        if (oldRootIdx == UINT32_MAX) {
            std::cerr << "WARNING: rerootByPlayedAction called with no root set" << std::endl;
            return;
        }

        ObsState oldRootState = m_states[oldRootIdx];
        std::atomic_thread_fence(std::memory_order_acquire);

        Node& oldRoot = m_nodes[oldRootIdx];
        uint32_t base = oldRoot.childOffset.load(std::memory_order_acquire);
        uint16_t count = oldRoot.childCount.load(std::memory_order_acquire);

        // Find played action
        uint16_t playedSlot = UINT16_MAX;
        for (uint16_t i = 0; i < count; ++i) {
            Action tempAction = m_childAction[base + i].load(std::memory_order_relaxed);
            if (tempAction == played)
            {
                playedSlot = i;
                break;
            }
        }

        ThreadLocalData tempTld(m_maxDepth, 16, m_historySize);

        if (playedSlot != UINT16_MAX) {
            uint32_t newRootIdx = m_childNodeIdx[base + playedSlot].load(std::memory_order_acquire);

            if (newRootIdx != UINT32_MAX && newRootIdx < m_maxNodes) {
                std::cout << "Reusing subtree at node " << newRootIdx << std::endl;

                // Reuse subtree
                m_nodes[newRootIdx].parentIdx.store(UINT32_MAX, std::memory_order_release);

                // Free siblings (but NOT the played child)
                for (uint16_t i = 0; i < count; ++i) {
                    if (i == playedSlot) continue;
                    uint32_t childIdx = m_childNodeIdx[base + i].load(std::memory_order_acquire);
                    if (childIdx != UINT32_MAX) {
                        freeSubtree(childIdx);
                    }
                }

                // Free old root
                freeNodeGlobal(oldRootIdx);

                // Set new root
                std::atomic_thread_fence(std::memory_order_seq_cst);
                m_rootIdx.store(newRootIdx, std::memory_order_release);

                // Update history
                {
                    std::lock_guard<std::mutex> lock(m_rootHistMtx);
                    IdxStateAction sa;
                    m_engine->obsToIdx(oldRootState, sa.state);
                    m_engine->actionToIdx(played, sa.lastAction);
                    m_rootIdxHist.insert(m_rootIdxHist.begin(), sa);
                    if (m_rootIdxHist.size() > m_historySize) {
                        m_rootIdxHist.resize(m_historySize);
                    }
                }

                std::cout << "Reroot complete. New root: " << newRootIdx
                    << " with " << m_nodes[newRootIdx].childCount.load() << " children" << std::endl;
                return;
            }
        }

        // No reusable subtree - start fresh
        std::cout << "No reusable subtree, starting fresh" << std::endl;

        ObsState newRootState = oldRootState;
        m_engine->applyAction(played, newRootState);

        // Free entire old tree
        freeSubtree(oldRootIdx);

        // Update history
        {
            std::lock_guard<std::mutex> lock(m_rootHistMtx);
            IdxStateAction sa;
            m_engine->obsToIdx(oldRootState, sa.state);
            m_engine->actionToIdx(played, sa.lastAction);
            m_rootIdxHist.insert(m_rootIdxHist.begin(), sa);
            if (m_rootIdxHist.size() > m_historySize) {
                m_rootIdxHist.resize(m_historySize);
            }
        }

        // Start with new state
        startSearch(newRootState);
    }

    // ============================================================================
    // WORKER INTERFACE - Called by thread pool
    // ============================================================================

    // Run one simulation and add to local batch. Returns false if batch is full.
    bool runSimulation(ThreadLocalData& tld) {
        tld.currentPath.clear();

        uint32_t rootIdx = m_rootIdx.load(std::memory_order_acquire);
        if (rootIdx >= m_maxNodes) {
            return true;
        }

        // ====================================================================
        // SELECTION PHASE
        // ====================================================================
        ObsState currentState = m_states[rootIdx];
        std::atomic_thread_fence(std::memory_order_acquire);

        uint32_t curIdx = rootIdx;
        int depth = 0;
        while (depth < m_maxDepth) {
            if (curIdx >= m_maxNodes) {
                cleanupPath(tld);
                return true;
            }

            Node& curNode = m_nodes[curIdx];

            if (curNode.isTerminal()) {
                // Terminal node - get values and backprop immediately
                tld.valuesBuf.clear();
                tld.valuesBuf.resize(kNumPlayers);
                m_engine->isTerminal(currentState, tld.valuesBuf);

                SimulationResult sim;
                sim.leafNodeIdx = curIdx;
                sim.pathCopy = tld.currentPath;
                sim.isTerminal = true;
                sim.terminalValues = tld.valuesBuf;

                backpropagate(sim, sim.terminalValues);
                m_simulationCount.fetch_add(1u, std::memory_order_relaxed);
                return true; // Continue simulating
            }

            if (!curNode.isExpanded()) {
                // Found unexpanded node
                break;
            }

            if (curNode.childCount.load(std::memory_order_acquire) == 0) {
                // C'est une impasse. On ne peut pas descendre plus bas.
                // On annule cette simulation et on en lance une nouvelle.
                cleanupPath(tld); // Annule les virtual loss
                return true; // 'true' = continuer à simuler (lancer une autre simulation)
            }

            // Select best child
            uint32_t nextIdx = selectChild(curIdx, currentState, tld);
            if (nextIdx >= m_maxNodes) {
                cleanupPath(tld);
                return true;
            }

            curIdx = nextIdx;
            currentState = m_states[curIdx];
            std::atomic_thread_fence(std::memory_order_acquire);

            depth++;
        }

        if (depth >= m_maxDepth) {
            cleanupPath(tld);
            return true;
        }

        // ====================================================================
        // EXPANSION PHASE
        // ====================================================================
        Node& leafNode = m_nodes[curIdx];
        bool isTerminal = false;

        if (!leafNode.isExpanded()) {
            if (leafNode.tryLockExpansion()) {
                try {
                    isTerminal = expandNode(curIdx, currentState, tld);
                }
                catch (...) {
                    leafNode.unlockExpansion();
                    cleanupPath(tld);
                    return true;
                }
                leafNode.unlockExpansion();
            }
            else {
                // Wait for expansion
                int spins = 0;
                while (leafNode.isExpanding() && spins++ < 100000) {
                    std::this_thread::yield();
                }

                if (!leafNode.isExpanded()) {
                    cleanupPath(tld);
                    return true;
                }

                isTerminal = leafNode.isTerminal();
                if (isTerminal) {
                    tld.valuesBuf.clear();
                    tld.valuesBuf.resize(kNumPlayers);
                    m_engine->isTerminal(currentState, tld.valuesBuf);
                }
            }
        }

        // ====================================================================
        // PREPARE RESULT
        // ====================================================================
        SimulationResult sim;
        sim.leafNodeIdx = curIdx;
        sim.pathCopy = tld.currentPath;
        sim.isTerminal = isTerminal;

        if (isTerminal) {
            sim.terminalValues = tld.valuesBuf;
            backpropagate(sim, sim.terminalValues);
            m_simulationCount.fetch_add(1u, std::memory_order_relaxed);
            return true;
        }

        if (leafNode.isExpanded() && leafNode.childCount.load(std::memory_order_acquire) == 0)
        {
            // On ne peut ni évaluer (pas un terminal), ni descendre (pas d'enfants).
            // La seule chose à faire est d'annuler cette simulation.
            //std::cerr << "WARNING: OOM detected at node " << curIdx << ". Aborting sim." << std::endl;
            cleanupPath(tld); // Annule les virtual loss
            return true; // 'true' = continuer à simuler (lancer une autre simulation)
        }

        // Build history for inference
        try {
            buildHistory(tld, tld.historyBuf);
            sim.historyCopy = tld.historyBuf;
        }
        catch (...) {
            cleanupPath(tld);
            return true;
        }

        // Add to batch
        tld.localBatch.push_back(std::move(sim));

        // Return false if batch is full
        return tld.localBatch.size() < tld.localBatchCapacity;
    }

private:
    void cleanupPath(ThreadLocalData& tld) {
        // Revert virtual loss for all edges in path
        for (auto [nodeIdx, slot] : tld.currentPath) {
            if (nodeIdx >= m_maxNodes) continue;

            Node& node = m_nodes[nodeIdx];
            uint32_t base = node.childOffset.load(std::memory_order_acquire);
            uint32_t slotIdx = base + slot;

            ObsState localState = m_states[nodeIdx];
            uint8_t curPlayer = m_engine->getCurrentPlayer(localState);

            // Revert: N -= 1, W += virtualLoss
            m_childN[slotIdx].fetch_sub(1u, std::memory_order_acq_rel);
            m_childW[slotIdx * kNumPlayers + curPlayer].fetch_add(
                m_virtualLoss, std::memory_order_acq_rel);
        }
        tld.currentPath.clear();
    }
};