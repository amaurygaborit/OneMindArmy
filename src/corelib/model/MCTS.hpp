#pragma once
#include "../interfaces/IEngine.hpp"
#include "../bootstrap/GameConfig.hpp"
#include "NeuralNet.cuh"
#include "AlignedVec.hpp"

#include <atomic>
#include <cmath>
#include <memory>
#include <cstring>
#include <algorithm>
#include <random>

template<typename GameTag> class MCTSThreadPool;

// === NODE EVENT ===
template<typename GameTag>
struct NodeEvent
{
    using ObsState = typename ObsStateT<GameTag>;
    using IdxStateAction = typename IdxStateActionT<GameTag>;
    using Action = typename ActionT<GameTag>;

    // --- IDENTITÉ & CONTEXTE ---
    uint32_t leafNodeIdx;
    AlignedVec<uint32_t> path;
    AlignedVec<uint8_t> pathPlayers; // Pour revert VirtualLoss correctement

    // --- ÉTAT ---
    ObsState leafState;
    bool isTerminal;
    bool collision;

    // --- DONNÉES POUR L'INFÉRENCE ---
    AlignedVec<IdxStateAction> nnHistory;

    // --- RÉSULTATS ---
    AlignedVec<Action> validActions;
    AlignedVec<float> policy;
    AlignedVec<float> values;

    NodeEvent(uint16_t historySize, uint16_t maxDepth)
    {
        path.reserve(maxDepth);
        pathPlayers.reserve(maxDepth);
        nnHistory.reserve(historySize);
        validActions.reserve(ITraits<GameTag>::kMaxValidActions);
        policy.reserve(ITraits<GameTag>::kActionSpace);
        values.reserve(ITraits<GameTag>::kNumPlayers);
        reset();
    }

    void reset()
    {
        leafNodeIdx = 0;
        path.clear();
        pathPlayers.clear();
        isTerminal = false;
        collision = false;
        nnHistory.clear();
        validActions.clear();
        policy.clear();
        values.clear();
    }
};

template<typename GameTag>
class MCTS
{
public:
    using GT = ITraits<GameTag>;
    using ObsState = typename ObsStateT<GameTag>;
    using Action = typename ActionT<GameTag>;
    using IdxAction = typename IdxActionT<GameTag>;
    using IdxStateAction = typename IdxStateActionT<GameTag>;
    using ModelResults = typename ModelResultsT<GameTag>;
    using Event = NodeEvent<GameTag>;

private:
    static constexpr uint8_t kNumPlayers = GT::kNumPlayers;
    static constexpr uint8_t FLAG_EXPANDED = 0x01;
    static constexpr uint8_t FLAG_EXPANDING = 0x02;
    static constexpr uint8_t FLAG_TERMINAL = 0x04;

    std::shared_ptr<IEngine<GameTag>> m_engine;

    // Config Moteur (maxNodes, reuseTree, cPUCT...)
    const MCTSConfig m_config;

    // Arbre (Structure of Arrays)
    std::atomic<uint32_t>* m_nodeN;
    std::atomic<float>* m_nodeW[kNumPlayers];
    std::atomic<uint8_t>* m_nodeFlags;

    AlignedVec<float>    m_nodePrior;
    AlignedVec<uint32_t> m_nodeFirstChild;
    AlignedVec<uint16_t> m_nodeNumChildren;
    AlignedVec<Action>   m_nodeAction;
    AlignedVec<ObsState> m_nodeStates;

    // État courant
    std::atomic<uint32_t>      m_nodeCount{ 0 };
    std::atomic<uint32_t>      m_finishedSimulations{ 0 };
    uint32_t                   m_rootIdx = UINT32_MAX;
    AlignedVec<IdxStateAction> m_rootHistory;

    void allocateMemory()
    {
        m_nodeN = new std::atomic<uint32_t>[m_config.maxNodes];
        m_nodeFlags = new std::atomic<uint8_t>[m_config.maxNodes];
        for (uint8_t p = 0; p < kNumPlayers; ++p)
            m_nodeW[p] = new std::atomic<float>[m_config.maxNodes];

        for (uint32_t i = 0; i < m_config.maxNodes; ++i)
        {
            m_nodeN[i] = 0;
            m_nodeFlags[i] = 0;
            for (uint8_t p = 0; p < kNumPlayers; ++p)
                m_nodeW[p][i] = 0.0f;
        }

        m_nodePrior.resize(m_config.maxNodes);
        m_nodeFirstChild.resize(m_config.maxNodes);
        m_nodeNumChildren.resize(m_config.maxNodes);
        m_nodeAction.resize(m_config.maxNodes);
        m_nodeStates.resize(m_config.maxNodes);
    }

    void deallocateMemory()
    {
        delete[] m_nodeN;
        delete[] m_nodeFlags;
        for (uint8_t p = 0; p < kNumPlayers; ++p) delete[] m_nodeW[p];
    }

    uint32_t allocNode()
    {
        uint32_t idx = m_nodeCount.fetch_add(1, std::memory_order_relaxed);
        if (idx >= m_config.maxNodes) return UINT32_MAX;

        m_nodeN[idx].store(0, std::memory_order_relaxed);
        m_nodeFlags[idx].store(0, std::memory_order_relaxed);
        for (auto p = 0; p < kNumPlayers; ++p)
            m_nodeW[p][idx].store(0.0f, std::memory_order_relaxed);
        m_nodeNumChildren[idx] = 0;
        return idx;
    }

    uint32_t allocNodes(uint32_t count)
    {
        uint32_t idx = m_nodeCount.fetch_add(count, std::memory_order_relaxed);
        if (idx + count > m_config.maxNodes) return UINT32_MAX;

        for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t ni = idx + i;
            m_nodeN[ni].store(0, std::memory_order_relaxed);
            m_nodeFlags[ni].store(0, std::memory_order_relaxed);
            for (auto p = 0; p < kNumPlayers; ++p)
                m_nodeW[p][ni].store(0.0f, std::memory_order_relaxed);
            m_nodeNumChildren[ni] = 0;
        }
        return idx;
    }

    void applyVirtualLoss(uint32_t nodeIdx, size_t player)
    {
        m_nodeN[nodeIdx].fetch_add(1, std::memory_order_relaxed);
        for (uint8_t p = 0; p < kNumPlayers; ++p)
        {
            // Si c'est le joueur courant, on diminue sa valeur (défaite virtuelle)
            float loss = (p == player) ? -m_config.virtualLoss : m_config.virtualLoss;
            m_nodeW[p][nodeIdx].fetch_add(loss, std::memory_order_relaxed);
        }
    }

    void revertVirtualLoss(const AlignedVec<uint32_t>& path, const AlignedVec<uint8_t>& players)
    {
        // Revert spécifique avec tracking des joueurs (Lc0 style)
        // path[0] est la racine. players[0] est le joueur qui a choisi path[1].
        // La VL a été appliquée sur path[i] (l'enfant) avec le joueur players[i-1] (le parent).
        // Dans gatherWalk, on push le joueur courant AVANT de descendre.

        size_t len = path.size();
        for (size_t i = 0; i < len; ++i)
        {
            uint32_t nodeIdx = path[i];
            uint8_t p = players[i];

            m_nodeN[nodeIdx].fetch_sub(1, std::memory_order_relaxed);

            float loss = (p == p) ? -m_config.virtualLoss : m_config.virtualLoss; // Simplification logique, revert exactement
            // NOTE: Pour simplifier et éviter les bugs de logique, on fait l'inverse exact de applyVirtualLoss :
            // m_nodeW[p] -= -loss  => m_nodeW[p] += loss
            m_nodeW[p][nodeIdx].fetch_sub(loss, std::memory_order_relaxed);
        }
    }

    uint32_t selectBestChild(uint32_t nodeIdx, const ObsState& state)
    {
        uint32_t numChildren = m_nodeNumChildren[nodeIdx];
        if (numChildren == 0) return UINT32_MAX;

        uint32_t bestChild = UINT32_MAX;
        float bestScore = -std::numeric_limits<float>::max();

        uint32_t parentN = m_nodeN[nodeIdx].load(std::memory_order_relaxed);
        float sqrtParentN = std::sqrt(static_cast<float>(parentN + 1));

        size_t player = m_engine->getCurrentPlayer(state);
        uint32_t start = m_nodeFirstChild[nodeIdx];

        for (uint32_t i = 0; i < numChildren; ++i)
        {
            uint32_t childIdx = start + i;
            uint32_t childN = m_nodeN[childIdx].load(std::memory_order_relaxed);

            float q = 0.0f;
            if (childN > 0)
            {
                float w = m_nodeW[player][childIdx].load(std::memory_order_relaxed);
                q = w / static_cast<float>(childN);
            }

            float p = m_nodePrior[childIdx];
            if (std::isnan(p)) p = 0.0f;

            float u = m_config.cPUCT * p * sqrtParentN / (1.0f + childN);

            if (q + u > bestScore)
            {
                bestScore = q + u;
                bestChild = childIdx;
            }
        }
        return bestChild;
    }

    void prepareHistory(const AlignedVec<uint32_t>& path, AlignedVec<IdxStateAction>& outHist)
    {
        outHist.clear();
        size_t rootHistSize = m_rootHistory.size();
        size_t pathSize = path.size();
        size_t totalItems = rootHistSize + (pathSize > 0 ? pathSize - 1 : 0);
        size_t needed = m_config.historySize;

        if (totalItems < needed)
        {
            size_t padCount = needed - totalItems;
            IdxStateAction padItem;
            padItem.idxAction = Fact<GameTag>::MakePad(FactType::ACTION);
            padItem.idxState.elemFacts.fill(Fact<GameTag>::MakePad(FactType::ELEMENT));
            for (size_t i = 0; i < padCount; ++i) outHist.push_back(padItem);
        }

        size_t startOffset = (totalItems > needed) ? (totalItems - needed) : 0;
        size_t currentIdx = 0;

        for (size_t i = 0; i < rootHistSize; ++i)
        {
            if (outHist.size() == needed) break;
            if (currentIdx >= startOffset)
            {
                outHist.push_back(m_rootHistory[i]);
            }
            currentIdx++;
        }
        for (size_t i = 1; i < pathSize; ++i)
        {
            if (outHist.size() == needed) break;
            if (currentIdx >= startOffset)
            {
                IdxStateAction pathItem;
                pathItem.idxState.elemFacts.fill(Fact<GameTag>::MakePad(FactType::ELEMENT));
                uint32_t nodeIdx = path[i];
                m_engine->actionToIdx(m_nodeAction[nodeIdx], pathItem.idxAction);
                outHist.push_back(pathItem);
            }
            currentIdx++;
        }
    }

    bool isActionEqual(const Action& a, const Action& b) const
    {
        return std::memcmp(&a, &b, sizeof(Action)) == 0;
    }

public:
    MCTS(std::shared_ptr<IEngine<GameTag>> engine, const MCTSConfig& config)
        : m_engine(std::move(engine)),
        m_config(config),
        m_nodeN(nullptr),
        m_nodeFlags(nullptr)
    {
        allocateMemory();
        m_rootHistory.reserve(m_config.historySize);
    }

    ~MCTS()
    {
        deallocateMemory();
    }

    // --- ACCESSEURS PUBLICS ---

    std::shared_ptr<IEngine<GameTag>> getEngine() const { return m_engine; }

    uint32_t getSimulationCount() const {
        return m_finishedSimulations.load(std::memory_order_relaxed);
    }

    // --- WORKERS ---

    bool gatherWalk(Event& event)
    {
        if (m_rootIdx == UINT32_MAX || isMemoryFull()) return false;

        event.reset();
        event.path.push_back(m_rootIdx);

        uint32_t currIdx = m_rootIdx;
        ObsState currState = m_nodeStates[m_rootIdx];

        size_t player = m_engine->getCurrentPlayer(currState);
        event.pathPlayers.push_back(static_cast<uint8_t>(player));

        applyVirtualLoss(currIdx, player);

        int depth = 0;
        while (true)
        {
            uint8_t flags = m_nodeFlags[currIdx].load(std::memory_order_acquire);

            if (flags & FLAG_TERMINAL)
            {
                event.leafNodeIdx = currIdx;
                event.leafState = currState;
                event.isTerminal = true;
                m_engine->isTerminal(currState, event.values);
                return true;
            }

            if (!(flags & FLAG_EXPANDED))
            {
                uint8_t expected = flags;
                if (!(expected & FLAG_EXPANDING))
                {
                    if (m_nodeFlags[currIdx].compare_exchange_weak(
                        expected,
                        flags | FLAG_EXPANDING,
                        std::memory_order_acquire))
                    {
                        event.leafNodeIdx = currIdx;
                        event.leafState = currState;
                        event.isTerminal = false;
                        prepareHistory(event.path, event.nnHistory);
                        return true;
                    }
                }
                event.leafNodeIdx = currIdx;
                event.collision = true;
                return true;
            }

            uint32_t bestChild = selectBestChild(currIdx, currState);
            if (bestChild == UINT32_MAX)
            {
                m_nodeFlags[currIdx].fetch_or(FLAG_TERMINAL, std::memory_order_release);
                event.leafNodeIdx = currIdx;
                event.isTerminal = true;
                m_engine->isTerminal(currState, event.values);
                return true;
            }

            // Avancer
            applyVirtualLoss(bestChild, player); // Note: on utilise le joueur courant (parent) pour VL
            m_engine->applyAction(m_nodeAction[bestChild], currState);
            currIdx = bestChild;

            // Nouveau joueur
            player = m_engine->getCurrentPlayer(currState);
            event.path.push_back(currIdx);
            event.pathPlayers.push_back(static_cast<uint8_t>(player));

            if (++depth > m_config.maxDepth) return false;
        }
    }

    void applyBackprop(const Event& event)
    {
        if (event.collision)
        {
            // Revert strict
            size_t len = event.path.size();
            for (size_t i = 0; i < len; ++i) {
                uint32_t idx = event.path[i];
                uint8_t p = event.pathPlayers[i];
                m_nodeN[idx].fetch_sub(1, std::memory_order_relaxed);
                float loss = (p == p) ? -m_config.virtualLoss : m_config.virtualLoss;
                m_nodeW[p][idx].fetch_sub(loss, std::memory_order_relaxed);
            }
            return;
        }

        // Revert Virtual Loss avant update
        size_t len = event.path.size();
        for (size_t i = 0; i < len; ++i) {
            uint32_t idx = event.path[i];
            uint8_t p = event.pathPlayers[i];
            m_nodeN[idx].fetch_sub(1, std::memory_order_relaxed);
            float loss = (p == p) ? -m_config.virtualLoss : m_config.virtualLoss;
            m_nodeW[p][idx].fetch_sub(loss, std::memory_order_relaxed);
        }

        uint32_t leafIdx = event.leafNodeIdx;

        // Expansion
        if (!event.isTerminal && !event.validActions.empty())
        {
            uint32_t nChildren = static_cast<uint32_t>(event.validActions.size());
            uint32_t startIdx = allocNodes(nChildren);

            if (startIdx != UINT32_MAX)
            {
                for (uint32_t i = 0; i < nChildren; ++i)
                {
                    uint32_t childIdx = startIdx + i;
                    m_nodeAction[childIdx] = event.validActions[i];
                    m_nodePrior[childIdx] = event.policy[i];
                }
                m_nodeFirstChild[leafIdx] = startIdx;
                m_nodeNumChildren[leafIdx] = static_cast<uint16_t>(nChildren);
                m_nodeFlags[leafIdx].store(FLAG_EXPANDED, std::memory_order_release);
            }
            else
            {
                m_nodeFlags[leafIdx].store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
            }
        }
        else if (event.isTerminal || event.validActions.empty())
        {
            m_nodeFlags[leafIdx].store(FLAG_TERMINAL | FLAG_EXPANDED, std::memory_order_release);
        }

        // Backprop Values
        for (int i = static_cast<int>(event.path.size()) - 1; i >= 0; --i)
        {
            uint32_t nodeIdx = event.path[i];
            m_nodeN[nodeIdx].fetch_add(1, std::memory_order_relaxed);
            for (uint8_t p = 0; p < kNumPlayers; ++p)
            {
                m_nodeW[p][nodeIdx].fetch_add(event.values[p], std::memory_order_relaxed);
            }
        }
        m_finishedSimulations.fetch_add(1, std::memory_order_release);
    }

    // --- GAME LOGIC ---

    void startSearch(const ObsState& rootState)
    {
        m_nodeCount.store(0, std::memory_order_relaxed);
        m_finishedSimulations.store(0, std::memory_order_relaxed);
        m_rootHistory.clear();

        IdxStateAction rootItem;
        m_engine->obsToIdx(rootState, rootItem.idxState);
        rootItem.idxAction = Fact<GameTag>::MakePad(FactType::ACTION);
        m_rootHistory.push_back(rootItem);

        m_rootIdx = allocNode();
        if (m_rootIdx != UINT32_MAX)
        {
            m_nodeStates[m_rootIdx] = rootState;
        }
    }

    // --- IMPLÉMENTATION DU REUSE TREE ---
    void advanceRoot(const Action& actionPlayed, const ObsState& newState)
    {
        // 1. Mise à jour de l'historique (Nécessaire même si on reset)
        if (m_rootIdx != UINT32_MAX)
        {
            typename NodeEvent<GameTag>::IdxStateAction histItem;
            m_engine->obsToIdx(m_nodeStates[m_rootIdx], histItem.idxState);
            m_engine->actionToIdx(actionPlayed, histItem.idxAction);
            m_rootHistory.push_back(histItem);
        }

        // 2. Gestion du Reuse Tree
        if (m_config.reuseTree && m_rootIdx != UINT32_MAX)
        {
            uint8_t flags = m_nodeFlags[m_rootIdx].load(std::memory_order_acquire);
            uint32_t nextRoot = UINT32_MAX;

            if (flags & FLAG_EXPANDED)
            {
                uint32_t start = m_nodeFirstChild[m_rootIdx];
                uint32_t end = start + m_nodeNumChildren[m_rootIdx];
                for (uint32_t i = start; i < end; ++i)
                {
                    if (isActionEqual(m_nodeAction[i], actionPlayed))
                    {
                        nextRoot = i;
                        break;
                    }
                }
            }

            if (nextRoot != UINT32_MAX)
            {
                // SUCCÈS : On a trouvé l'enfant, on conserve l'arbre
                m_rootIdx = nextRoot;
                m_nodeStates[m_rootIdx] = newState; // Sécurité : on met à jour l'état exact
                m_finishedSimulations.store(0, std::memory_order_relaxed);

                // Vérif mémoire après promotion
                if (isMemoryFull()) {
                    auto savedHist = m_rootHistory;
                    startSearch(newState);
                    m_rootHistory = std::move(savedHist);
                }
                return; // Sortie anticipée : on a réutilisé l'arbre
            }
        }

        // 3. Fallback (Pas de reuse ou Enfant non trouvé ou Reset forcé)
        auto savedHist = m_rootHistory;
        startSearch(newState);
        m_rootHistory = std::move(savedHist);
    }

    void selectMove(float temperature, Action& out)
    {
        if (m_rootIdx == UINT32_MAX || m_nodeNumChildren[m_rootIdx] == 0)
        {
            out = Action{};
            return;
        }

        uint32_t start = m_nodeFirstChild[m_rootIdx];
        uint32_t end = start + m_nodeNumChildren[m_rootIdx];

        if (temperature < 1e-3f)
        {
            uint32_t bestIdx = UINT32_MAX;
            uint32_t maxN = 0;
            float maxP = -1.0f;
            for (uint32_t i = start; i < end; ++i) {
                uint32_t n = m_nodeN[i].load(std::memory_order_relaxed);
                if (n > maxN) { maxN = n; bestIdx = i; }
                else if (n == maxN) {
                    if (m_nodePrior[i] > maxP) { maxP = m_nodePrior[i]; if (maxN == 0) bestIdx = i; }
                }
            }
            if (bestIdx != UINT32_MAX) out = m_nodeAction[bestIdx];
            else out = m_nodeAction[start];
            return;
        }

        double sum = 0.0;
        AlignedVec<double> weights;
        weights.reserve(m_nodeNumChildren[m_rootIdx]);

        bool isTempOne = (std::abs(temperature - 1.0f) < 1e-3f);
        double invTemp = 1.0 / static_cast<double>(temperature);

        for (uint32_t i = start; i < end; ++i) {
            uint32_t n = m_nodeN[i].load(std::memory_order_relaxed);
            double w = 0.0;
            if (n > 0) {
                if (isTempOne) w = static_cast<double>(n);
                else w = std::pow(static_cast<double>(n), invTemp);
            }
            weights.push_back(w);
            sum += w;
        }

        if (sum < 1e-9) { selectMove(0.0f, out); return; }

        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, sum);
        double sample = dist(gen);
        double runningSum = 0.0;
        size_t childOffset = 0;

        for (uint32_t i = start; i < end; ++i, ++childOffset) {
            runningSum += weights[childOffset];
            if (runningSum >= sample) { out = m_nodeAction[i]; return; }
        }
        out = m_nodeAction[end - 1];
    }

    bool isMemoryFull() const
    {
        return m_nodeCount.load(std::memory_order_relaxed) >=
            static_cast<uint32_t>(m_config.maxNodes * m_config.memoryThreshold);
    }
};