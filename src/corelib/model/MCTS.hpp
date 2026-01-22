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
	using GT = ITraits<GameTag>;
    using ObsState = typename GT::ObsState;
    using FactStateAction = typename FactStateActionT<GameTag>;
    using FactAction = typename Fact<GameTag>;
    using Action = typename GT::Action;

    // --- IDENTITÉ & CONTEXTE ---
    uint32_t leafNodeIdx;
    AlignedVec<uint32_t> path;
    AlignedVec<uint8_t> pathPlayers;

    // --- ÉTAT ---
    ObsState leafState;
    bool isTerminal;
    bool collision;

    // --- DONNÉES POUR L'INFÉRENCE ---
    // On stocke les Facts d'actions collectés pendant la descente
    AlignedVec<FactAction> pathActionFacts;

    // L'historique complet prêt pour le réseau de neurones
    AlignedVec<FactStateAction> nnHistory;

    // --- RÉSULTATS ---
    AlignedVec<Action> validActions;
    AlignedVec<float> policy;
    AlignedVec<float> values;

    NodeEvent(uint16_t historySize, uint16_t maxDepth)
    {
        path.reserve(maxDepth);
        pathPlayers.reserve(maxDepth);
        pathActionFacts.reserve(maxDepth);

        nnHistory.reserve(historySize);

        validActions.reserve(GT::kMaxValidActions);
        policy.reserve(GT::kActionSpace);
        values.reserve(GT::kNumPlayers);
        reset();
    }

    void reset()
    {
        leafNodeIdx = 0;
        path.clear();
        pathPlayers.clear();
        pathActionFacts.clear();

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
    using ObsState = typename GT::ObsState;
    using Action = typename GT::Action;

    using FactState = typename FactStateT<GameTag>;
    using FactAction = typename Fact<GameTag>;
    using FactStateAction = typename FactStateActionT<GameTag>;

    using ModelResults = typename ModelResultsT<GameTag>;
    using Event = NodeEvent<GameTag>;

private:
    static constexpr uint8_t kNumPlayers = GT::kNumPlayers;
    static constexpr uint8_t FLAG_EXPANDED = 0x01;
    static constexpr uint8_t FLAG_EXPANDING = 0x02;
    static constexpr uint8_t FLAG_TERMINAL = 0x04;

    std::shared_ptr<IEngine<GameTag>> m_engine;

    const MCTSConfig m_config;

    // Arbre (Structure of Arrays)
    std::atomic<uint32_t>* m_nodeN;
    std::atomic<float>* m_nodeW[kNumPlayers];
    std::atomic<uint8_t>* m_nodeFlags;

    AlignedVec<float>    m_nodePrior;
    AlignedVec<uint32_t> m_nodeFirstChild;
    AlignedVec<uint16_t> m_nodeNumChildren;
    AlignedVec<Action>   m_nodeAction;
    AlignedVec<ObsState> m_nodeStates; // Stocke uniquement la racine (généralement)

    // État courant
    std::atomic<uint32_t>      m_nodeCount{ 0 };
    std::atomic<uint32_t>      m_finishedSimulations{ 0 };
    uint32_t                   m_rootIdx = UINT32_MAX;

    // Historique à la racine (Invariant pour tout l'arbre)
    AlignedVec<FactStateAction> m_rootHistory;

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
            float loss = (p == player) ? -m_config.virtualLoss : m_config.virtualLoss;
            m_nodeW[p][nodeIdx].fetch_add(loss, std::memory_order_relaxed);
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

    // Assemble l'historique final (Root History + Path Actions)
    void prepareHistory(const AlignedVec<FactAction>& pathActions, AlignedVec<FactStateAction>& outHist)
    {
        outHist.clear();
        size_t rootHistSize = m_rootHistory.size();
        size_t pathSize = pathActions.size();
        size_t totalItems = rootHistSize + pathSize;
        size_t needed = m_config.historySize;

        // 1. Padding si nécessaire
        if (totalItems < needed)
        {
            size_t padCount = needed - totalItems;
            FactStateAction padItem;
            // Par défaut FactStateAction est vide/pad
            for (size_t i = 0; i < padCount; ++i) outHist.push_back(padItem);
        }

        // 2. Calcul du point de départ (fenêtre glissante)
        size_t startOffset = (totalItems > needed) ? (totalItems - needed) : 0;
        size_t currentIdx = 0;

        // 3. Ajout Root History
        for (size_t i = 0; i < rootHistSize; ++i)
        {
            if (outHist.size() == needed) break;
            if (currentIdx >= startOffset)
            {
                outHist.push_back(m_rootHistory[i]);
            }
            currentIdx++;
        }

        // 4. Ajout Path History (Actions récentes)
        for (size_t i = 0; i < pathSize; ++i)
        {
            if (outHist.size() == needed) break;
            if (currentIdx >= startOffset)
            {
                FactStateAction pathItem;
                // On met juste l'action, l'état reste PAD (Frame Stacking Action-Only pour les nœuds internes)
                // Ou alors ton architecture prévoit de remplir 'stateFacts' avec des placeholders.
                pathItem.actionFact = pathActions[i];
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

        // Copie locale de l'état pour la traversée
        ObsState currState = m_nodeStates[m_rootIdx];

        size_t player = m_engine->getCurrentPlayer(currState);
        event.pathPlayers.push_back(static_cast<uint8_t>(player));

        applyVirtualLoss(currIdx, player);

        int depth = 0;
        while (true)
        {
            uint8_t flags = m_nodeFlags[currIdx].load(std::memory_order_acquire);

            // Cas Terminal
            if (flags & FLAG_TERMINAL)
            {
                event.leafNodeIdx = currIdx;
                event.leafState = currState;
                event.isTerminal = true;
                m_engine->isTerminal(currState, event.values);
                return true;
            }

            // Cas Feuille non étendue -> On demande l'évaluation
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

                        // C'est le moment de construire l'historique pour le NN
                        prepareHistory(event.pathActionFacts, event.nnHistory);
                        return true;
                    }
                }
                event.leafNodeIdx = currIdx;
                event.collision = true;
                return true;
            }

            // Sélection
            uint32_t bestChild = selectBestChild(currIdx, currState);
            if (bestChild == UINT32_MAX)
            {
                // Pas d'enfant valide -> Terminal
                m_nodeFlags[currIdx].fetch_or(FLAG_TERMINAL, std::memory_order_release);
                event.leafNodeIdx = currIdx;
                event.isTerminal = true;
                m_engine->isTerminal(currState, event.values);
                return true;
            }

            // --- CRITICAL STEP: CONVERSION ACTION TO FACT ---
            // On convertit l'action MAINTENANT car on possède 'currState' (le contexte).
            // Si on attend la fin de la boucle, on aura perdu les états intermédiaires.
            FactAction actFact;
            const Action& nextAction = m_nodeAction[bestChild];
            m_engine->actionToFact(nextAction, currState, actFact);
            event.pathActionFacts.push_back(actFact);

            // Avancer
            applyVirtualLoss(bestChild, player);
            m_engine->applyAction(nextAction, currState); // currState mute ici
            currIdx = bestChild;

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

        // Revert Virtual Loss
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

        // Construction de l'élément initial de l'historique
        FactStateAction rootItem;
        m_engine->stateToFacts(rootState, rootItem.stateFacts);
        rootItem.actionFact.clear(); // Pas d'action précédente, PAD
        m_rootHistory.push_back(rootItem);

        m_rootIdx = allocNode();
        if (m_rootIdx != UINT32_MAX)
        {
            m_nodeStates[m_rootIdx] = rootState;
        }
    }

    void advanceRoot(const Action& actionPlayed, const ObsState& newState)
    {
        // 1. Mise à jour de l'historique
        if (m_rootIdx != UINT32_MAX)
        {
            FactStateAction histItem;
            // On utilise l'état précédent (stocké à la racine) pour contextualiser l'action
            ObsState prevState = m_nodeStates[m_rootIdx];

            // Pour l'historique, on sauvegarde l'état complet
            m_engine->stateToFacts(prevState, histItem.stateFacts);
            // Et l'action qui a mené à newState
            m_engine->actionToFact(actionPlayed, prevState, histItem.actionFact);

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
                m_rootIdx = nextRoot;
                m_nodeStates[m_rootIdx] = newState;
                m_finishedSimulations.store(0, std::memory_order_relaxed);

                if (isMemoryFull()) {
                    auto savedHist = m_rootHistory;
                    startSearch(newState);
                    m_rootHistory = std::move(savedHist);
                }
                return;
            }
        }

        auto savedHist = m_rootHistory;
        startSearch(newState);
        m_rootHistory = std::move(savedHist);
    }

    void selectMove(float temperature, Action& out)
    {
        if (m_rootIdx == UINT32_MAX || m_nodeNumChildren[m_rootIdx] == 0)
        {
            out = Action{}; // Default constructible action
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

    float getMemoryUsage() const
    {
        return static_cast<float>(m_nodeCount.load(std::memory_order_relaxed)) / static_cast<float>(m_config.maxNodes);
    }
};