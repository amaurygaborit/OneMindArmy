#pragma once
#include "../interfaces/IEngine.hpp"
#include "NeuralNet.cuh"

#include <iostream>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <algorithm>

template<typename GameTag>
class MCTS
{
private:
    struct Node
    {
		uint32_t parentIdx;     // index of the parent
		uint32_t childOffset;   // offset in the child array
        uint16_t childCount;    // number of legal actions
        uint8_t flags;          // expanded, in_eval, pinned, etc.

        inline bool isExpanded() const noexcept { return flags & 0x1; }
        inline bool isPinned()   const noexcept { return flags & 0x2; }

        inline void setExpanded(bool v) noexcept { v ? flags |= 0x1 : flags &= ~0x1; }
        inline void setPinned(bool v) noexcept { v ? flags |= 0x2 : flags &= ~0x2; }

        inline void reset() noexcept
        {
            parentIdx = UINT32_MAX;
            childOffset = UINT32_MAX;
            childCount = 0;
			flags = 0;
        }
    };

	AlignedVec<uint32_t> m_childNodeIdx;    // child node index
    AlignedVec<ActionT<GameTag>>  m_childAction;     // action per child slot
    AlignedVec<float>    m_childPrior;      // prior per child slot
    AlignedVec<uint32_t> m_childN;          // visits per child slot
    AlignedVec<float>    m_childW;          // sum of values per child slot

    // Parameters
    const uint8_t  m_numPlayers;
    const uint16_t m_maxValidActions;
    const uint16_t m_maxChildren;

    const uint32_t m_maxNodes;
    const uint8_t  m_numBeliefSamples;
    const float    m_cPUCT;
	const uint16_t m_keepK;
    const uint16_t m_maxDepth;

    std::shared_ptr<IEngine<GameTag>>   m_engine;
	std::shared_ptr<NeuralNet<GameTag>> m_neuralNet;

    AlignedVec<Node>               m_nodes;
    AlignedVec<ObsStateT<GameTag>> m_states;

    uint32_t m_rootIdx = UINT32_MAX;                        // current root index

    AlignedVec<float>            m_valuesBuf;               // values for every players at the leaf
    AlignedVec<float>            m_policyBuf;               // policy at the leaf
    AlignedVec<float>            m_beliefBuf;               // buffer for belief probabilities
	AlignedVec<std::pair
        <ObsStateT<GameTag>, float>> m_beliefSamplesBuf;    // { states, weights }
    AlignedVec<ActionT<GameTag>> m_validActionsBuf;         // valid actions at the leaf
    AlignedVec<IdxActionT>       m_validIdxActionsBuf;      // for direct mapping to policy
	IdxStateT<GameTag>           m_idxStateBuf;             // idx state for neural net input

    AlignedVec<uint32_t> m_freeStack;                       // free node indices
    AlignedVec<uint32_t> m_stack;                           // for subtree freeing

    struct Candidate { uint16_t slot; uint32_t child; uint32_t visits; };
    AlignedVec<Candidate> m_candGCPrune;

	AlignedVec<std::pair
        <uint32_t, uint32_t>> m_selectPath;                 // nodeIdx, childSlot for backpropagation

private:
    uint32_t allocNode()
    {
        if (m_freeStack.empty()) return UINT32_MAX;
        uint32_t idx = m_freeStack.pop_back_value();
        Node& newNode = m_nodes[idx];
        newNode.reset();
        return idx;
    }

	// move last child to slot (to keep the array compact)
    void removeChildCompact(uint32_t pIdx, uint16_t slot)
    {
        Node& pNode = m_nodes[pIdx];

        if (pNode.childCount <= 1)
            throw std::runtime_error("MCTS::removeChildCompact: parent has zero/one children");

        const uint16_t lastSlot = pNode.childCount - 1;
        const uint32_t from = pNode.childOffset + lastSlot;
        const uint32_t to = pNode.childOffset + slot;

        const uint32_t delChildNodeIdx = m_childNodeIdx[to];
        const uint32_t lastChildNodeIdx = m_childNodeIdx[from];

        // If the slot to remove is not the last one, move the last slot into slot
        if (slot != lastSlot)
        {
			// move child info
            m_childNodeIdx[to] = lastChildNodeIdx;
            m_childAction[to] = std::move(m_childAction[from]);
            m_childPrior[to] = m_childPrior[from];
            m_childN[to] = m_childN[from];

            for (uint8_t p = 0; p < m_numPlayers; ++p)
                m_childW[to * m_numPlayers + p] = m_childW[from * m_numPlayers + p];
        }

        // Free the removed child node index
        m_freeStack.push_back(delChildNodeIdx);
        pNode.childCount--;
    }

    // free subtree from pIdx
    void freeSubtreeChildren(uint32_t pIdx)
    {
        Node& pNode = m_nodes[pIdx];
        if (pNode.childCount == 0) return;

        m_stack.clear();

		// stack children of pIdx
        const uint32_t base = pNode.childOffset;
        for (uint16_t slot = 0; slot < pNode.childCount; ++slot)
        {
            uint32_t childIdx = m_childNodeIdx[base + slot];
            m_stack.push_back(childIdx);
        }

		// reset parent node
        pNode.childCount = 0;
        pNode.setExpanded(false);

		// free all children in the stack
        while (!m_stack.empty())
        {
            uint32_t idx = m_stack.pop_back_value();
            Node& node = m_nodes[idx];

            const uint32_t baseChild = node.childOffset;
            for (uint16_t slot = 0; slot < node.childCount; ++slot)
            {
                uint32_t childIdx = m_childNodeIdx[baseChild + slot];
                m_stack.push_back(childIdx);
            }
            node.reset();
            m_freeStack.push_back(idx);
        }
    }

    void gcPruneFromRoot(uint32_t keepK)
    {
        Node& rootNode = m_nodes[m_rootIdx];
        if (rootNode.childCount <= keepK) return;

        const uint32_t base = rootNode.childOffset;

        m_candGCPrune.clear();
        // build candidates with correct visit counts (use slot-indexed m_childN)
        for (uint16_t slot = 0; slot < rootNode.childCount; ++slot)
        {
            uint32_t childNodeIdx = m_childNodeIdx[base + slot];
            if (m_nodes[childNodeIdx].isPinned()) continue; // never prune pinned nodes
            uint32_t visits = m_childN[base + slot]; // <-- use slot-based index
            m_candGCPrune.push_back({ slot, childNodeIdx, visits });
        }

        if (m_candGCPrune.size() <= keepK) return;

        // Partition so first keepK elements are the best (by visits)
        std::nth_element(
            m_candGCPrune.begin(),
            m_candGCPrune.begin() + keepK,
            m_candGCPrune.end(),
            [](const Candidate& a, const Candidate& b) { return a.visits > b.visits; }
        );

        // Build a list of slots to REMOVE (those not in first keepK)
        std::vector<uint16_t> slotsToRemove;
        slotsToRemove.reserve(m_candGCPrune.size() - keepK);
        for (size_t i = keepK; i < m_candGCPrune.size(); ++i)
            slotsToRemove.push_back(m_candGCPrune[i].slot);

        // Important: remove in descending order of slot, so removeChildCompact's swap
        // doesn't change the meaning of slots we still need to remove.
        std::sort(slotsToRemove.begin(), slotsToRemove.end(), std::greater<uint16_t>());

        for (uint16_t slot : slotsToRemove)
        {
            uint32_t slotIdx = base + slot;
            uint32_t delChildIdx = m_childNodeIdx[slotIdx];

            // free subtree for that child node
            if (delChildIdx != UINT32_MAX)
                freeSubtreeChildren(delChildIdx);

            // remove the child slot (this will compact the array)
            removeChildCompact(m_rootIdx, slot);
        }
    }

    void ensureCapacityOrPrune()
    {
        if (m_freeStack.empty() && m_rootIdx != UINT32_MAX)
            gcPruneFromRoot(m_keepK);
    }

    // selectPUCT (returns UINT32_MAX if no selectable child)
    uint32_t selectPUCT(uint32_t pIdx)
    {
        Node& pNode = m_nodes[pIdx];
        const uint32_t base = pNode.childOffset;

        // compute sum of child visits
        uint32_t sumN = 0;
        for (uint16_t i = 0; i < pNode.childCount; ++i)
            sumN += m_childN[base + i];
        const float sqrtSumN = std::sqrt(static_cast<float>(std::max<uint32_t>(1u, sumN)));

        float bestScore = -std::numeric_limits<float>::infinity();
        uint16_t bestSlot = UINT16_MAX;

        const uint8_t curPlayer = m_engine->getCurrentPlayer(m_states[pIdx]);

        // iterate children
        for (uint16_t slot = 0; slot < pNode.childCount; ++slot)
        {
            const uint32_t slotIdx = base + slot;

            uint32_t childNodeIdx = m_childNodeIdx[slotIdx];
            if (childNodeIdx != UINT32_MAX)
            {
                if (m_nodes[childNodeIdx].isPinned())
                    continue;
            }

            const float N = static_cast<float>(m_childN[slotIdx]);
            const float Qp = (N > 0.f) ? (m_childW[static_cast<size_t>(slotIdx) * m_numPlayers + curPlayer] / N) : 0.f;
            const float U = m_cPUCT * m_childPrior[slotIdx] * (sqrtSumN / (1.f + N));
            const float S = Qp + U;

            if (S > bestScore)
            {
                bestScore = S;
                bestSlot = slot;
            }
        }

        if (bestSlot == UINT16_MAX)
            return UINT32_MAX; // no selectable child found

        const uint32_t bestSlotIdx = base + bestSlot;
        m_selectPath.push_back({ pIdx, bestSlot });

        return m_childNodeIdx[bestSlotIdx];
    }

    void topKFromBelief(const ObsStateT<GameTag>& state)
    {
		m_beliefSamplesBuf.push_back({ state, 1.0f });
    }

    // expands node pIdx by generating (action x K) child samples, batching NN forwards
    void expand(uint32_t pIdx)
    {
        Node& pNode = m_nodes[pIdx];
        const ObsStateT<GameTag>& pState = m_states[pIdx];

        m_valuesBuf.reset(); // reuse for parent value fetch

        if (pNode.isExpanded()) return;
        if (m_engine->isTerminal(pState, m_valuesBuf))
        {
            pNode.setExpanded(true);
            return;
        }

        // get valid actions from parent observed state
        m_validActionsBuf.clear();
        m_engine->getValidActions(pState, m_validActionsBuf);
        const uint16_t m = static_cast<uint16_t>(m_validActionsBuf.size());
        if (m == 0)
        {
            pNode.setExpanded(true);
            return;
        }

        // Sanity: ensure node arrays can hold at most m * K children for this node
        const uint16_t K = m_numBeliefSamples;
        const size_t maxNeededChildren = static_cast<size_t>(m) * static_cast<size_t>(K);
        if (maxNeededChildren > m_maxChildren)
            throw std::runtime_error("MCTS::expand(): m_maxChildren too small for m * K");

        // quick capacity check for node allocation
        if (m_freeStack.size() < maxNeededChildren)
            return; // not enough free node indices -> abort expansion (do not mark expanded)

        // parent forward (policy + value + belief)
        m_engine->obsToIdx(pState, m_idxStateBuf);
        m_neuralNet->forward(m_idxStateBuf, m_policyBuf.data(), m_valuesBuf.data(), m_beliefBuf.data());

        // normalize policy restricted to valid actions
        m_validIdxActionsBuf.reset();
        for (uint16_t slot = 0; slot < m; ++slot)
            m_engine->actionToIdx(m_validActionsBuf[slot], m_validIdxActionsBuf[slot]);
        m_neuralNet->normalizeToProba(m_validIdxActionsBuf, m_policyBuf.data());

        // compute base offset in child arrays for this node
        const uint32_t base = static_cast<uint32_t>(pIdx) * static_cast<uint32_t>(m_maxChildren);
        pNode.childOffset = base; // fixed slot region for this node
        uint16_t totalSamples = 0; // number of child slots actually used

        // For each action, sample belief states and create children sequentially in child array
        for (uint16_t slot = 0; slot < m; ++slot)
        {
            // compute action prior (scalar) from parent's policy
            const float actionPrior = m_policyBuf[m_validIdxActionsBuf[slot].factIdx];

            // after applying action to public part only, we sample beliefs for next player
            ObsStateT<GameTag> afterActionState = pState;
            m_engine->applyAction(m_validActionsBuf[slot], afterActionState);

            // get up to K samples (ObsState, rawWeight) for this action
            m_beliefSamplesBuf.clear();
            topKFromBelief(afterActionState);
            size_t produced = m_beliefSamplesBuf.size();
            if (produced == 0)
                throw std::runtime_error("MCTS::expand(): topKFromBelief returned no samples");

            // normalize the returned raw weights for this action
            double sumW = 0.0;
            for (size_t i = 0; i < produced; ++i) sumW += static_cast<double>(m_beliefSamplesBuf[i].second);

            if (sumW <= 0.0)
            {
                // fallback: uniform weights across produced samples
                for (size_t i = 0; i < produced; ++i) m_beliefSamplesBuf[i].second = 1.0f / static_cast<float>(produced);
            }
            else
            {
                for (size_t i = 0; i < produced; ++i)
                    m_beliefSamplesBuf[i].second = static_cast<float>(m_beliefSamplesBuf[i].second / static_cast<float>(sumW));
            }

            // create child entries for each produced sample
            for (size_t s = 0; s < produced; ++s)
            {
                // global slot in child arrays for this new child
                const uint32_t childSlot = base + static_cast<uint32_t>(totalSamples);
                if (childSlot >= (m_nodes.size() * m_maxChildren))
                    throw std::out_of_range("MCTS::expand(): childSlot out of range");

                // allocate a child node index from node pool
                uint32_t newChildIdx = allocNode();

                // write the sampled observed-state into the node's state
                m_states[newChildIdx] = m_beliefSamplesBuf[s].first;
                m_nodes[newChildIdx].parentIdx = pIdx;

                // fill AoS arrays at childSlot
                m_childNodeIdx[childSlot] = newChildIdx;
                m_childAction[childSlot] = m_validActionsBuf[slot];
                m_childN[childSlot] = 0u;
                m_childPrior[childSlot] = actionPrior * m_beliefSamplesBuf[s].second;

                // initialize W (per-player) to zero: children don't have evaluated values yet
                const size_t wDst = static_cast<size_t>(childSlot) * static_cast<size_t>(m_numPlayers);
                for (uint8_t pl = 0; pl < m_numPlayers; ++pl)
                    m_childW[wDst + pl] = 0.0f;

                ++totalSamples;
                // safety: do not exceed node's max child capacity
                if (totalSamples > m_maxChildren)
                    throw std::runtime_error("MCTS::expand(): totalSamples exceeded m_maxChildren for node");
            }
        }
        // finalize parent node
        pNode.childCount = totalSamples;
        pNode.setExpanded(true);
    }

	// backprop values from leafNodeIdx up to root
    void backprop(uint32_t leafNodeIdx)
    {
        if (leafNodeIdx != UINT32_MAX)
        {
            m_nodes[leafNodeIdx].setPinned(false);
        }

        // Walk the selection path in reverse
        for (int i = static_cast<int>(m_selectPath.size()) - 1; i >= 0; --i)
        {
            const auto& entry = m_selectPath[i];
            const uint32_t pIdx = entry.first;
            const uint16_t slot = entry.second;

            if (pIdx == UINT32_MAX) continue;
            Node& pNode = m_nodes[pIdx];
            const uint32_t slotIdx = pNode.childOffset + static_cast<uint32_t>(slot);

            if (slotIdx >= m_childNodeIdx.size())
                throw std::out_of_range("MCTS::backprop: slotIdx out of range");

            ++m_childN[slotIdx];
            const uint32_t base = static_cast<uint32_t>(slotIdx) * static_cast<uint32_t>(m_numPlayers);
            for (uint8_t p = 0; p < m_numPlayers; ++p)
            {
                m_childW[base + p] += m_valuesBuf[p];
            }

            const uint32_t childNodeIdx = m_childNodeIdx[slotIdx];
            if (childNodeIdx != UINT32_MAX)
            {
                m_nodes[childNodeIdx].setPinned(false);
            }
            else
            {
                pNode.setPinned(false);
            }
        }
        m_selectPath.clear();
    }

    // iterateOnce (selection handles UINT32_MAX)
    void iterateOnce()
    {
		uint32_t curIdx = m_rootIdx;
        m_nodes[curIdx].setPinned(true);
        while (m_nodes[curIdx].isExpanded())
        {
            uint32_t childIdx = selectPUCT(curIdx);
            if (childIdx == UINT32_MAX) break; // treat childNodeIdx as leaf
            m_nodes[childIdx].setPinned(true);
            curIdx = childIdx;
        }
        expand(curIdx);
        backprop(curIdx);
    }

public:
    MCTS(std::shared_ptr<IEngine<GameTag>> engine,
        std::shared_ptr<NeuralNet<GameTag>> neuralNet,
        uint8_t numPlayers, uint16_t maxValidActions, uint16_t actionSpaceSize, uint32_t maxNodes, uint8_t numBeliefSamples, float cPUCT, uint16_t keepK, uint16_t maxDepth)
        : m_engine(std::move(engine)), m_neuralNet(std::move(neuralNet)),
        m_numPlayers(numPlayers),
        m_maxValidActions(maxValidActions),
        m_maxChildren(maxValidActions * numBeliefSamples),
        m_maxNodes(maxNodes),
        m_numBeliefSamples(numBeliefSamples),
        m_cPUCT(cPUCT),
        m_keepK(keepK),
		m_maxDepth(maxDepth),
        m_nodes(maxNodes),
        m_states(maxNodes),
        m_childNodeIdx(maxNodes * maxValidActions * numBeliefSamples, UINT32_MAX),
        m_childAction(maxNodes * maxValidActions * numBeliefSamples),
		m_childPrior(maxNodes * maxValidActions * numBeliefSamples),
		m_childN(maxNodes * maxValidActions * numBeliefSamples),
		m_childW(maxNodes * maxValidActions * numBeliefSamples * numPlayers),
        m_policyBuf(actionSpaceSize),
        m_valuesBuf(numPlayers),
        m_beliefBuf(GameTraits<GameTag>::kNumElems * GameTraits<GameTag>::kNumPrivatePos),
		m_beliefSamplesBuf(reserve_only, numBeliefSamples),
        m_validActionsBuf(reserve_only, maxValidActions),
        m_validIdxActionsBuf(maxValidActions),
        m_freeStack(reserve_only, maxNodes),
        m_stack(reserve_only, maxNodes),
		m_selectPath(maxDepth + 1)
    {
        for (uint32_t i = 0; i < m_maxNodes; ++i) m_freeStack.push_back(i);
    }
    MCTS(const MCTS&) = delete;
    MCTS& operator=(const MCTS&) = delete;

    MCTS(MCTS&&) noexcept = default;
    MCTS& operator=(MCTS&&) noexcept = default;

    // Démarrer une recherche sur un état racine
    void startSearch(const ObsStateT<GameTag>& rootState)
    {
        m_rootIdx = allocNode();
        if (m_rootIdx == UINT32_MAX)
        {
            throw std::runtime_error("No alloc at startSearch()");
        }
        m_states[m_rootIdx] = rootState;
    }

    // Lance K simulations avec GC si besoin
    void run(uint32_t numSimulations)
    {
        for (uint32_t i = 0; i < numSimulations; ++i)
        {
            iterateOnce();
        }
    }

    // bestActionFromRoot (if all visits 0, fall back on highest prior)
    ActionT<GameTag> bestActionFromRoot() const
    {
        if (m_rootIdx == UINT32_MAX)
            throw std::runtime_error("MCTS::bestActionFromRoot(): root not set");

        const Node& root = m_nodes[m_rootIdx];
        if (root.childCount == 0)
            throw std::runtime_error("MCTS::bestActionFromRoot(): No child to choose");

        const uint32_t base = root.childOffset;
        if (base == UINT32_MAX)
            throw std::runtime_error("MCTS::bestActionFromRoot(): invalid childOffset");

        float bestP = -std::numeric_limits<float>::infinity();
        uint32_t bestSlotIdx = UINT32_MAX;

        for (uint16_t i = 0; i < root.childCount; ++i)
        {
            const uint32_t slotIdx = base + static_cast<uint32_t>(i);
            if (slotIdx >= m_childPrior.size())
                throw std::out_of_range("MCTS::bestActionFromRoot(): child slot out of range");

            const float p = m_childPrior[slotIdx];
            if (p > bestP)
            {
                bestP = p;
                bestSlotIdx = slotIdx;
            }
        }

        if (bestSlotIdx == UINT32_MAX)
            throw std::runtime_error("MCTS::bestActionFromRoot(): no selectable child found");

        return m_childAction[bestSlotIdx];
    }

	// Reroot by applying the played action to the current root
    void rerootByPlayedAction(const ActionT<GameTag>& played)
    {
        if (m_rootIdx == UINT32_MAX) return;

        const uint32_t oldRoot = m_rootIdx;
        Node& rootNode = m_nodes[oldRoot];
        ObsStateT<GameTag>& rootState = m_states[oldRoot];

        const uint32_t base = rootNode.childOffset;

        // find the slot corresponding to the played action
        uint16_t playedSlot = UINT16_MAX;
        for (uint16_t slot = 0; slot < rootNode.childCount; ++slot)
        {
            const uint32_t childIdx = base + static_cast<uint32_t>(slot);
            if (m_childAction[childIdx] == played)
            {
                playedSlot = slot;
                break;
            }
        }

        if (playedSlot != UINT16_MAX)
        {
            const uint32_t playedSlotIdx = base + static_cast<uint32_t>(playedSlot);
            const uint32_t candidateNewRoot = m_childNodeIdx[playedSlotIdx];

            // if the child node was already allocated, we can reroot into it
            if (candidateNewRoot != UINT32_MAX)
            {
                m_nodes[candidateNewRoot].parentIdx = UINT32_MAX;
                m_nodes[candidateNewRoot].setPinned(false);

                // remove all other children of old root
                for (int slot = static_cast<int>(rootNode.childCount) - 1; slot >= 0; --slot)
                {
                    if (static_cast<uint16_t>(slot) == playedSlot) continue;
                    const uint32_t slotIdx = base + static_cast<uint32_t>(slot);
                    const uint32_t childNode = m_childNodeIdx[slotIdx];
                    if (childNode != UINT32_MAX)
                    {
                        freeSubtreeChildren(childNode);
                    }
                    removeChildCompact(oldRoot, static_cast<uint16_t>(slot));
                }

                // free the old root node itself and set the new root
                m_freeStack.push_back(oldRoot);
                m_rootIdx = candidateNewRoot;
                return;
            }
        }

        // free the whole previous subtree and restart search from the updated rootState.
        m_engine->applyAction(played, rootState);
        freeSubtreeChildren(oldRoot);
        m_freeStack.push_back(oldRoot);
        startSearch(rootState);
    }
};