#pragma once

#include <atomic>
#include <cmath>
#include <algorithm>
#include <array>
#include <random>
#include <limits>

namespace Core
{
    // ============================================================================
    // UNIVERSAL SEARCH STRATEGY
    // N-Player MCTS implementation combining PUCT evaluation and Gumbel policy 
    // improvement. 
    //
    // Design Intent:
    // Completely lock-free. Nodes use atomic operations for visit counts and 
    // Q-values to support massive multi-threading traversal. Templates are used
    // to allocate zero-overhead static arrays based on compile-time game limits.
    // ============================================================================
    template<ValidGameTraits GT>
    struct StrategyPUCT
    {
        USING_GAME_TYPES(GT);

        struct EdgeData
        {
            std::atomic<uint32_t> visitCount{ 0 };
            std::atomic<float>    totalValue{ 0.0f };

            EdgeData() = default;

            // Explicit copy constructors needed because std::atomic is non-copyable.
            // Uses relaxed ordering since edge duplication only occurs during safe tree expansion.
            EdgeData(const EdgeData& other) {
                visitCount.store(other.visitCount.load(std::memory_order_relaxed), std::memory_order_relaxed);
                totalValue.store(other.totalValue.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }

            EdgeData& operator=(const EdgeData& other) {
                visitCount.store(other.visitCount.load(std::memory_order_relaxed), std::memory_order_relaxed);
                totalValue.store(other.totalValue.load(std::memory_order_relaxed), std::memory_order_relaxed);
                return *this;
            }
        };

        static inline float getPolicyMetric(const EdgeData& edge) {
            return static_cast<float>(edge.visitCount.load(std::memory_order_relaxed));
        }

        static inline float getQ(const EdgeData& edge) {
            const float n = getPolicyMetric(edge);
            if (n < 1.0f) return 0.0f;
            return edge.totalValue.load(std::memory_order_relaxed) / n;
        }

        // PUCT Formula (Predictor + UCB applied to Trees).
        // Balances exploiting known high-Q branches vs exploring high-prior, low-visit branches.
        static inline float computeScore(const EdgeData& edge, uint32_t parentVisits, float prior, float cPUCT, float fpuValue)
        {
            if (prior <= 1e-9f) {
                return -1e9f; // Filter out actions pruned by Gumbel-Top-K
            }

            const uint32_t childVisits = edge.visitCount.load(std::memory_order_relaxed);
            // First Play Urgency overrides Q for unvisited nodes to control exploration depth.
            const float    q = (childVisits == 0) ? fpuValue : getQ(edge);
            const float    u = cPUCT * prior * (std::sqrt(static_cast<float>(parentVisits)) / (1.0f + static_cast<float>(childVisits)));

            return q + u;
        }

        // Atomic Backpropagation.
        static inline void update(EdgeData& edge, float value) {
            edge.visitCount.fetch_add(1, std::memory_order_release);
            edge.totalValue.fetch_add(value, std::memory_order_release);
        }

        // Virtual Loss injection prevents parallel threads from collapsing 
        // down the exact same search path during concurrent MCTS traversal.
        static inline void applyVirtualLoss(EdgeData& edge, float penalty) {
            edge.visitCount.fetch_add(1, std::memory_order_relaxed);
            edge.totalValue.fetch_sub(penalty, std::memory_order_relaxed);
        }

        static inline void removeVirtualLoss(EdgeData& edge, float penalty) {
            edge.visitCount.fetch_sub(1, std::memory_order_relaxed);
            edge.totalValue.fetch_add(penalty, std::memory_order_relaxed);
        }

        // ====================================================================
        // GUMBEL-TOP-K ROOT SAMPLING 
        // Samples initial actions using Gumbel noise to accelerate policy convergence.
        // Modifies priors in-place to kill branches outside the Top-K.
        // ====================================================================
        static void applyGumbelTopK(uint32_t startIdx, uint32_t nChildren, float* priors, uint32_t k,
            uint32_t* outActiveIndices, uint32_t& outActiveCount)
        {
            if (nChildren == 0 || nChildren > Defs::kMaxValidActions) return;

            uint32_t actualK = std::min(k, nChildren);
            if (actualK == 0) actualK = nChildren; // Safely bypass Gumbel if K is 0

            thread_local std::mt19937 rng{ std::random_device{}() };
            std::uniform_real_distribution<float> udist(1e-8f, 1.0f);

            // Avoids heap allocation using compile-time bounded arrays.
            std::array<std::pair<float, uint32_t>, Defs::kMaxValidActions> scores;

            for (uint32_t i = 0; i < nChildren; ++i) {
                const float u = udist(rng);
                const float gumbel = -std::log(-std::log(u));
                const float prior = priors[startIdx + i];
                const float logPrior = (prior > 1e-9f) ? std::log(prior) : -1e9f;
                scores[i] = { gumbel + logPrior, i };
            }

            // In-place sort to locate the Top-K indices.
            std::partial_sort(scores.begin(), scores.begin() + actualK, scores.begin() + nChildren,
                [](const auto& a, const auto& b) { return a.first > b.first; });

            std::array<bool, Defs::kMaxValidActions> selected{};
            for (uint32_t i = 0; i < actualK; ++i) {
                selected[scores[i].second] = true;
                if (outActiveIndices) outActiveIndices[i] = scores[i].second;
            }
            outActiveCount = actualK;

            float selectedSum = 0.0f;
            for (uint32_t i = 0; i < nChildren; ++i) {
                if (!selected[i]) {
                    priors[startIdx + i] = 0.0f; // Silence non-Top-K branches
                }
                else {
                    selectedSum += priors[startIdx + i];
                }
            }

            // Renormalize the remaining priors to maintain a valid probability distribution.
            if (selectedSum > 1e-9f) {
                const float invSum = 1.0f / selectedSum;
                for (uint32_t i = 0; i < nChildren; ++i) {
                    if (selected[i]) priors[startIdx + i] *= invSum;
                }
            }
        }

        // ====================================================================
        // COMPLETED Q-VALUE POLICY IMPROVEMENT
        // Converts raw MCTS visit counts into a refined target policy, incorporating 
        // the empirical Q-values found during search to sharpen the network's learning signal.
        // ====================================================================
        template<typename ActionIdFn>
        static void computeImprovedPolicy(
            uint32_t startIdx, uint32_t nChildren, const float* priors, const EdgeData* edges,
            float cVisit, float cScale, ActionIdFn getActionId, uint32_t actionSpace, float* outPolicy)
        {
            if (nChildren == 0 || nChildren > Defs::kMaxValidActions) return;

            float sumQ = 0.0f, maxN = 0.0f;
            int nVisit = 0;

            for (uint32_t i = 0; i < nChildren; ++i) {
                const float n = getPolicyMetric(edges[startIdx + i]);
                if (n > maxN) maxN = n;
                if (n >= 1.0f) { sumQ += getQ(edges[startIdx + i]); ++nVisit; }
            }

            // Unvisited nodes use a fallback Q slightly lower than the average to penalize them.
            const float fallbackQ = (nVisit > 0) ? (sumQ / static_cast<float>(nVisit)) - 1.0f : -1.0f;
            const float safeScale = (cScale > 1e-5f) ? cScale : 1.0f;
            const float sigma = (cVisit + maxN) / safeScale;

            std::array<float, Defs::kMaxValidActions> logits;
            float maxLogit = -std::numeric_limits<float>::max();

            for (uint32_t i = 0; i < nChildren; ++i) {
                const float prior = priors[startIdx + i];
                const float logP = (prior > 1e-9f) ? std::log(prior) : -1e9f;
                const float n = getPolicyMetric(edges[startIdx + i]);
                const float qComp = (n >= 1.0f) ? getQ(edges[startIdx + i]) : fallbackQ;

                logits[i] = logP + sigma * qComp;
                if (logits[i] > maxLogit) maxLogit = logits[i];
            }

            // Softmax transformation
            float expSum = 0.0f;
            for (uint32_t i = 0; i < nChildren; ++i) {
                logits[i] = std::exp(logits[i] - maxLogit);
                expSum += logits[i];
            }

            if (expSum > 1e-9f) {
                const float invSum = 1.0f / expSum;
                for (uint32_t i = 0; i < nChildren; ++i) {
                    const uint32_t aId = getActionId(i);
                    if (aId < actionSpace) outPolicy[aId] += logits[i] * invSum;
                }
            }
            else {
                // Fallback to a uniform distribution if numeric instability occurs
                const float uni = 1.0f / static_cast<float>(nChildren);
                for (uint32_t i = 0; i < nChildren; ++i) {
                    const uint32_t aId = getActionId(i);
                    if (aId < actionSpace) outPolicy[aId] += uni;
                }
            }
        }
    };
}