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
    // UNIVERSAL SEARCH STRATEGY (N-Player MCTS-PUCT + Gumbel policy improvement)
    // TEMPLATED ON GT to allow zero-allocation static arrays based on game bounds.
    // ============================================================================
    template<ValidGameTraits GT>
    struct StrategyPUCT
    {
        USING_GAME_TYPES(GT);

        // ====================================================================
        // EDGE DATA
        // ====================================================================
        struct EdgeData
        {
            std::atomic<uint32_t> visitCount{ 0 };
            std::atomic<float>    totalValue{ 0.0f };

            EdgeData() = default;

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

        static inline float computeScore(const EdgeData& edge, uint32_t parentVisits, float prior, float cPUCT, float fpuValue)
        {
            if (prior <= 1e-9f) {
                return -1e9f; // Kill action pruned by Gumbel
            }

            const uint32_t childVisits = edge.visitCount.load(std::memory_order_relaxed);
            const float    q = (childVisits == 0) ? fpuValue : getQ(edge);
            const float    u = cPUCT * prior * (std::sqrt(static_cast<float>(parentVisits)) / (1.0f + static_cast<float>(childVisits)));

            return q + u;
        }

        static inline void update(EdgeData& edge, float value) {
            edge.visitCount.fetch_add(1, std::memory_order_release);
            edge.totalValue.fetch_add(value, std::memory_order_release);
        }

        static inline void applyVirtualLoss(EdgeData& edge, float penalty) {
            edge.visitCount.fetch_add(1, std::memory_order_relaxed);
            edge.totalValue.fetch_sub(penalty, std::memory_order_relaxed);
        }

        static inline void removeVirtualLoss(EdgeData& edge, float penalty) {
            edge.visitCount.fetch_sub(1, std::memory_order_relaxed);
            edge.totalValue.fetch_add(penalty, std::memory_order_relaxed);
        }

        // ====================================================================
        // GUMBEL-TOP-K ROOT SAMPLING (Zero Allocation & Fixed Bypass)
        // ====================================================================
        static void applyGumbelTopK(uint32_t startIdx, uint32_t nChildren, float* priors, uint32_t k,
            uint32_t* outActiveIndices, uint32_t& outActiveCount)
        {
            if (nChildren == 0 || nChildren > Defs::kMaxValidActions) return;

            // FIX: Gère correctement le cas où k >= nChildren sans ignorer Gumbel
            uint32_t actualK = std::min(k, nChildren);
            if (actualK == 0) actualK = nChildren;

            thread_local std::mt19937 rng{ std::random_device{}() };
            std::uniform_real_distribution<float> udist(1e-8f, 1.0f);

            // OPTIMISATION : Allocation sur la pile
            std::array<std::pair<float, uint32_t>, Defs::kMaxValidActions> scores;

            for (uint32_t i = 0; i < nChildren; ++i) {
                const float u = udist(rng);
                const float gumbel = -std::log(-std::log(u));
                const float prior = priors[startIdx + i];
                const float logPrior = (prior > 1e-9f) ? std::log(prior) : -1e9f;
                scores[i] = { gumbel + logPrior, i };
            }

            // Tri in-place du Top-K
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
                    priors[startIdx + i] = 0.0f; // Tue les branches non Top-K
                }
                else {
                    selectedSum += priors[startIdx + i];
                }
            }

            // Renormalisation du prior du Top-K
            if (selectedSum > 1e-9f) {
                const float invSum = 1.0f / selectedSum;
                for (uint32_t i = 0; i < nChildren; ++i) {
                    if (selected[i]) priors[startIdx + i] *= invSum;
                }
            }
        }

        // ====================================================================
        // COMPLETED Q-VALUE POLICY IMPROVEMENT (Zero Allocation)
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
                const float uni = 1.0f / static_cast<float>(nChildren);
                for (uint32_t i = 0; i < nChildren; ++i) {
                    const uint32_t aId = getActionId(i);
                    if (aId < actionSpace) outPolicy[aId] += uni;
                }
            }
        }
    };
}