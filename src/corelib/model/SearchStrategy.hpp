#pragma once

#include <atomic>
#include <cmath>
#include <algorithm>
#include <array>
#include <vector>
#include <random>
#include <limits>

namespace Core
{
    // ============================================================================
    // UNIVERSAL SEARCH STRATEGY (N-Player MCTS-PUCT + Gumbel policy improvement)
    //
    // Two complementary mechanisms from Gumbel MuZero (Danihelka et al., 2022)
    // are integrated here without breaking the existing async parallel pipeline:
    //
    //  1. Gumbel-Top-K root sampling
    //     Replaces Dirichlet noise for root exploration.
    //     Samples m candidate actions using Gumbel(0,1) noise added to
    //     log(prior), then restricts the root search to those m candidates.
    //
    //  2. Completed Q-value policy improvement
    //     After all simulations, the training policy target is computed not
    //     from raw visit counts but from an improved distribution:
    //       improved_pi(a) ∝ softmax( log(prior(a)) + sigma * Q_completed(a) )
    // ============================================================================
    struct StrategyPUCT
    {
        // ====================================================================
        // EDGE DATA
        // Stores per-action statistics updated lock-free by concurrent threads.
        // ====================================================================
        struct EdgeData
        {
            std::atomic<uint32_t> visitCount{ 0 };
            std::atomic<float>    totalValue{ 0.0f };

            EdgeData() = default;

            EdgeData(const EdgeData& other) {
                visitCount.store(other.visitCount.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
                totalValue.store(other.totalValue.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
            }

            EdgeData& operator=(const EdgeData& other) {
                visitCount.store(other.visitCount.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
                totalValue.store(other.totalValue.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
                return *this;
            }
        };

        // ====================================================================
        // BASIC ACCESSORS
        // ====================================================================

        static inline float getPolicyMetric(const EdgeData& edge)
        {
            return static_cast<float>(edge.visitCount.load(std::memory_order_relaxed));
        }

        // Q-value = mean value for the player who moved to this edge.
        // Returns 0 if the edge has never been visited (FPU handles that case).
        static inline float getQ(const EdgeData& edge)
        {
            const float n = getPolicyMetric(edge);
            if (n < 1.0f) return 0.0f;
            return edge.totalValue.load(std::memory_order_relaxed) / n;
        }

        // ====================================================================
        // PUCT SCORE (Avec correction de robustesse Gumbel vs FPU)
        //
        // Formula: score(a) = Q(a) + cPUCT * P(a) * sqrt(N_parent) / (1 + N(a))
        // ====================================================================
        static inline float computeScore(const EdgeData& edge,
            uint32_t        parentVisits,
            float           prior,
            float           cPUCT,
            float           fpuValue)
        {
            // --- FIX CRITIQUE GUMBEL vs FPU ---
            // Si Gumbel a ramené ce "prior" à 0.0, l'action a été élaguée.
            // On renvoie un score infiniment bas pour la tuer dans l'œuf,
            // évitant que la FPU (+0.25 par exemple) ne la relance si 
            // le Top-K en cours s'avère mauvais.
            if (prior <= 1e-9f) {
                return -1e9f;
            }

            const uint32_t childVisits = edge.visitCount.load(std::memory_order_relaxed);

            // FPU: les branches inexplorées reçoivent la valeur FPU plutôt que 0.
            const float    q = (childVisits == 0) ? fpuValue : getQ(edge);

            const float    u = cPUCT * prior
                * (std::sqrt(static_cast<float>(parentVisits))
                    / (1.0f + static_cast<float>(childVisits)));

            return q + u;
        }

        // ====================================================================
        // BACKPROPAGATION
        // ====================================================================

        static inline void update(EdgeData& edge, float value)
        {
            edge.visitCount.fetch_add(1, std::memory_order_release);
            edge.totalValue.fetch_add(value, std::memory_order_release);
        }

        // ====================================================================
        // VIRTUAL LOSS
        // ====================================================================

        static inline void applyVirtualLoss(EdgeData& edge, float penalty)
        {
            edge.visitCount.fetch_add(1, std::memory_order_relaxed);
            edge.totalValue.fetch_sub(penalty, std::memory_order_relaxed);
        }

        static inline void removeVirtualLoss(EdgeData& edge, float penalty)
        {
            edge.visitCount.fetch_sub(1, std::memory_order_relaxed);
            edge.totalValue.fetch_add(penalty, std::memory_order_relaxed);
        }

        // ====================================================================
        // GUMBEL-TOP-K ROOT SAMPLING
        // ====================================================================
        static void applyGumbelTopK(uint32_t     startIdx,
            uint32_t     nChildren,
            float* priors,       // m_nodePrior[startIdx..]
            uint32_t     k)
        {
            if (k == 0 || k >= nChildren) return;  // disabled or trivial

            thread_local std::mt19937 rng{ std::random_device{}() };

            // Draw Gumbel(0,1) samples: g = -log(-log(U)),  U ~ Uniform(0,1)
            std::uniform_real_distribution<float> udist(1e-8f, 1.0f);

            std::vector<std::pair<float, uint32_t>> scores(nChildren);
            for (uint32_t i = 0; i < nChildren; ++i) {
                const float u = udist(rng);
                const float gumbel = -std::log(-std::log(u));
                const float logPrior = (priors[startIdx + i] > 1e-9f)
                    ? std::log(priors[startIdx + i])
                    : -1e9f;
                scores[i] = { gumbel + logPrior, i };
            }

            // Partial sort: find top-k in O(n log k)
            std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                [](const auto& a, const auto& b) {
                    return a.first > b.first;  // descending
                });

            // Zero-out the priors of non-selected actions
            std::vector<bool> selected(nChildren, false);
            for (uint32_t i = 0; i < k; ++i)
                selected[scores[i].second] = true;

            float selectedSum = 0.0f;
            for (uint32_t i = 0; i < nChildren; ++i) {
                if (!selected[i])
                    priors[startIdx + i] = 0.0f;
                else
                    selectedSum += priors[startIdx + i];
            }

            // Renormalize selected priors to sum to 1
            if (selectedSum > 1e-9f) {
                const float invSum = 1.0f / selectedSum;
                for (uint32_t i = 0; i < nChildren; ++i)
                    if (selected[i])
                        priors[startIdx + i] *= invSum;
            }
        }

        // ====================================================================
        // COMPLETED Q-VALUE POLICY IMPROVEMENT
        // ====================================================================
        template<typename ActionIdFn>
        static void computeImprovedPolicy(
            uint32_t        startIdx,
            uint32_t        nChildren,
            const float* priors,
            const EdgeData* edges,
            float           cVisit,
            float           cScale,
            ActionIdFn      getActionId,
            uint32_t        actionSpace,
            float* outPolicy)
        {
            if (nChildren == 0) return;

            // ----------------------------------------------------------------
            // Step 1: Compute Q_completed for each child & find maxN
            // ----------------------------------------------------------------
            float sumQ = 0.0f;
            int   nVisit = 0;
            float maxN = 0.0f;

            for (uint32_t i = 0; i < nChildren; ++i) {
                const float n = getPolicyMetric(edges[startIdx + i]);
                if (n > maxN) maxN = n;

                if (n >= 1.0f) {
                    sumQ += getQ(edges[startIdx + i]);
                    ++nVisit;
                }
            }

            // Conservative estimate for unvisited: mean - 1 (penalised)
            const float fallbackQ = (nVisit > 0)
                ? (sumQ / static_cast<float>(nVisit)) - 1.0f
                : -1.0f;

            // ----------------------------------------------------------------
            // Step 1.5: Compute Dynamic Sigma
            // ----------------------------------------------------------------
            const float safeScale = (cScale > 1e-5f) ? cScale : 1.0f;
            const float sigma = (cVisit + maxN) / safeScale;

            // ----------------------------------------------------------------
            // Step 2: Compute logit = log(prior) + sigma * Q_completed
            // ----------------------------------------------------------------
            std::vector<float> logits(nChildren);
            float              maxLogit = -std::numeric_limits<float>::max();

            for (uint32_t i = 0; i < nChildren; ++i) {
                const float prior = priors[startIdx + i];
                // Sécurité cohérente avec Gumbel : les priors de 0 donnent -1e9
                const float logP = (prior > 1e-9f) ? std::log(prior) : -1e9f;
                const float n = getPolicyMetric(edges[startIdx + i]);
                const float qComp = (n >= 1.0f) ? getQ(edges[startIdx + i]) : fallbackQ;

                logits[i] = logP + sigma * qComp;
                if (logits[i] > maxLogit) maxLogit = logits[i];
            }

            // ----------------------------------------------------------------
            // Step 3: Stable softmax over logits
            // ----------------------------------------------------------------
            float expSum = 0.0f;
            for (uint32_t i = 0; i < nChildren; ++i) {
                logits[i] = std::exp(logits[i] - maxLogit);
                expSum += logits[i];
            }

            // ----------------------------------------------------------------
            // Step 4: Write improved policy into the output array
            // ----------------------------------------------------------------
            if (expSum > 1e-9f) {
                const float invSum = 1.0f / expSum;
                for (uint32_t i = 0; i < nChildren; ++i) {
                    const uint32_t aId = getActionId(i);
                    if (aId < actionSpace)
                        outPolicy[aId] += logits[i] * invSum;
                }
            }
            else {
                // Fallback: uniform over children
                const float uni = 1.0f / static_cast<float>(nChildren);
                for (uint32_t i = 0; i < nChildren; ++i) {
                    const uint32_t aId = getActionId(i);
                    if (aId < actionSpace)
                        outPolicy[aId] += uni;
                }
            }
        }
    };
}