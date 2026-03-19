#pragma once

#include <atomic>
#include <cmath>
#include <algorithm>

namespace Core
{
    // ============================================================================
    // UNIVERSAL SEARCH STRATEGY (N-Player MCTS-PUCT)
    // Core mathematical evaluator for the AlphaZero Monte Carlo Tree Search.
    // Supports lock-free asynchronous updates via C++20 atomic floats.
    // ============================================================================
    struct StrategyPUCT
    {
        // ------------------------------------------------------------------------
        // EDGE DATA
        // Represents the statistics of a specific action (edge) in the tree.
        // ------------------------------------------------------------------------
        struct EdgeData
        {
            std::atomic<uint32_t> visitCount{ 0 };

            // Stores the cumulative expected reward (Q-Value) FOR the player who 
            // takes this action. Requires C++20 for atomic float operations.
            std::atomic<float>    totalValue{ 0.0f };

            EdgeData() = default;

            // Explicit copy semantics required for std::vector resizing.
            // Relaxed memory order is sufficient during initialization/resizing 
            // as the tree is not actively searched by other threads at this stage.
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

        // ------------------------------------------------------------------------
        // POLICY EXTRACTION
        // ------------------------------------------------------------------------
        static inline float getPolicyMetric(const EdgeData& edge)
        {
            // The training policy target (Pi) is directly proportional to the visit counts
            return static_cast<float>(edge.visitCount.load(std::memory_order_relaxed));
        }

        // À ajouter dans SearchStrategy.hpp côté StrategyPUCT
        // (nécessaire pour getRootValue)
        static float getQ(const EdgeData& e) {
            float n = getPolicyMetric(e);
            if (n < 1.0f) return 0.0f;
            return e.totalValue.load(std::memory_order_relaxed) / n;
        }

        // ------------------------------------------------------------------------
        // PUCT ALGORITHM (Predictor Upper Confidence Bound)
        // Formula: Q(s, a) + cPUCT * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        // ------------------------------------------------------------------------
        static inline float computeScore(const EdgeData& edge, uint32_t parentVisits, float prior, float cPUCT, float fpuValue) {
            uint32_t childVisits = getPolicyMetric(edge);

            // Si l'enfant n'a jamais été visité, on lui force le score du FPU
            float q = (childVisits == 0) ? fpuValue : getQ(edge);

            float u = cPUCT * prior * (std::sqrt(parentVisits) / (1.0f + childVisits));
            return q + u;
        }

        // ------------------------------------------------------------------------
        // BACKPROPAGATION UPDATE
        // ------------------------------------------------------------------------
        static inline void update(EdgeData& edge, float value)
        {
            // Release semantics to ensure value visibility across all threads
            edge.visitCount.fetch_add(1, std::memory_order_release);
            edge.totalValue.fetch_add(value, std::memory_order_release);
        }

        // ------------------------------------------------------------------------
        // VIRTUAL LOSS MECHANICS (Multi-Threading Collision Prevention)
        // ------------------------------------------------------------------------

        // Applied during the 'Gather' phase.
        // Temporarily inflates the visit count and heavily penalizes the value.
        // This artificially lowers the PUCT score, forcing concurrent threads to 
        // explore different branches of the tree while waiting for the GPU evaluation.
        static inline void applyVirtualLoss(EdgeData& edge, float virtualLossPenalty)
        {
            edge.visitCount.fetch_add(1, std::memory_order_relaxed);
            edge.totalValue.fetch_sub(virtualLossPenalty, std::memory_order_relaxed);
        }

        // Applied during the 'Backprop' phase.
        // Safely removes the temporary penalty before applying the true GPU value.
        static inline void removeVirtualLoss(EdgeData& edge, float virtualLossPenalty)
        {
            edge.visitCount.fetch_sub(1, std::memory_order_relaxed);
            edge.totalValue.fetch_add(virtualLossPenalty, std::memory_order_relaxed);
        }
    };
}