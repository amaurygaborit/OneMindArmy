#pragma once
#include <atomic>
#include <cmath>

namespace Core
{
    // ============================================================================
    // UNIVERSAL STRATEGY (N-Player MCTS-PUCT)
    // Works for Perfect Information (Exact States) AND 
    // Imperfect Information (Belief States encoded via BitsetT).
    // ============================================================================
    struct StrategyPUCT
    {
        struct EdgeData
        {
            std::atomic<uint32_t> visitCount{ 0 };
            std::atomic<float>    totalValue{ 0.0f }; // Stocke la valeur POUR le joueur ayant pris cette décision

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

        static float computeScore(const EdgeData& edge, uint32_t parentVisits, float prior, float cPUCT)
        {
            uint32_t N = edge.visitCount.load(std::memory_order_relaxed);
            float Q = (N > 0) ? (edge.totalValue.load(std::memory_order_relaxed) / static_cast<float>(N)) : 0.0f;
            float U = cPUCT * prior * std::sqrt(static_cast<float>(parentVisits)) / (1.0f + static_cast<float>(N));
            return Q + U; // Q est directemment la valeur du joueur courant, on cherche à la maximiser
        }

        static void update(EdgeData& edge, float value)
        {
            edge.visitCount.fetch_add(1, std::memory_order_release);
            edge.totalValue.fetch_add(value, std::memory_order_release);
        }

        static void applyVirtualLoss(EdgeData& edge, float loss)
        {
            edge.visitCount.fetch_add(1, std::memory_order_relaxed);
            // La virtual loss pénalise le score du joueur courant
            edge.totalValue.fetch_sub(loss, std::memory_order_relaxed);
        }
        static void removeVirtualLoss(EdgeData& edge, float loss)
        {
            edge.visitCount.fetch_sub(1, std::memory_order_relaxed);
            edge.totalValue.fetch_add(loss, std::memory_order_relaxed);
        }

        static float getPolicyMetric(const EdgeData& edge)
        {
            return static_cast<float>(edge.visitCount.load(std::memory_order_relaxed));
        }
    };
}