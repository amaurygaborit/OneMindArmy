#pragma once
#include <random>
#include <iostream>
#include <cassert>

namespace Core
{
    // ============================================================================
    // UNIVERSAL ZOBRIST HASHING
    // 
    // Design Intent:
    // Automatically generates and manages random 64-bit keys for every possible 
    // combination of [Owner] x [FactType] x [Position]. This allows the Engine 
    // to maintain an O(1) state hash representing the board, which is essential 
    // for cycle detection (e.g., three-fold repetition in Chess) and MCTS tree 
    // transposition caching.
    // ============================================================================
    template<ValidGameTraits GT>
    class GenericZobrist
    {
    private:
        using Defs = Core::GameDefs<GT>;

        // Capacities padded by +1 to safely handle sentinel values 
        // (kNoOwner, kPadFact, kNoPos) without triggering out-of-bounds access.
        static constexpr uint32_t kNumOwnersAlloc = Defs::kNumPlayers + 1;
        static constexpr uint32_t kNumFactsAlloc = Defs::kNumFactTypes + 1;
        static constexpr uint32_t kNumPosAlloc = Defs::kNumPos + 1;

        struct ZobristTable
        {
            uint64_t keys[kNumOwnersAlloc][kNumFactsAlloc][kNumPosAlloc];
            BitsetT<kNumFactsAlloc> hashMask;

            ZobristTable()
            {
                std::cout << "[Zobrist] Initializing RNG keys for Game..." << std::endl;

                // Fixed seed guarantees identical hashes across distributed cluster nodes.
                std::mt19937_64 rng(123456789ULL);

                for (uint32_t o = 0; o < kNumOwnersAlloc; ++o) {
                    for (uint32_t f = 0; f < kNumFactsAlloc; ++f) {
                        for (uint32_t p = 0; p < kNumPosAlloc; ++p) {
                            keys[o][f][p] = rng();
                        }
                    }
                }

                hashMask.setRange(0, kNumFactsAlloc - 1);
            }
        };

        // Meyers Singleton implementation guarantees thread-safe initialization 
        // without requiring explicit mutex locking during runtime.
        static ZobristTable& getTable() noexcept
        {
            static ZobristTable instance;
            return instance;
        }

    public:
        // Allows the Engine to selectively ignore certain entities in the hash calculation 
        // (e.g., ignoring a turn counter to recognize a repeating physical board layout).
        static void ignoreElemType(uint32_t elemId) noexcept
        {
            assert(elemId < Defs::kNumElemTypes);
            getTable().hashMask.unset(elemId);
        }

        static void ignoreMetaType(uint32_t metaId) noexcept
        {
            assert(metaId < Defs::kNumMetaTypes);
            getTable().hashMask.unset(Defs::kNumElemTypes + metaId);
        }

        static uint64_t getKey(const Fact<GT>& f) noexcept
        {
            // Dead or padding entities inherently contribute nothing to the state hash.
            if (!f.exists()) return 0;
            if (f.factId() == Defs::kPadFact) return 0;

            auto& table = getTable();

            if (!table.hashMask.test(f.factId())) return 0;

            uint32_t singlePos = f.pos();

            // PATH 1: Perfect Information (Entity is collapsed to a single square).
            // Represents 99.9% of calls in deterministic games. Optimized for O(1).
            if (singlePos != Defs::kNoPos)
            {
                return table.keys[f.ownerId()][f.factId()][singlePos];
            }

            const auto& loc = f.rawLocation();

            // PATH 2: Imperfect Information (Entity exists in a probability superposition).
            // The hash incorporates all possible locations via XOR blending.
            if (loc.popcount() > 1)
            {
                uint64_t h = 0;
                for (size_t p = 0; p < Defs::kNumPos; ++p) {
                    if (loc.test(p)) {
                        h ^= table.keys[f.ownerId()][f.factId()][p];
                    }
                }
                return h;
            }

            // PATH 3: Metadata or Off-Board Entity.
            // Entity actively influences the state but possesses no spatial footprint.
            return table.keys[f.ownerId()][f.factId()][Defs::kNoPos];
        }
    };
}