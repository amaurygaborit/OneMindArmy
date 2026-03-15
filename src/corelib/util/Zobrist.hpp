#pragma once
#include <random>
#include <iostream>
#include <cassert>

namespace Core
{
    template<ValidGameTraits GT>
    class GenericZobrist
    {
    private:
        using Defs = Core::GameDefs<GT>;

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
                std::mt19937_64 rng(123456789ULL);

                for (uint32_t o = 0; o < kNumOwnersAlloc; ++o) {
                    for (uint32_t f = 0; f < kNumFactsAlloc; ++f) {
                        for (uint32_t p = 0; p < kNumPosAlloc; ++p) {
                            keys[o][f][p] = rng();
                        }
                    }
                }

                // Default: all FactTypes are hashed
                hashMask.setRange(0, kNumFactsAlloc - 1);
            }
        };

        // ====================================================================
        // MEYERS SINGLETON (100% Thread-Safe)
        // ====================================================================
        static ZobristTable& getTable() noexcept
        {
            static ZobristTable instance;
            return instance;
        }

    public:
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
            // CRITICAL: Dead entities contribute nothing to the state hash.
            if (!f.exists()) return 0;
            if (f.factId() == Defs::kPadFact) return 0;

            auto& table = getTable();

            if (!table.hashMask.test(f.factId())) return 0;

            uint32_t singlePos = f.pos();

            // Case 1: Perfect Info / Collapsed entity (Extremely fast O(1) path)
            if (singlePos != Defs::kNoPos)
            {
                return table.keys[f.ownerId()][f.factId()][singlePos];
            }

            const auto& loc = f.rawLocation();

            // Case 2: Imperfect Info (Superposition of multiple positions)
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

            // Case 3: Metadata or off-board entity (Exists, but no spatial presence)
            return table.keys[f.ownerId()][f.factId()][Defs::kNoPos];
        }
    };
}