#pragma once
#include <algorithm>
#include <type_traits>
#include <cassert>
#include <array>
#include "../model/GameTypes.hpp"

namespace Core
{
    // ============================================================================
    // POINT OF VIEW (POV) TRANSFORMATIONS
    // 
    // Design Intent:
    // A high-performance, stateless utility class used to rotate the board geometry 
    // and piece ownership relative to the current viewer. Operations are strictly 
    // in-place. 
    // 
    // Critical Optimizations:
    // Bypasses Zobrist hash synchronization entirely by using the `modifyFactNoHash` 
    // backdoor in the State object. This is essential because POV rotation occurs 
    // hundreds of thousands of times per second right before Neural Network batching,
    // and recalculating hashes here would cripple throughput.
    // ============================================================================

    template<ValidGameTraits GT>
    class PovUtils
    {
    private:
        using Defs = Core::GameDefs<GT>;
        using State = Core::State<GT>;
        using Fact = Core::Fact<GT>;
        using Action = Core::Action<GT>;

        enum class SpatialOp { None, Mirror1D, Shift1D, CustomPermutation };

        template<typename T>
        static void transformEntity(
            T& entity,
            uint32_t viewer,
            SpatialOp op,
            uint32_t shiftOffset = 0,
            const std::array<uint32_t, Defs::kNumPos>* lut = nullptr) noexcept
        {
            if (viewer == 0) return; // Base perspective requires no rotation

            // 1. OWNER ROTATION
            // Cycles the ownership of pieces so that the 'viewer' always perceives 
            // themselves as Player 0.
            if (entity.ownerId() < Defs::kNumPlayers)
            {
                uint32_t ownerShift = (Defs::kNumPlayers - (viewer % Defs::kNumPlayers)) % Defs::kNumPlayers;
                uint32_t newOwner = (entity.ownerId() + ownerShift) % Defs::kNumPlayers;

                if constexpr (std::is_same_v<T, Fact>) {
                    entity.setOwner(newOwner);
                }
                else if constexpr (std::is_same_v<T, Action>) {
                    // Actions lack high-level setters to minimize overhead; reconfigured directly.
                    entity.configure(entity.factId(), newOwner, entity.source(), entity.dest(), entity.value());
                }
            }

            // 2. SPATIAL TRANSFORMATION
            // Projects the physical location of the piece across the board geometry.
            if (op != SpatialOp::None && entity.exists())
            {
                if constexpr (std::is_same_v<T, Fact>)
                {
                    auto oldLoc = entity.rawLocation();
                    entity.kill(); // Wipe spatial mask
                    entity.setValue(1.0f); // Re-awaken semantic presence

                    for (uint32_t i = 0; i < Defs::kNumPos; ++i) {
                        if (oldLoc.test(i)) {
                            uint32_t newPos = i;
                            switch (op) {
                            case SpatialOp::Mirror1D:
                                newPos = (Defs::kNumPos - 1) - i;
                                break;
                            case SpatialOp::Shift1D:
                                newPos = (i + shiftOffset) % Defs::kNumPos;
                                break;
                            case SpatialOp::CustomPermutation:
                                assert(lut != nullptr);
                                newPos = (*lut)[i];
                                break;
                            default: break;
                            }
                            entity.addPossiblePos(newPos);
                        }
                    }
                }
                else if constexpr (std::is_same_v<T, Action>)
                {
                    auto applyOp = [&](uint32_t pos) -> uint32_t {
                        if (pos == Defs::kNoPos) return Defs::kNoPos;

                        switch (op) {
                        case SpatialOp::Mirror1D: return (Defs::kNumPos - 1) - pos;
                        case SpatialOp::Shift1D:  return (pos + shiftOffset) % Defs::kNumPos;
                        case SpatialOp::CustomPermutation: return (*lut)[pos];
                        default: return pos;
                        }
                        };

                    uint32_t newSrc = applyOp(entity.source());
                    uint32_t newDst = applyOp(entity.dest());

                    entity.setPos(newSrc, newDst, entity.value());
                }
            }
        }

    public:
        // ========================================================================
        // ACTIONS 
        // ========================================================================

        static void doRotateOwnerOnlyAction(Action& act, uint32_t viewer) noexcept {
            transformEntity(act, viewer, SpatialOp::None);
        }

        static void doRotateOwnerAndMirrorAction(Action& act, uint32_t viewer) noexcept {
            transformEntity(act, viewer, SpatialOp::Mirror1D);
        }

        static void doRotateOwnerAndShiftSpaceAction(Action& act, uint32_t viewer, uint32_t offset) noexcept {
            transformEntity(act, viewer, SpatialOp::Shift1D, offset);
        }

        static void doRotateOwnerAndPermuteSpaceAction(Action& act, uint32_t viewer, const std::array<uint32_t, Defs::kNumPos>& lut) noexcept {
            transformEntity(act, viewer, SpatialOp::CustomPermutation, 0, &lut);
        }

        // ========================================================================
        // ELEMENTS 
        // Handles physical board entities. Bypasses Zobrist hashing.
        // ========================================================================

        static void doRotateOwnerOnlyElem(State& state, uint32_t elemIdx, uint32_t viewer) noexcept {
            assert(elemIdx < Defs::kMaxElems);
            Fact& f = state.modifyFactNoHash(elemIdx);
            transformEntity(f, viewer, SpatialOp::None);
        }

        static void doRotateOwnerAndMirrorElem(State& state, uint32_t elemIdx, uint32_t viewer) noexcept {
            assert(elemIdx < Defs::kMaxElems);
            Fact& f = state.modifyFactNoHash(elemIdx);
            transformEntity(f, viewer, SpatialOp::Mirror1D);
        }

        static void doRotateOwnerAndShiftSpaceElem(State& state, uint32_t elemIdx, uint32_t viewer, uint32_t offset) noexcept {
            assert(elemIdx < Defs::kMaxElems);
            Fact& f = state.modifyFactNoHash(elemIdx);
            transformEntity(f, viewer, SpatialOp::Shift1D, offset);
        }

        static void doRotateOwnerAndPermuteSpaceElem(State& state, uint32_t elemIdx, uint32_t viewer, const std::array<uint32_t, Defs::kNumPos>& lut) noexcept {
            assert(elemIdx < Defs::kMaxElems);
            Fact& f = state.modifyFactNoHash(elemIdx);
            transformEntity(f, viewer, SpatialOp::CustomPermutation, 0, &lut);
        }

        // ========================================================================
        // METADATA
        // Handles abstract game rules or scoring slots. Bypasses Zobrist hashing.
        // ========================================================================

        static void doRotateOwnerOnlyMeta(State& state, uint32_t metaIdx, uint32_t viewer) noexcept {
            assert(metaIdx < Defs::kMaxMetas);
            Fact& f = state.modifyFactNoHash(Defs::kMaxElems + metaIdx);
            transformEntity(f, viewer, SpatialOp::None);
        }

        static void doRotateOwnerAndMirrorMeta(State& state, uint32_t metaIdx, uint32_t viewer) noexcept {
            assert(metaIdx < Defs::kMaxMetas);
            Fact& f = state.modifyFactNoHash(Defs::kMaxElems + metaIdx);
            transformEntity(f, viewer, SpatialOp::Mirror1D);
        }

        static void doRotateOwnerAndShiftSpaceMeta(State& state, uint32_t metaIdx, uint32_t viewer, uint32_t offset) noexcept {
            assert(metaIdx < Defs::kMaxMetas);
            Fact& f = state.modifyFactNoHash(Defs::kMaxElems + metaIdx);
            transformEntity(f, viewer, SpatialOp::Shift1D, offset);
        }

        static void doRotateOwnerAndPermuteSpaceMeta(State& state, uint32_t metaIdx, uint32_t viewer, const std::array<uint32_t, Defs::kNumPos>& lut) noexcept {
            assert(metaIdx < Defs::kMaxMetas);
            Fact& f = state.modifyFactNoHash(Defs::kMaxElems + metaIdx);
            transformEntity(f, viewer, SpatialOp::CustomPermutation, 0, &lut);
        }
    };
}