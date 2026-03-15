#pragma once
#include <algorithm>
#include <type_traits>
#include <cassert>
#include <array>
#include "../model/GameTypes.hpp"

namespace Core
{
    // ============================================================================
    // POV UTILS � Point of View Transformation Toolkit
    //
    // A high-performance, stateless utility class to apply Point-of-View (POV) 
    // transformations to Facts and Actions. Operations are strictly in-place and 
    // isolated per entity. It relies on the "NoHash" backdoor of the State class
    // to bypass Zobrist calculations, maximizing throughput for NN inference.
    // ============================================================================

    template<ValidGameTraits GT>
    class PovUtils
    {
    private:
        using Defs = Core::GameDefs<GT>;
        using State = Core::State<GT>;
        using Fact = Core::Fact<GT>;
        using Action = Core::Action<GT>;

        // ========================================================================
        // PRIVATE HELPERS
        // ========================================================================

        // Internal enumeration to define the type of spatial transformation.
        enum class SpatialOp { None, Mirror1D, Shift1D, CustomPermutation };

        template<typename T>
        static void transformEntity(
            T& entity,
            uint32_t viewer,
            SpatialOp op,
            uint32_t shiftOffset = 0,
            const std::array<uint32_t, Defs::kNumPos>* lut = nullptr) noexcept
        {
            if (viewer == 0) return; // Base POV, nothing to do

            // 1. OWNER ROTATION (Relative to the viewer)
            if (entity.ownerId() < Defs::kNumPlayers)
            {
                uint32_t ownerShift = (Defs::kNumPlayers - (viewer % Defs::kNumPlayers)) % Defs::kNumPlayers;
                uint32_t newOwner = (entity.ownerId() + ownerShift) % Defs::kNumPlayers;

                if constexpr (std::is_same_v<T, Fact>) {
                    entity.setOwner(newOwner);
                }
                else if constexpr (std::is_same_v<T, Action>) {
                    // L'Action n'a pas de setOwner(), on la reconfigure directement avec ses index primitifs
                    entity.configure(entity.factId(), newOwner, entity.source(), entity.dest(), entity.value());
                }
            }

            // 2. SPATIAL TRANSFORMATION
            if (op != SpatialOp::None && entity.exists())
            {
                if constexpr (std::is_same_v<T, Fact>)
                {
                    auto oldLoc = entity.rawLocation();
                    entity.kill(); // Clear current bits
                    entity.setValue(1.0f); // Re-awaken entity

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
                    // Utilisation directe de kNoPos selon ton API
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
        // ACTIONS (Direct object mutation)
        // ========================================================================

        // Rotates owner ID only. Space remains untouched.
        static void doRotateOwnerOnlyAction(Action& act, uint32_t viewer) noexcept {
            transformEntity(act, viewer, SpatialOp::None);
        }

        // Rotates owner ID and applies a 1D spatial mirror (Max - 1 - pos).
        static void doRotateOwnerAndMirrorAction(Action& act, uint32_t viewer) noexcept {
            transformEntity(act, viewer, SpatialOp::Mirror1D);
        }

        // Rotates owner ID and circularly shifts spatial positions by 'offset'.
        static void doRotateOwnerAndShiftSpaceAction(Action& act, uint32_t viewer, uint32_t offset) noexcept {
            transformEntity(act, viewer, SpatialOp::Shift1D, offset);
        }

        // Rotates owner ID and applies an arbitrary spatial permutation based on a Look-Up Table.
        static void doRotateOwnerAndPermuteSpaceAction(Action& act, uint32_t viewer, const std::array<uint32_t, Defs::kNumPos>& lut) noexcept {
            transformEntity(act, viewer, SpatialOp::CustomPermutation, 0, &lut);
        }

        // ========================================================================
        // ELEMENTS (Index-based access, bypasses Zobrist Hash via NoHash mutator)
        // ========================================================================

        // Rotates owner ID only. Space remains untouched. Ideal for shared markets or static geography.
        static void doRotateOwnerOnlyElem(State& state, uint32_t elemIdx, uint32_t viewer) noexcept {
            assert(elemIdx < Defs::kMaxElems);
            Fact& f = state.modifyFactNoHash(elemIdx);
            transformEntity(f, viewer, SpatialOp::None);
        }

        // Rotates owner ID and applies a 1D spatial mirror. Ideal for standard 2-player board games.
        static void doRotateOwnerAndMirrorElem(State& state, uint32_t elemIdx, uint32_t viewer) noexcept {
            assert(elemIdx < Defs::kMaxElems);
            Fact& f = state.modifyFactNoHash(elemIdx);
            transformEntity(f, viewer, SpatialOp::Mirror1D);
        }

        // Rotates owner ID and circularly shifts spatial positions. Ideal for Mancala or circular tracks.
        static void doRotateOwnerAndShiftSpaceElem(State& state, uint32_t elemIdx, uint32_t viewer, uint32_t offset) noexcept {
            assert(elemIdx < Defs::kMaxElems);
            Fact& f = state.modifyFactNoHash(elemIdx);
            transformEntity(f, viewer, SpatialOp::Shift1D, offset);
        }

        // Rotates owner ID and applies an arbitrary spatial permutation. Ideal for hexagonal/star grids.
        static void doRotateOwnerAndPermuteSpaceElem(State& state, uint32_t elemIdx, uint32_t viewer, const std::array<uint32_t, Defs::kNumPos>& lut) noexcept {
            assert(elemIdx < Defs::kMaxElems);
            Fact& f = state.modifyFactNoHash(elemIdx);
            transformEntity(f, viewer, SpatialOp::CustomPermutation, 0, &lut);
        }

        // ========================================================================
        // METADATA (Relative index access, bypasses Zobrist Hash via NoHash mutator)
        // ========================================================================

        // Rotates owner ID only. Ideal for abstract metadata like 'Current Turn' or 'Score'.
        static void doRotateOwnerOnlyMeta(State& state, uint32_t metaIdx, uint32_t viewer) noexcept {
            assert(metaIdx < Defs::kMaxMetas);
            Fact& f = state.modifyFactNoHash(Defs::kMaxElems + metaIdx);
            transformEntity(f, viewer, SpatialOp::None);
        }

        // Rotates owner ID and applies a 1D spatial mirror. Ideal for spatial metadata like 'En Passant'.
        static void doRotateOwnerAndMirrorMeta(State& state, uint32_t metaIdx, uint32_t viewer) noexcept {
            assert(metaIdx < Defs::kMaxMetas);
            Fact& f = state.modifyFactNoHash(Defs::kMaxElems + metaIdx);
            transformEntity(f, viewer, SpatialOp::Mirror1D);
        }

        // Rotates owner ID and circularly shifts spatial positions for metadata.
        static void doRotateOwnerAndShiftSpaceMeta(State& state, uint32_t metaIdx, uint32_t viewer, uint32_t offset) noexcept {
            assert(metaIdx < Defs::kMaxMetas);
            Fact& f = state.modifyFactNoHash(Defs::kMaxElems + metaIdx);
            transformEntity(f, viewer, SpatialOp::Shift1D, offset);
        }

        // Rotates owner ID and applies an arbitrary spatial permutation for metadata.
        static void doRotateOwnerAndPermuteSpaceMeta(State& state, uint32_t metaIdx, uint32_t viewer, const std::array<uint32_t, Defs::kNumPos>& lut) noexcept {
            assert(metaIdx < Defs::kMaxMetas);
            Fact& f = state.modifyFactNoHash(Defs::kMaxElems + metaIdx);
            transformEntity(f, viewer, SpatialOp::CustomPermutation, 0, &lut);
        }
    };
}