#pragma once
#include <array>
#include <span>
#include <algorithm>
#include <cstdint>
#include <cmath>

#include "../interfaces/IEngine.hpp"

namespace Core
{
    // ============================================================================
    // NEURAL NETWORK ENCODER (Game-Agnostic Serializer)
    // Translates symbolic POD structures (Facts/Actions) into flat float tensors.
    // Implements Strict Positional Anchoring for Transformer stability, supporting
    // both Perfect Information (Dirac) and Imperfect Information (Probability Cloud).
    // ============================================================================
    template<ValidGameTraits GT>
    struct StateEncoder
    {
        USING_GAME_TYPES(GT);

        // ------------------------------------------------------------------------
        // ENCODE FACT: Multi-Hot Spatial Representation
        // Maps a board element or metadata into a geometric token.
        // Dead facts (!exists) are still encoded to preserve their Transformer
        // positional slot and identity (Ghost Tokens), but with zeroed presence.
        // Layout: [Type, FactID, OwnerID, Value, Pos0, Pos1, ..., PosN]
        // ------------------------------------------------------------------------
        static inline void encodeFact(const Fact& fact, float* outToken) noexcept
        {
            // 1. Identity Base (Preserved even if the entity is dead)
            outToken[0] = static_cast<float>(fact.type());
            outToken[1] = static_cast<float>(fact.factId());
            outToken[2] = static_cast<float>(fact.ownerId());

            // 2. SymLog Compression: sign(x) * ln(|x| + 1)
            // If fact is dead, value is 0.0f, SymLog maps safely to 0.0f.
            float val = static_cast<float>(fact.value());
            outToken[3] = std::copysign(std::log1p(std::abs(val)), val);

            // 3. Spatial Mask (Imperfect Info compatible)
            // isPossiblePos() automatically returns false if the fact is dead.
            for (uint32_t i = 0; i < Defs::kNumPos; ++i) {
                outToken[4 + i] = fact.isPossiblePos(i) ? 1.0f : 0.0f;
            }
        }

        // ------------------------------------------------------------------------
        // ENCODE ACTION: Dipole Movement Representation
        // Maps an action/move into a directional spatial vector.
        // Source location is heavily penalized (-1.0), Destination is boosted (+1.0).
        // ------------------------------------------------------------------------
        static inline void encodeAction(const Action& action, float* outToken) noexcept
        {
            outToken[0] = static_cast<float>(action.type());
            outToken[1] = static_cast<float>(action.factId());
            outToken[2] = static_cast<float>(action.ownerId());

            float val = static_cast<float>(action.value());
            outToken[3] = std::copysign(std::log1p(std::abs(val)), val);

            // Clear the spatial map
            for (uint32_t i = 0; i < Defs::kNumPos; ++i) {
                outToken[4 + i] = 0.0f;
            }

            // Apply the movement Dipole
            if (action.isValid())
            {
                if (action.source() < Defs::kNumPos) {
                    outToken[4 + action.source()] = -1.0f;
                }
                if (action.dest() < Defs::kNumPos) {
                    outToken[4 + action.dest()] = 1.0f;
                }
            }
        }

        // ------------------------------------------------------------------------
        // FULL STATE ENCODING (Strict Absolute Anchoring)
        // Memory Layout Guarantee (1:1 mapping with State::m_facts):
        // [ 0 ... kMaxElems-1 ]        -> Elements (Physical Pieces)
        // [ kMaxElems ... kMaxFacts-1] -> Metas (Rules, Turns, Scores)
        // [ kMaxFacts ... end ]        -> History (Action(t), Action(t-1), ...)
        // ------------------------------------------------------------------------
        static inline void encode(const State& state, std::span<const Action> history,
            std::array<float, Defs::kNNInputSize>& out) noexcept
        {
            out.fill(0.0f);
            float* baseCursor = out.data();

            // 1. Encode all Board Facts (Elements + Metas)
            // state.all() returns the fixed-size underlying memory span. 
            // The index 'i' acts as the absolute positional anchor for the Transformer.
            const auto& allFacts = state.all();
            for (uint32_t i = 0; i < allFacts.size(); ++i)
            {
                float* tokenCursor = baseCursor + (i * Defs::kTokenDim);
                encodeFact(allFacts[i], tokenCursor);
            }

            // 2. Encode Action History (Reverse Chronological Order: t, t-1, t-2...)
            // This anchors the most recent move to the exact same starting tensor index.
            float* historyCursor = baseCursor + (Defs::kMaxFacts * Defs::kTokenDim);
            uint32_t histCount = std::min<uint32_t>(static_cast<uint32_t>(history.size()), Defs::kMaxHistory);

            for (uint32_t i = 0; i < histCount; ++i)
            {
                const Action& action = history[history.size() - 1 - i];
                float* tokenCursor = historyCursor + (i * Defs::kTokenDim);
                encodeAction(action, tokenCursor);
            }
        }
    };
}