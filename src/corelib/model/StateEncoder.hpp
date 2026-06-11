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
    // NEURAL NETWORK ENCODER
    // Translates the engine's structured Object-Oriented representations (Facts/Actions)
    // into flat numerical tensors digestable by the Neural Network.
    //
    // Architecture:
    // Implements Strict Positional Anchoring. Each entity in the game is assigned a 
    // permanent slot in the input tensor. This ensures Transformers can learn stable 
    // positional embeddings.
    // ============================================================================
    template<ValidGameTraits GT>
    struct StateEncoder
    {
        USING_GAME_TYPES(GT);

        // ------------------------------------------------------------------------
        // ENCODE FACT 
        // Maps an entity (piece, resource, rule) into a multi-hot spatial token.
        // 
        // Design Intent:
        // Dead or non-existent facts are still encoded as zeroed "Ghost Tokens" to 
        // prevent shifting the array indices and ruining the positional encoding.
        // Layout: [Type, FactID, OwnerID, SymLog(Value), Pos0, Pos1, ..., PosN]
        // ------------------------------------------------------------------------
        static inline void encodeFact(const Fact& fact, float* outToken) noexcept
        {
            // Identity Base
            outToken[0] = static_cast<float>(fact.type());
            outToken[1] = static_cast<float>(fact.factId());
            outToken[2] = static_cast<float>(fact.ownerId());

            // Symmetric Logarithmic Compression: sign(x) * ln(|x| + 1)
            // Neutralizes extreme value spikes while maintaining directionality.
            float val = static_cast<float>(fact.value());
            outToken[3] = std::copysign(std::log1p(std::abs(val)), val);

            // Spatial Multi-Hot Mask. Supports imperfect information where an 
            // entity might simultaneously exist across multiple possible tiles.
            for (uint32_t i = 0; i < Defs::kNumPos; ++i) {
                outToken[4 + i] = fact.isPossiblePos(i) ? 1.0f : 0.0f;
            }
        }

        // ------------------------------------------------------------------------
        // ENCODE ACTION 
        // Maps historical moves into directional spatial dipoles.
        // 
        // Design Intent:
        // Highlights trajectory by applying a negative signal (-1.0) to the origin 
        // square and a positive signal (+1.0) to the destination square.
        // ------------------------------------------------------------------------
        static inline void encodeAction(const Action& action, float* outToken) noexcept
        {
            outToken[0] = static_cast<float>(action.type());
            outToken[1] = static_cast<float>(action.factId());
            outToken[2] = static_cast<float>(action.ownerId());

            float val = static_cast<float>(action.value());
            outToken[3] = std::copysign(std::log1p(std::abs(val)), val);

            for (uint32_t i = 0; i < Defs::kNumPos; ++i) {
                outToken[4 + i] = 0.0f;
            }

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
        // FULL STATE ENCODING 
        // Constructs the final concatenated array sent to TensorRT.
        // Memory Layout Guarantee:
        // [ 0 ... kMaxElems-1 ]        -> Physical Board Pieces
        // [ kMaxElems ... kMaxFacts-1] -> Metadata (Rules, Turns, Scores)
        // [ kMaxFacts ... end ]        -> Reverse Chronological Action History
        // ------------------------------------------------------------------------
        static inline void encode(const State& state, std::span<const Action> history,
            std::array<float, Defs::kNNInputSize>& out) noexcept
        {
            out.fill(0.0f);
            float* baseCursor = out.data();

            const auto& allFacts = state.all();
            for (uint32_t i = 0; i < allFacts.size(); ++i)
            {
                float* tokenCursor = baseCursor + (i * Defs::kTokenDim);
                encodeFact(allFacts[i], tokenCursor);
            }

            // Anchors the most recent move to a fixed, predictable tensor index.
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