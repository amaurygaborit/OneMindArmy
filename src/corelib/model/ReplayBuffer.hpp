#pragma once

#include <array>
#include <fstream>
#include <stdexcept>
#include <cstring>

#include "../interfaces/IEngine.hpp"
#include "../util/AlignedVec.hpp"

namespace Core
{
    // ============================================================================
    // TRAINING SAMPLE (Discrete Timestep Memory)
    // Represents a single MCTS decision step (State, Policy, Value).
    // 
    // Memory layout is strictly mapped to PyTorch's Dataset requirements to 
    // enable blazingly fast Zero-Copy memory mapping (mmap) during training.
    // ============================================================================
    #pragma pack(push, 1) // Force le compilateur à coller les variables à l'octet près (0 padding)

    template<ValidGameTraits GT>
    struct TrainingSample
    {
        USING_GAME_TYPES(GT);

        std::array<float, Defs::kNNInputSize> nnInput;
        std::array<float, Defs::kActionSpace> policy;
        std::array<float, Defs::kActionSpace> legalMovesMask;
        std::array<float, Defs::kNumPlayers> result;
    };

    #pragma pack(pop) // Remet le compilateur dans son mode normal

    // ============================================================================
    // REPLAY BUFFER (Local Game Memory Streamer)
    // Collects training samples during a single Self-Play game and flushes 
    // them continuously to a binary dataset stream.
    // ============================================================================
    template<ValidGameTraits GT>
    class ReplayBuffer
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        // Using AlignedVec is absolutely critical here. Standard std::vector 
        // allocators often fail to respect over-aligned types (alignas(32)), 
        // which would cause silent memory corruption or AVX segfaults.
        AlignedVec<TrainingSample<GT>> m_samples;

    public:
        ReplayBuffer()
            // Pre-allocate to prevent dynamic memory resizing during the hot loop 
            // of a game. 512 is generally safe for typical board games 
            // (Chess avg 80 moves, Go avg 250 moves).
            : m_samples(reserve_only, 512)
        {
        }

        void clear() noexcept
        {
            m_samples.clear();
        }

        // ------------------------------------------------------------------------
        // RECORD TURN
        // Stores a single MCTS decision step. Uses pre-encoded float arrays 
        // and memcpy to eliminate redundant CPU serialization overhead.
        // ------------------------------------------------------------------------
        void recordTurn(const std::array<float, Defs::kNNInputSize>& encodedInput,
            const std::array<float, Defs::kActionSpace>& policy,
            const std::array<float, Defs::kActionSpace>& legalMask)
        {
            m_samples.emplace_back();
            auto& sample = m_samples.back();

            // 3 copies mémoires ultra-rapides et c'est plié !
            std::memcpy(sample.nnInput.data(), encodedInput.data(), Defs::kNNInputSize * sizeof(float));
            std::memcpy(sample.policy.data(), policy.data(), Defs::kActionSpace * sizeof(float));
            std::memcpy(sample.legalMovesMask.data(), legalMask.data(), Defs::kActionSpace * sizeof(float));
        }

        // ------------------------------------------------------------------------
        // FLUSH TO FILE (Binary Stream Dump)
        // Retroactively applies the true game outcome and performs a single 
        // massive binary write to the hard drive.
        // ------------------------------------------------------------------------
        void flushToFile(const GameResult& finalOutcome, std::ofstream& outFile)
        {
            if (m_samples.empty()) return;

            if (!outFile.good()) {
                throw std::runtime_error("Fatal Error: ReplayBuffer cannot write to a corrupted or closed output stream.");
            }

            // 1. Retroactively apply the true game result (Z) to every recorded step
            for (auto& sample : m_samples)
            {
                // Assuming GameResult is a std::array or trivial type, assignment is safe
                sample.result = finalOutcome.scores;
            }

            // 2. Perform a massive, single-call binary DMA write to the disk
            outFile.write(reinterpret_cast<const char*>(m_samples.data()),
                m_samples.size() * sizeof(TrainingSample<GT>));

            // 3. Reset the buffer for the next game
            clear();
        }

        [[nodiscard]] size_t size() const noexcept { return m_samples.size(); }
    };
}