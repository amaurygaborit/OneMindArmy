#pragma once

#include <array>
#include <fstream>
#include <random>
#include <stdexcept>
#include <cstring>

#include "../interfaces/IEngine.hpp"
#include "../util/AlignedVec.hpp"

namespace Core
{
    // ============================================================================
    // TRAINING SAMPLE (Discrete Timestep Memory)
    //
    // Memory layout is strictly mapped to PyTorch's Dataset requirements to
    // enable blazingly fast Zero-Copy memory mapping (mmap) during training.
    //
    // #pragma pack(1) ensures zero padding between fields — the Python dataset
    // reads this binary layout directly via numpy memmap.
    // ============================================================================
#pragma pack(push, 1)

    template<ValidGameTraits GT>
    struct TrainingSample
    {
        USING_GAME_TYPES(GT);

        static constexpr size_t kMaskBytes = (Defs::kActionSpace + 7) / 8;

        std::array<float, Defs::kNNInputSize>    nnInput;
        std::array<float, Defs::kActionSpace>    policy;
        std::array<float, Defs::kNumPlayers * 3> wdlTarget;      // WDL per player
        std::array<uint8_t, kMaskBytes>          legalMovesMask; // bit-packed
    };

#pragma pack(pop)

    // ============================================================================
    // REPLAY BUFFER (Local Game Memory Streamer)
    //
    // Collects training samples during a single Self-Play game, then flushes
    // them to a binary file in a single write.
    //
    // Draw filtering (drawSampleRate):
    //   A game is considered a draw if wdl[0*3+1] > 0.5 (Draw is the dominant
    //   outcome for player 0). When a draw is detected, flushToFile keeps it
    //   with probability `drawSampleRate` and discards it otherwise.
    //
    //   This reduces the draw bias in the replay buffer during early training
    //   when most games end in a draw by adjudication. It should be relaxed
    //   progressively as the network matures (iter ~50+).
    //
    //   drawSampleRate = 1.0 → keep all draws (no filtering)
    //   drawSampleRate = 0.2 → keep only 20% of draws
    // ============================================================================
    template<ValidGameTraits GT>
    class ReplayBuffer
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        AlignedVec<TrainingSample<GT>> m_samples;

        // Thread-local RNG — one per worker thread, no contention
        static std::mt19937& getRng()
        {
            thread_local std::mt19937 rng{ std::random_device{}() };
            return rng;
        }

    public:
        ReplayBuffer()
            : m_samples(reserve_only, 512)  // pre-allocate for ~512 plies (safe for chess/go)
        {
        }

        void clear() noexcept { m_samples.clear(); }

        [[nodiscard]] size_t size() const noexcept { return m_samples.size(); }

        // ------------------------------------------------------------------------
        // RECORD TURN
        // Stores a single MCTS decision step (state + policy + legal mask).
        // The WDL target is filled retroactively in flushToFile().
        // ------------------------------------------------------------------------
        void recordTurn(
            const std::array<float, Defs::kNNInputSize>& encodedInput,
            const std::array<float, Defs::kActionSpace>& policy,
            const std::array<bool, Defs::kActionSpace>& legalMaskBool)
        {
            m_samples.emplace_back();
            auto& sample = m_samples.back();

            std::memcpy(sample.nnInput.data(),
                encodedInput.data(),
                Defs::kNNInputSize * sizeof(float));

            std::memcpy(sample.policy.data(),
                policy.data(),
                Defs::kActionSpace * sizeof(float));

            // Bit-pack the legal mask: bit i set ↔ legalMaskBool[i] == true
            // Uses little-endian convention: bit i → byte[i/8], bit (i%8)
            // This matches np.unpackbits(..., bitorder='little') in dataset.py
            std::memset(sample.legalMovesMask.data(), 0, sample.legalMovesMask.size());
            for (size_t i = 0; i < Defs::kActionSpace; ++i) {
                if (legalMaskBool[i])
                    sample.legalMovesMask[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }

            // wdlTarget is intentionally left zeroed here — filled in flushToFile()
        }

        // ------------------------------------------------------------------------
        // FLUSH TO FILE (Binary Stream Dump)
        //
        // 1. Applies draw filtering: if the game is a draw and a random draw
        //    exceeds drawSampleRate, the whole game is silently discarded.
        // 2. Retroactively stamps the WDL target on every recorded sample.
        // 3. Writes all samples in a single contiguous binary write.
        //
        // Returns true if the game was written, false if it was filtered out.
        // ------------------------------------------------------------------------
        bool flushToFile(
            const GameResult& finalOutcome,
            std::ofstream& outFile,
            float drawSampleRate)
        {
            if (m_samples.empty()) { return false; }

            if (!outFile.good())
                throw std::runtime_error("ReplayBuffer: output stream is not writable.");

            // ----------------------------------------------------------------
            // Draw filtering
            // A game is a draw when the Draw probability of player 0 dominates.
            // We use wdl[0*3 + 1] (= P(Draw) for player 0) as the indicator.
            // drawSampleRate = 1.0 disables filtering entirely.
            // ----------------------------------------------------------------
            if (drawSampleRate < 1.0f)
            {
                const bool isDraw = (finalOutcome.wdl[0 * 3 + 1] > 0.5f);
                if (isDraw)
                {
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    if (dist(getRng()) > drawSampleRate)
                    {
                        clear();
                        return false;   // game discarded
                    }
                }
            }

            // ----------------------------------------------------------------
            // Retroactively stamp the true WDL target on every sample
            // ----------------------------------------------------------------
            for (auto& sample : m_samples)
                sample.wdlTarget = finalOutcome.wdl;

            // ----------------------------------------------------------------
            // Single contiguous binary write
            // ----------------------------------------------------------------
            outFile.write(
                reinterpret_cast<const char*>(m_samples.data()),
                static_cast<std::streamsize>(m_samples.size() * sizeof(TrainingSample<GT>)));

            clear();
            return true;
        }
    };
}