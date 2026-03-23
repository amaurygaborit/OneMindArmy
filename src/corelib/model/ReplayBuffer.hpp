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
#pragma pack(push, 1)

    template<ValidGameTraits GT>
    struct TrainingSample
    {
        USING_GAME_TYPES(GT);

        static constexpr size_t kMaskBytes = (Defs::kActionSpace + 7) / 8;

        std::array<float, Defs::kNNInputSize>    nnInput;
        std::array<float, Defs::kActionSpace>    policy;
        std::array<float, Defs::kNumPlayers * 3> wdlTarget;
        std::array<uint8_t, kMaskBytes>          legalMovesMask;
    };

#pragma pack(pop)

    template<ValidGameTraits GT>
    class ReplayBuffer
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        AlignedVec<TrainingSample<GT>> m_samples;
        AlignedVec<uint32_t>           m_viewers; // Utilisation de AlignedVec pour correspondre à m_samples

        static std::mt19937& getRng()
        {
            thread_local std::mt19937 rng{ std::random_device{}() };
            return rng;
        }

    public:
        ReplayBuffer()
            : m_samples(reserve_only, 512)
            , m_viewers(reserve_only, 512)
        {
        }

        void clear() noexcept
        {
            m_samples.clear();
            m_viewers.clear();
        }

        [[nodiscard]] size_t size() const noexcept { return m_samples.size(); }

        void recordTurn(
            const std::array<float, Defs::kNNInputSize>& encodedInput,
            const std::array<float, Defs::kActionSpace>& policy,
            const std::array<bool, Defs::kActionSpace>& legalMaskBool,
            uint32_t currentPlayer) // NOUVEAU : On trace qui devait jouer
        {
            m_samples.emplace_back();
            m_viewers.push_back(currentPlayer);
            auto& sample = m_samples.back();

            std::memcpy(sample.nnInput.data(), encodedInput.data(), Defs::kNNInputSize * sizeof(float));
            std::memcpy(sample.policy.data(), policy.data(), Defs::kActionSpace * sizeof(float));

            std::memset(sample.legalMovesMask.data(), 0, sample.legalMovesMask.size());
            for (size_t i = 0; i < Defs::kActionSpace; ++i) {
                if (legalMaskBool[i])
                    sample.legalMovesMask[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }
        }

        bool flushToFile(
            const GameResult& finalOutcome,
            std::ofstream& outFile,
            float drawSampleRate)
        {
            if (m_samples.empty()) { return false; }

            if (!outFile.good())
                throw std::runtime_error("ReplayBuffer: output stream is not writable.");

            // 1. Filtrage des parties nulles
            if (drawSampleRate < 1.0f)
            {
                const bool isDraw = [&]() {
                    for (uint32_t p = 0; p < Defs::kNumPlayers; ++p)
                        if (finalOutcome.wdl[p * 3 + 0] > 0.5f) return false; // quelqu'un a gagné
                    return true;
                    }();

                if (isDraw)
                {
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    if (dist(getRng()) > drawSampleRate)
                    {
                        clear();
                        return false;
                    }
                }
            }

            // 2. OPTIMISATION WDL : Précalcul des cibles pour chaque POV possible
            // precomputedWDL[viewer_id] contiendra l'array WDL exactement tourné pour ce joueur.
            std::array<std::array<float, Defs::kNumPlayers * 3>, Defs::kNumPlayers> precomputedWDL{};

            for (uint32_t viewer = 0; viewer < Defs::kNumPlayers; ++viewer)
            {
                for (uint32_t p = 0; p < Defs::kNumPlayers; ++p)
                {
                    uint32_t absPlayer = (viewer + p) % Defs::kNumPlayers;
                    precomputedWDL[viewer][p * 3 + 0] = finalOutcome.wdl[absPlayer * 3 + 0];
                    precomputedWDL[viewer][p * 3 + 1] = finalOutcome.wdl[absPlayer * 3 + 1];
                    precomputedWDL[viewer][p * 3 + 2] = finalOutcome.wdl[absPlayer * 3 + 2];
                }
            }

            // 3. Application vectorisée (plus besoin de modulos, juste une copie mémoire)
            for (size_t i = 0; i < m_samples.size(); ++i) {
                m_samples[i].wdlTarget = precomputedWDL[m_viewers[i]];
            }

            // 4. Dump binaire brut
            outFile.write(
                reinterpret_cast<const char*>(m_samples.data()),
                static_cast<std::streamsize>(m_samples.size() * sizeof(TrainingSample<GT>)));

            clear();
            return true;
        }
    };
}