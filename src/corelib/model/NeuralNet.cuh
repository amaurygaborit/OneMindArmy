#pragma once
#include "../interfaces/ITraits.hpp"
#include "../bootstrap/GameConfig.hpp"
#include "../AlignedVec.hpp"

#include <string>
#include <math.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>
#include <cassert>

template<typename GameTag>
struct ModelResultsT
{
    AlignedVec<float> values;       // values for every players at the leaf
    AlignedVec<float> policy;       // policy at the leaf

    ModelResultsT() noexcept
        : values(ITraits<GameTag>::kNumPlayers)
        , policy(ITraits<GameTag>::kActionSpace)
    {
    }
};

template<typename GameTag>
class NeuralNet
{
private:
    using GT = ITraits<GameTag>;
    using FactStateAction = typename FactStateActionT<GameTag>;
    using ModelResults = ModelResultsT<GameTag>;

    int m_deviceId;

public:
    NeuralNet(int deviceId)
        : m_deviceId(deviceId)
    {
        cudaSetDevice(m_deviceId);
        std::cout << "NeuralNet initialized on GPU " << m_deviceId << std::endl;
    }

    void forwardBatch(const AlignedVec<FactStateAction>& inferenceBuf,
        AlignedVec<ModelResults>& resultsBuf)
    {
        cudaSetDevice(m_deviceId);

        // 1. Simulation de la latence GPU
        std::this_thread::sleep_for(std::chrono::microseconds(40000));

        const size_t numInferences = resultsBuf.size();

        // 2. Génération Aléatoire
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_real_distribution<float> distPolicy(0.0f, 1.0f);
        std::uniform_real_distribution<float> distValue(-1.0f, 1.0f);

        for (size_t b = 0; b < numInferences; ++b)
        {
            ModelResults& results = resultsBuf[b];

            // A. Policy (Random probas normalisées)
            float sum = 0.0f;
            for (size_t i = 0; i < GT::kActionSpace; ++i)
            {
                float val = distPolicy(generator);
                results.policy[i] = val;
                sum += val;
            }

            if (sum > 1e-6f) {
                float norm = 1.0f / sum;
                for (size_t i = 0; i < GT::kActionSpace; ++i) results.policy[i] *= norm;
            }
            else {
                float uniform = 1.0f / GT::kActionSpace;
                std::fill(results.policy.begin(), results.policy.end(), uniform);
            }

            // B. Values
            for (size_t p = 0; p < GT::kNumPlayers; ++p)
            {
                results.values[p] = distValue(generator);
            }
        }
    }
};