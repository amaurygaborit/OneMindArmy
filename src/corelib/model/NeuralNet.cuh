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
    // belief retiré ou inutilisé pour ce test simple

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
    using IdxStateAction = typename IdxStateActionT<GameTag>;
    using ModelResults = ModelResultsT<GameTag>;

    int m_deviceId;

public:
    NeuralNet(int deviceId)
        : m_deviceId(deviceId)
    {
        cudaSetDevice(m_deviceId);
		std::cout << "NeuralNet initialized on GPU " << m_deviceId << std::endl;
    }

    void forwardBatch(const AlignedVec<IdxStateAction>& inferenceBuf,
        AlignedVec<ModelResults>& resultsBuf)
    {
        cudaSetDevice(m_deviceId); // Sécurité en cas de changement de contexte

        // 1. Simulation de la latence GPU (important pour tester le multithreading)
        std::this_thread::sleep_for(std::chrono::microseconds(1000));

        const size_t numInferences = resultsBuf.size();

        // Sécurité : On s'assure qu'on a bien reçu des inputs pour nos outputs
        // (Attention : cela suppose que l'historique a une taille fixe connue, sinon retirer l'assert)
        // assert(inferenceBuf.size() >= numInferences); 

        // 2. Génération Aléatoire (Pour voir le MCTS faire des choix différents)
        // Utilisation de thread_local pour éviter les verrous et être thread-safe
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

            // Normalisation pour éviter le bug de Division par Zéro plus tard
            if (sum > 1e-6f) {
                float norm = 1.0f / sum;
                for (size_t i = 0; i < GT::kActionSpace; ++i) results.policy[i] *= norm;
            }
            else {
                // Fallback uniforme
                float uniform = 1.0f / GT::kActionSpace;
                std::fill(results.policy.begin(), results.policy.end(), uniform);
            }

            // B. Values (Random entre -1 et 1)
            for (size_t p = 0; p < GT::kNumPlayers; ++p)
            {
                results.values[p] = distValue(generator);
            }
        }
    }
};