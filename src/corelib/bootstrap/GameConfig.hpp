#pragma once
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>
#include <stdexcept>

namespace Core
{
    // --- Helper de chargement sécurisé ---
    template <typename T>
    T loadVal(const YAML::Node& node, const std::string& key, T minVal, T maxVal)
    {
        if (!node[key])
            throw std::runtime_error("Config Error: Missing field '" + key + "'");

        T val;

        try {
            // Cas spécial : uint8_t est souvent lu comme un char par yaml-cpp
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                int temp = node[key].as<int>();
                if (temp < static_cast<int>(minVal) || temp > static_cast<int>(maxVal))
                    throw std::runtime_error("Config Error: Value out of range for '" + key + "'");
                return static_cast<T>(temp);
            }
            // Cas général
            else {
                val = node[key].as<T>();
            }
        }
        catch (const YAML::BadConversion& e) {
            throw std::runtime_error("Config Error: Bad conversion for field '" + key + "'");
        }

        if (val < minVal || val > maxVal)
            throw std::runtime_error("Config Error: Value out of range for '" + key + "'");

        return val;
    }

    // ============================================================================
    // 1. CONFIGURATION SYSTÈME (Hardware)
    // ============================================================================
    struct SystemConfig
    {
        int numGPUs;
        uint16_t batchSize;
        uint8_t numSearchThreads;
        uint8_t numBackpropThreads;
        uint8_t numInferenceThreadsPerGPU;
        float queueScale;
        bool fastDrain;

        void load(const YAML::Node& root)
        {
            const auto& node = root["backend"];

            batchSize = loadVal<uint16_t>(node["device"], "maxBatchSize", 1, UINT16_MAX);

            int availableGPUs = 0;
            cudaGetDeviceCount(&availableGPUs);
            try {
                std::string val = node["device"]["numGPUs"].as<std::string>();
                if (val == "auto") numGPUs = availableGPUs;
                else {
                    int cfg = std::stoi(val);
                    numGPUs = (cfg <= 0 || cfg > availableGPUs) ? availableGPUs : cfg;
                }
            }
            catch (...) { numGPUs = availableGPUs; }

            numSearchThreads = loadVal<uint8_t>(node["threading"], "numSearchThreads", 1, UINT8_MAX);
            numBackpropThreads = loadVal<uint8_t>(node["threading"], "numBackpropThreads", 1, UINT8_MAX);
            numInferenceThreadsPerGPU = loadVal<uint8_t>(node["threading"], "numInferenceThreads", 1, UINT8_MAX);

            queueScale = loadVal<float>(node["optimization"], "queueScale", 1.0f, 100.0f);
            fastDrain = loadVal<bool>(node["optimization"], "fastDrain", 0, 1);
        }
    };

    // ============================================================================
    // 2. CONFIGURATION MOTEUR (Tree Search)
    // ============================================================================
    struct TreeSearchConfig
    {
        uint32_t maxNodes;
        float memoryThreshold;
        bool reuseTree;

        uint16_t maxDepth;
        float cPUCT;
        float virtualLoss;
        uint16_t historySize;

        void load(const YAML::Node& root)
        {
            const auto& node = root["engine"];

            maxNodes = loadVal<uint32_t>(node["memory"], "maxNodes", 1000, UINT32_MAX);
            memoryThreshold = loadVal<float>(node["memory"], "memoryThreshold", 0.1f, 1.0f);
            reuseTree = loadVal<bool>(node["memory"], "reuseTree", 0, 1);

            maxDepth = loadVal<uint16_t>(node["search"], "maxDepth", 1, UINT16_MAX);
            cPUCT = loadVal<float>(node["search"], "cPUCT", 0.0f, 100.0f);
            virtualLoss = loadVal<float>(node["search"], "virtualLoss", 0.0f, 100.0f);

            historySize = loadVal<uint16_t>(node["network"], "historySize", 1, UINT8_MAX);
        }
    };

    // ============================================================================
    // 3. CONFIGURATION RENDU (Display)
    // ============================================================================
    struct RendererConfig
    {
        bool renderState;
        bool renderValidActions;
        bool renderActionPlayed;
        bool renderResult;

        void load(const YAML::Node& root)
        {
            const auto& node = root["session"]["render"]; // Sous-section session

            renderState = loadVal<bool>(node, "renderState", 0, 1);
            renderValidActions = loadVal<bool>(node, "renderValidActions", 0, 1);
            renderActionPlayed = loadVal<bool>(node, "renderActionPlayed", 0, 1);
            renderResult = loadVal<bool>(node, "renderResult", 0, 1);
        }
    };

    // ============================================================================
    // 4. CONFIGURATION SESSION (Match)
    // ============================================================================
    // C'est ici qu'on a besoin du Template GameConfig (ex: ChessTypes)
    // pour connaître le nombre maximum de joueurs possibles (kNumPlayers).

    template<typename GameConfig>
    struct SessionConfig
    {
        uint8_t numHumans;
        uint8_t numAIs;
        bool autoInitialState;
        uint32_t numSimulations;
        float temperature;

        RendererConfig displayConfig;

        void load(const YAML::Node& root)
        {
            const auto& node = root["session"];

            // Utilisation directe de GameConfig::kNumPlayers (plus besoin de ITraits)
            numHumans = loadVal<uint8_t>(node["players"], "numHumans", 0, GameConfig::kNumPlayers);

            // Calcul automatique des IA restantes
            numAIs = static_cast<uint8_t>(GameConfig::kNumPlayers - numHumans);

            autoInitialState = loadVal<bool>(node["players"], "autoInitialState", 0, 1);

            numSimulations = loadVal<uint32_t>(node["timeControl"], "numSimulations", 1, UINT32_MAX);
            temperature = loadVal<float>(node["strategy"], "temperature", 0.0f, 100.0f);

            displayConfig.load(root);
        }
    };

    // ============================================================================
    // 5. CONFIGURATION TRAINING (Apprentissage)
    // ============================================================================
    struct TrainingConfig
    {
        uint32_t bufferSize;
        uint32_t batchSize;
        float learningRate;
        uint32_t epochs;

        void load(const YAML::Node& root)
        {
            const auto& node = root["training"];

            bufferSize = loadVal<uint32_t>(node, "bufferSize", 1, UINT32_MAX);
            batchSize = loadVal<uint32_t>(node, "batchSize", 1, UINT32_MAX);
            learningRate = loadVal<float>(node, "learningRate", 0.000001f, 1.0f);
            epochs = loadVal<uint32_t>(node, "epochs", 1, UINT32_MAX);
        }
    };
}