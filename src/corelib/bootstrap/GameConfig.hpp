#pragma once

#include <iostream>
#include <string>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <cuda_runtime.h>

namespace Core
{
    // ============================================================================
    // SAFE YAML LOADER HELPER
    // Ensures missing fields or out-of-bounds values trigger explicit exceptions.
    // ============================================================================
    template <typename T>
    T loadVal(const YAML::Node& node, const std::string& key, T minVal, T maxVal)
    {
        if (!node[key])
            throw std::runtime_error("Config Error: Strict Mode - Missing mandatory field '" + key + "'");

        T val;
        try {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                int temp = node[key].as<int>();
                if (temp < static_cast<int>(minVal) || temp > static_cast<int>(maxVal))
                    throw std::runtime_error("Config Error: Value out of range for '" + key + "'");
                return static_cast<T>(temp);
            }
            else {
                val = node[key].as<T>();
            }
        }
        catch (const YAML::BadConversion& e) {
            throw std::runtime_error("Config Error: Bad conversion (wrong type) for field '" + key + "'");
        }

        if (val < minVal || val > maxVal)
            throw std::runtime_error("Config Error: Value out of range for '" + key + "'");

        return val;
    }

    // ============================================================================
    // 1. NETWORK CONFIGURATION 
    // ============================================================================
    struct NetworkConfig
    {
        uint32_t dModel = 0;
        uint32_t nHeads = 0;
        uint32_t nLayers = 0;
        uint32_t dimFeedforward = 0;

        void load(const YAML::Node& root, const std::string& runMode)
        {
            const auto& node = root["network"];

            // Le r�seau est obligatoire pour l'entra�nement ou l'export. En mode play, TensorRT s'en fiche.
            if (!node) {
                if (runMode == "train" || runMode == "export-meta")
                    throw std::runtime_error("Config Error: Missing 'network' block (Mandatory for " + runMode + ").");
                return; // Optionnel en mode "play"
            }

            dModel = loadVal<uint32_t>(node, "dModel", 1u, 8192u);
            nHeads = loadVal<uint32_t>(node, "nHeads", 1u, 128u);
            nLayers = loadVal<uint32_t>(node, "nLayers", 1u, 128u);
            dimFeedforward = loadVal<uint32_t>(node, "dimFeedforward", 1u, 32768u);
        }
    };

    // ============================================================================
    // 2. ENGINE CONFIGURATION (MCTS)
    // ============================================================================
    struct EngineConfig
    {
        uint32_t numSimulations;
        uint32_t maxDepth;
        float    cPUCT;
        float    virtualLoss;
        float    temperatureDropPly;

        uint32_t gumbelK;
        float    gumbelCVisit;
        float    gumbelCScale;

		float    fpuValue;
        uint32_t maxNodes;
        float    memoryThreshold;
        bool     reuseTree;
        float    resignThreshold;
        uint32_t resignMinPly;

        void load(const YAML::Node& root, const std::string& /*runMode*/)
        {
            const auto& node = root["engine"];
            if (!node) throw std::runtime_error("Config Error: Missing 'engine' block (Always mandatory).");

            numSimulations = loadVal<uint32_t>(node, "numSimulations", 1u, UINT32_MAX);
            maxDepth = loadVal<uint32_t>(node, "maxDepth", 1u, UINT32_MAX);
            cPUCT = loadVal<float>(node, "cPUCT", 0.0f, 100.0f);
            virtualLoss = loadVal<float>(node, "virtualLoss", 0.0f, 100.0f);
			temperatureDropPly = loadVal<float>(node, "temperatureDropPly", 0.0f, 100.0f);

            gumbelK = loadVal<uint32_t>(node, "gumbelK", 0u, UINT32_MAX);
            gumbelCVisit = loadVal<float>(node, "gumbelCVisit", 0.0f, INFINITY);
            gumbelCScale = loadVal<float>(node, "gumbelCScale", 0.0f, INFINITY);

			fpuValue = loadVal<float>(node, "fpuValue", -100.0f, 100.0f);
            maxNodes = loadVal<uint32_t>(node, "maxNodes", 1u, UINT32_MAX);
            memoryThreshold = loadVal<float>(node, "memoryThreshold", 0.1f, 1.0f);
            reuseTree = loadVal<bool>(node, "reuseTree", false, true);
            resignThreshold = loadVal<float>(node, "resignThreshold", -2.0f, 0.0f);
            resignMinPly = loadVal<uint32_t>(node, "resignMinPly", 1u, UINT16_MAX);
        }
    };

    // ============================================================================
    // 3. TRAINING CONFIGURATION 
    // ============================================================================
    struct TrainingConfig
    {
        uint32_t gamesPerIteration = 0;
        uint32_t epochs = 0;
        uint32_t trainBatchSize = 0;
        float learningRate = 0.0f;
        float weightDecay = 0.0f;
        float valueLossWeight = 0.0f;
		uint32_t currentIteration = 0;
		float drawSampleRate = 0.0f;

        void load(const YAML::Node& root, const std::string& runMode)
        {
            const auto& node = root["training"];

            if (!node) {
                if (runMode == "train")
                    throw std::runtime_error("Config Error: Missing 'training' block (Mandatory for training).");
                return; // Ignor� proprement en mode play
            }

            gamesPerIteration = loadVal<uint32_t>(node, "gamesPerIteration", 1u, UINT32_MAX);
            epochs = loadVal<uint32_t>(node, "epochs", 1u, 10000u);
            trainBatchSize = loadVal<uint32_t>(node, "trainBatchSize", 1u, 8192u);
            learningRate = loadVal<float>(node, "learningRate", 0.0000001f, 1.0f);
            weightDecay = loadVal<float>(node, "weightDecay", 0.0f, 1.0f);
            valueLossWeight = loadVal<float>(node, "valueLossWeight", 0.0f, 10.0f);
			currentIteration = loadVal<uint32_t>(node, "currentIteration", 0u, UINT32_MAX);
            drawSampleRate = loadVal<float>(node, "drawSampleRate", 0.0f, 1.0f);
        }
    };

    // ============================================================================
    // 4. BACKEND CONFIGURATION 
    // ============================================================================
    struct BackendConfig
    {
        uint32_t numGPUs;
        uint32_t inferenceBatchSize;
        uint32_t numParallelGames;
        std::string precision;
        uint32_t numSearchThreads;
        uint32_t numBackpropThreads;
        uint32_t numInferenceThreads;
        float queueScale;
        bool fastDrain;

        void load(const YAML::Node& root, const std::string& /*runMode*/)
        {
            const auto& node = root["backend"];
            if (!node) throw std::runtime_error("Config Error: Missing 'backend' block (Always mandatory).");

            int cudaCount = 0;
            cudaError_t err = cudaGetDeviceCount(&cudaCount);
            uint32_t availableGPUs = (err == cudaSuccess && cudaCount > 0) ? static_cast<uint32_t>(cudaCount) : 1u;

            try {
                if (node["numGPUs"]) {
                    std::string val = node["numGPUs"].as<std::string>();
                    if (val == "auto") numGPUs = availableGPUs;
                    else {
                        uint32_t cfg = std::stoi(val);
                        numGPUs = (cfg == 0 || cfg > availableGPUs) ? availableGPUs : cfg;
                    }
                }
                else {
                    numGPUs = availableGPUs;
                }
            }
            catch (...) { numGPUs = availableGPUs; }

            inferenceBatchSize = loadVal<uint32_t>(node, "inferenceBatchSize", 1u, UINT32_MAX);
            numParallelGames = loadVal<uint32_t>(node, "numParallelGames", 1u, UINT32_MAX);

            if (node["precision"]) precision = node["precision"].as<std::string>();
            else precision = "fp16";

            numSearchThreads = loadVal<uint32_t>(node, "numSearchThreads", 1u, 1024u);
            numBackpropThreads = loadVal<uint32_t>(node, "numBackpropThreads", 1u, 1024u);
            numInferenceThreads = loadVal<uint32_t>(node, "numInferenceThreads", 1u, 1024u);
            queueScale = loadVal<float>(node, "queueScale", 1.0f, 100.0f);
            fastDrain = loadVal<bool>(node, "fastDrain", false, true);
        }
    };

    // ============================================================================
    // 5. SESSION CONFIGURATION 
    // ============================================================================
    template<typename GameConfig>
    struct SessionConfig
    {
        uint32_t numAIs = 0;
        bool autoInitialState = false;
        float maxTimePerMove = 0.0f;
        float temperature = 0.0f;
        bool verbose = false;

        bool renderState = false;
        bool renderValidActions = false;
        bool renderActionPlayed = false;
        bool renderResult = false;

        void load(const YAML::Node& root, const std::string& runMode)
        {
            const auto& node = root["session"];

            if (!node) {
                if (runMode == "play")
                    throw std::runtime_error("Config Error: Missing 'session' block (Mandatory for match play).");
                return; // Ignor� proprement en training
            }

            numAIs = loadVal<uint32_t>(node, "numAIs", 0u, GameConfig::kNumPlayers);
            autoInitialState = loadVal<bool>(node, "autoInitialState", false, true);
            maxTimePerMove = loadVal<float>(node, "maxTimePerMove", 0.0f, 10000.0f);
            temperature = loadVal<float>(node, "temperature", 0.0f, 100.0f);
            verbose = loadVal<bool>(node, "verbose", false, true);

            renderState = loadVal<bool>(node, "renderState", false, true);
            renderValidActions = loadVal<bool>(node, "renderValidActions", false, true);
            renderActionPlayed = loadVal<bool>(node, "renderActionPlayed", false, true);
            renderResult = loadVal<bool>(node, "renderResult", false, true);
        }
    };
}