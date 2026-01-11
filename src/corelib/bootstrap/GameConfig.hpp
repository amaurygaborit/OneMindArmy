#pragma once
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>
#include "../interfaces/ITraits.hpp"

// --- Helper de chargement sécurisé (Gardé tel quel) ---
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
        // Cas général (int, float, bool)
        else {
            val = node[key].as<T>();
        }
    }
    catch (const YAML::BadConversion& e) {
        throw std::runtime_error("Config Error: Bad conversion for field '" + key +
            "' (Expected type mismatch or bad format).");
    }

    // Vérification des bornes
    if (val < minVal || val > maxVal)
        throw std::runtime_error("Config Error: Value out of range for '" + key + "'");

    return val;
}

// 1. CONFIGURATION SYSTÈME (Threads, GPU)
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
        const auto& node = root["backend"]; // Section Principale

        // Device
        batchSize = loadVal<uint16_t>(node["device"], "maxBatchSize", 1, UINT16_MAX);

        // GPU Auto-detect logic
        int availableGPUs = 0;
        cudaGetDeviceCount(&availableGPUs);
        try {
            int cfg = node["device"]["numGPUs"].as<int>();
            numGPUs = (cfg <= 0 || cfg > availableGPUs) ? availableGPUs : cfg;
        }
        catch (...) { numGPUs = availableGPUs; } // "auto" -> max

        // Threading
        numSearchThreads = loadVal<uint8_t>(node["threading"], "numSearchThreads", 1, UINT8_MAX);
        numBackpropThreads = loadVal<uint8_t>(node["threading"], "numBackpropThreads", 1, UINT8_MAX);
        numInferenceThreadsPerGPU = loadVal<uint8_t>(node["threading"], "numInferenceThreads", 1, UINT8_MAX);

        // Optimization
        queueScale = loadVal<float>(node["optimization"], "queueScale", 1.0f, 100.0f);
        fastDrain = loadVal<bool>(node["optimization"], "fastDrain", 0, 1);
    }
};

// 2. CONFIGURATION MOTEUR (MCTS, Hyperparamètres)
struct MCTSConfig
{
    // Memory
    uint32_t maxNodes;
    float memoryThreshold;
    bool reuseTree;

    // Search
    uint16_t maxDepth;
    float cPUCT;
    float virtualLoss;

    // Network Context
    uint16_t historySize;

    void load(const YAML::Node& root)
    {
        const auto& node = root["engine"]; // Section Principale

        maxNodes = loadVal<uint32_t>(node["memory"], "maxNodes", 1000, UINT32_MAX);
        memoryThreshold = loadVal<float>(node["memory"], "memoryThreshold", 0.1f, 1.0f);
        reuseTree = loadVal<bool>(node["memory"], "reuseTree", 0, 1);

        maxDepth = loadVal<uint16_t>(node["search"], "maxDepth", 1, UINT16_MAX);
        cPUCT = loadVal<float>(node["search"], "cPUCT", 0.0f, 100.0f);
        virtualLoss = loadVal<float>(node["search"], "virtualLoss", 0.0f, 100.0f);

        historySize = loadVal<uint16_t>(node["network"], "historySize", 1, UINT8_MAX);
    }
};

// --- 3. SESSION (Inclut maintenant le Renderer de base) ---
struct RendererConfig
{
    bool renderState;
    bool renderValidActions;
    bool renderActionPlayed;
    bool renderResult;

    void load(const YAML::Node& root)
    {
        const auto& node = root["session"]["render"];

        renderState = loadVal<bool>(node, "renderState", 0, 1);
        renderValidActions = loadVal<bool>(node, "renderValidActions", 0, 1);
        renderActionPlayed = loadVal<bool>(node, "renderActionPlayed", 0, 1);
        renderResult = loadVal<bool>(node, "renderResult", 0, 1);
    }
};

template<typename GameTag>
struct SessionConfig
{
    uint8_t numHumans;
    uint8_t numAIs;
    bool autoInitialState;
    uint32_t numSimulations;
    float temperature;

    // On peut inclure la config de rendu ici ou la garder à part
    RendererConfig displayConfig;

    void load(const YAML::Node& root)
    {
        const auto& node = root["session"];

        numHumans = loadVal<uint8_t>(node["players"], "numHumans", 0, ITraits<GameTag>::kNumPlayers);
        numAIs = ITraits<GameTag>::kNumPlayers - numHumans;
        autoInitialState = loadVal<bool>(node["players"], "autoInitialState", 0, 1);

        numSimulations = loadVal<uint32_t>(node["timeControl"], "numSimulations", 1, UINT32_MAX);
        temperature = loadVal<float>(node["strategy"], "temperature", 0.0f, 100.0f);

        // Chargement du display
        displayConfig.load(root);
    }
};

// --- 4. TRAINING (Nouvelle Struct) ---
struct TrainingConfig
{
    uint32_t bufferSize;
    uint32_t batchSize;
    float learningRate;
    uint32_t epochs;

    void load(const YAML::Node& root)
    {
        const auto& node = root["training"]; // Section racine

        bufferSize = loadVal<uint32_t>(node, "bufferSize", 1, UINT32_MAX);
        batchSize = loadVal<uint32_t>(node, "batchSize", 1, UINT32_MAX);
        // Note: pour les floats comme learningRate, on peut utiliser max()
        learningRate = loadVal<float>(node, "learningRate", 0.000001f, 1.0f);
        epochs = loadVal<uint32_t>(node, "epochs", 1, UINT32_MAX);
    }
};