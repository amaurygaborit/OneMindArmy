#pragma once
#include <iostream>
#include <memory>

#include "TypeResolver.hpp"
#include "../interfaces/IHandler.hpp"
#include "../model/MCTSThreadPool.hpp"

template<
    typename GameTag,
    class EngineT,
    class RequesterT,
    class RendererT,
    class HandlerT
>
class GameTypeRegistry : public TypeResolverBase
{
private:
    const std::string m_gameName;

public:
    GameTypeRegistry(const std::string& gameName)
        : m_gameName(gameName)
    {
        TypeResolverRegistry::instance().registerResolver(m_gameName, this);
    }

    // NeuralNet
    std::shared_ptr<NeuralNet<GameTag>> createNeuralNet(
        const YAML::Node& config) const
    {
        if (!config["common"]["model"]["mctsBatchSize"])
            throw std::runtime_error("Configuration missing 'common.model.mctsBatchSize' field.");

        size_t batchSize = config["common"]["model"]["mctsBatchSize"].as<size_t>();
        if (batchSize < 1)
            throw std::runtime_error("Wrong configuration: mctsBatchSize < 1");

        auto neuralNet = std::make_shared<NeuralNet<GameTag>>(static_cast<uint16_t>(batchSize));
        return neuralNet;
    }

    // Thread Pool for all MCTS
    std::shared_ptr<MCTSThreadPool<GameTag>> createThreadPool(
        const YAML::Node& config,
        std::shared_ptr<NeuralNet<GameTag>> neuralNet) const
    {
        if (!config["common"]["model"]["mctsBatchSize"])
            throw std::runtime_error("Configuration missing 'common.model.mctsBatchSize' field.");
        if (!config["common"]["model"]["historySize"])
            throw std::runtime_error("Configuration missing 'common.model.historySize' field.");
        if (!config["common"]["model"]["maxDepth"])
            throw std::runtime_error("Configuration missing 'common.model.maxDepth' field.");
        if (!config["common"]["model"]["numThreads"])
            throw std::runtime_error("Configuration missing 'common.model.numThreads' field.");
        if (!config["common"]["model"]["maxSamples"])
            throw std::runtime_error("Configuration missing 'common.model.maxSamples' field.");

        size_t batchSize = config["common"]["model"]["mctsBatchSize"].as<size_t>();
        if (batchSize < 1)
            throw std::runtime_error("Wrong configuration: mctsBatchSize < 1");

        size_t historySize = config["common"]["model"]["historySize"].as<size_t>();
        if (historySize < 1)
            throw std::runtime_error("Wrong configuration: historySize < 1");

        size_t maxDepth = config["common"]["model"]["maxDepth"].as<size_t>();
        if (maxDepth < 1)
            throw std::runtime_error("Wrong configuration: maxDepth < 1");

        size_t numThreads = config["common"]["model"]["numThreads"].as<size_t>();
        if (numThreads < 1 || numThreads > 64)
            throw std::runtime_error("Wrong configuration: numThreads must be between 1 and 64");

        size_t maxSamples = config["common"]["model"]["maxSamples"].as<size_t>();
        if (maxSamples < 1)
            throw std::runtime_error("Wrong configuration: maxSamples < 1");

        return std::make_shared<MCTSThreadPool<GameTag>>(
            neuralNet,
            static_cast<uint16_t>(batchSize),
            static_cast<uint8_t>(historySize),
            static_cast<uint16_t>(maxDepth),
            static_cast<uint8_t>(numThreads)
        );
    }

    // MCTS
    void createMCTS(
        const YAML::Node& config,
        AlignedVec<std::shared_ptr<MCTS<GameTag>>>& mctsVec,
        const std::shared_ptr<EngineT>& engine,
        size_t numAI) const
    {
        if (!config["common"]["model"]["maxNodes"])
            throw std::runtime_error("Configuration missing 'common.model.maxNodes' field.");
        if (!config["common"]["model"]["maxSamples"])
            throw std::runtime_error("Configuration missing 'common.model.maxSamples' field.");
        if (!config["common"]["model"]["historySize"])
            throw std::runtime_error("Configuration missing 'common.model.historySize' field.");
        if (!config["common"]["model"]["maxDepth"])
            throw std::runtime_error("Configuration missing 'common.model.maxDepth' field.");
        if (!config["common"]["model"]["cPUCT"])
            throw std::runtime_error("Configuration missing 'common.model.cPUCT' field.");
        if (!config["common"]["model"]["virtualLoss"])
            throw std::runtime_error("Configuration missing 'common.model.virtualLoss' field.");
        if (!config["common"]["model"]["keepK"])
            throw std::runtime_error("Configuration missing 'common.model.keepK' field.");

        size_t maxNodes = config["common"]["model"]["maxNodes"].as<size_t>();
        if (maxNodes < 1)
            throw std::runtime_error("Wrong configuration: maxNodes < 1");

        size_t maxSamples = config["common"]["model"]["maxSamples"].as<size_t>();
        if (maxSamples < 1)
            throw std::runtime_error("Wrong configuration: maxSamples < 1");

        size_t historySize = config["common"]["model"]["historySize"].as<size_t>();
        if (historySize < 1)
            throw std::runtime_error("Wrong configuration: historySize < 1");

        size_t maxDepth = config["common"]["model"]["maxDepth"].as<size_t>();
        if (maxDepth < 1)
            throw std::runtime_error("Wrong configuration: maxDepth < 1");

        float cPUCT = config["common"]["model"]["cPUCT"].as<float>();
        if (cPUCT < 0.0f)
            throw std::runtime_error("Wrong configuration: cPUCT < 0");

        float virtualLoss = config["common"]["model"]["virtualLoss"].as<float>();
        if (virtualLoss < 0.0f)
            throw std::runtime_error("Wrong configuration: virtualLoss < 0");

        size_t keepK = config["common"]["model"]["keepK"].as<size_t>();
        if (keepK < 1)
            throw std::runtime_error("Wrong configuration: keepK < 1");

        // Créer les instances MCTS (sans threads)
        for (size_t i = 0; i < numAI; ++i)
        {
            auto mcts = std::make_shared<MCTS<GameTag>>(
                engine,
                static_cast<uint32_t>(maxNodes),
                static_cast<uint8_t>(historySize),
                static_cast<uint16_t>(maxDepth),
                cPUCT,
                virtualLoss,
                static_cast<uint16_t>(keepK)
            );
            mctsVec.push_back(std::move(mcts));
        }
    }

    void run(const YAML::Node& config) const override
    {
        std::cout << "Initializing " << m_gameName << "\n";

        auto engine = std::make_shared<EngineT>();
        engine->setup(config);

        size_t numPlayers = ITraits<GameTag>::kNumPlayers;

        if (!config["common"]["inference"]["numHumans"])
            throw std::runtime_error("Configuration missing 'common.inference.numHumans' field.");

        size_t numHumans = config["common"]["inference"]["numHumans"].as<size_t>();
        if (numHumans > numPlayers)
            throw std::runtime_error("Wrong configuration: numHumans > numPlayers");

        size_t numAI = numPlayers - numHumans;

        // Créer le neural net
        std::shared_ptr<NeuralNet<GameTag>> neuralNet = createNeuralNet(config);

        // Créer le pool de threads UNIQUE pour tous les MCTS
        std::shared_ptr<MCTSThreadPool<GameTag>> threadPool = createThreadPool(config, neuralNet);

        // Créer les instances MCTS
        AlignedVec<std::shared_ptr<MCTS<GameTag>>> mctsVec(reserve_only, numAI);
        createMCTS(config, mctsVec, engine, numAI);

        auto requester = std::make_unique<RequesterT>();
        requester->setup(config, engine);

        auto renderer = std::make_unique<RendererT>();
        renderer->setup(config, engine);

        auto handler = std::make_unique<HandlerT>();
        handler->setup(config, engine, std::move(mctsVec), std::move(threadPool),
            std::move(requester), std::move(renderer), numAI);

        handler->execute();
    }
};