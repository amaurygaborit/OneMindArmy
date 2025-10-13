#pragma once
#include <iostream>
#include <memory>

#include "TypeResolver.hpp"
#include "../player/HumanPlayer.hpp"
#include "../player/AIPlayer.hpp"
#include "../interfaces/IHandler.hpp"

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

    // MCTS
    MCTS<GameTag> createMCTS(
        const YAML::Node& config,
        const std::shared_ptr<EngineT>& engine,
        const std::shared_ptr<NeuralNet<GameTag>>& neuralNet) const
    {
        if (!config["common"]["engine"]["numPlayers"])
            throw std::runtime_error("Configuration missing 'common.engine.numPlayers' field.");
        if (!config["common"]["model"]["maxNodes"])
            throw std::runtime_error("Configuration missing 'common.model.maxNodes' field.");
        if (!config["common"]["model"]["numBeliefSamples"])
            throw std::runtime_error("Configuration missing 'common.model.numBeliefSamples' field.");
        if (!config["common"]["model"]["cPUCT"])
            throw std::runtime_error("Configuration missing 'common.model.cPUCT' field.");
        if (!config["common"]["model"]["keepK"])
            throw std::runtime_error("Configuration missing 'common.model.keepK' field.");
        if (!config["common"]["model"]["maxDepth"])
            throw std::runtime_error("Configuration missing 'common.model.maxDepth' field.");

        uint8_t numPlayers = config["common"]["engine"]["numPlayers"].as<uint8_t>();
        if (numPlayers < 1)
            throw std::runtime_error("Wrong configuration: numPlayers < 1");

        uint16_t actionSpaceSize = config["common"]["model"]["actionSpaceSize"].as<uint16_t>();
        if (actionSpaceSize < 1)
            throw std::runtime_error("Wrong configuration: actionSpaceSize < 1");

        uint32_t maxNodes = config["common"]["model"]["maxNodes"].as<uint32_t>();
        if (maxNodes < 1)
            throw std::runtime_error("Wrong configuration: maxNodes < 1");
        
        uint8_t numBeliefSamples = config["common"]["model"]["numBeliefSamples"].as<uint8_t>();
        if (numBeliefSamples < 1)
            throw std::runtime_error("Wrong configuration: numBeliefSamples < 1");

        float cPUCT = config["common"]["model"]["cPUCT"].as<float>();
        if (cPUCT < 1)
            throw std::runtime_error("Wrong configuration: cPUCT < 1");

        uint16_t keepK = config["common"]["model"]["keepK"].as<uint16_t>();
        if (keepK < 1)
            throw std::runtime_error("Wrong configuration: keepK < 1");

        uint16_t maxDepth = config["common"]["model"]["maxDepth"].as<uint16_t>();
        if (maxDepth < 1)
            throw std::runtime_error("Wrong configuration: maxDepth < 1");

        return MCTS<GameTag>(engine, neuralNet, numPlayers, engine->getMaxValidActions(), actionSpaceSize, maxNodes, numBeliefSamples, cPUCT, keepK, maxDepth);
    }

    // Players
    AlignedVec<std::unique_ptr<IPlayer<GameTag>>> createPlayers(
        const YAML::Node& config,
        const std::shared_ptr<EngineT>& engine,
        const std::shared_ptr<RequesterT>& requester) const
    {
        AlignedVec<std::unique_ptr<IPlayer<GameTag>>> players;

        if (!config["common"]["engine"]["numPlayers"])
            throw std::runtime_error("Configuration missing 'common.engine.numPlayers' field.");
        if (!config["common"]["inference"]["numHumans"])
            throw std::runtime_error("Configuration missing 'common.inference.numHumans' field.");

        uint8_t numPlayers = config["common"]["engine"]["numPlayers"].as<uint8_t>();
        uint8_t numHumans = config["common"]["inference"]["numHumans"].as<uint8_t>();

        if (numHumans > numPlayers)
            throw std::runtime_error("Wrong configuration: numHumans > numPlayers");

        uint8_t numAI = numPlayers - numHumans;
        // Creates IA Players
        if (numAI > 0)
        {
			auto neuralNet = std::make_shared<NeuralNet<GameTag>>();
            for (int i = 0; i < numAI; ++i)
            {
                auto mcts = createMCTS(config, engine, neuralNet);
                players.push_back(std::make_unique<AIPlayer<GameTag>>(std::move(mcts)));
            }
        }

        // Creates Humans Players
        for (int i = 0; i < numHumans; ++i)
            players.push_back(std::make_unique<HumanPlayer<GameTag>>(requester));

        return players;
    }

    void run(const YAML::Node& config) const override
    {
        std::cout << "Initializing " << m_gameName << "\n";

        auto engine = std::make_shared<EngineT>();
        engine->setup(config);

        auto requester = std::make_shared<RequesterT>();
        requester->setup(config, engine);

		auto players = createPlayers(config, engine, requester);

        auto renderer = std::make_unique<RendererT>();
        renderer->setup(config, engine);

        auto handler = std::make_unique<HandlerT>();
        handler->setup(config, engine, std::move(players), std::move(renderer));

        handler->execute();
    }
};