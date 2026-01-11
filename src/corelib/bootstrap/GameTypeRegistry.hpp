#pragma once
#include <iostream>
#include <memory>

#include "TypeResolver.hpp"
#include "../interfaces/IHandler.hpp"
#include "../model/MCTSThreadPool.hpp"
#include "../bootstrap/GameConfig.hpp"

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

    void run(const YAML::Node& config) const override
    {
        std::cout << "Initializing " << m_gameName << "\n";

        // 1. Chargement des configurations
        // Session (Joueurs, Temps, Renderer settings)
        SessionConfig<GameTag> sessionConfig;
        sessionConfig.load(config);

        // Moteur (MCTS, Arbre, Algo)
        MCTSConfig mctsConfig;
        mctsConfig.load(config);

        // Système (Threads, GPU, Batch)
        SystemConfig sysConfig;
        sysConfig.load(config);

        // 2. Initialisation Engine
        auto engine = std::make_shared<EngineT>();
        engine->setup(config);

        // 3. Création des Réseaux de Neurones (sur chaque GPU demandé)
        AlignedVec<std::unique_ptr<NeuralNet<GameTag>>> neuralNets(reserve_only, sysConfig.numGPUs);
        for (int i = 0; i < sysConfig.numGPUs; ++i)
        {
            auto net = std::make_unique<NeuralNet<GameTag>>(i);
            neuralNets.push_back(std::move(net));
        }

        // 4. Création du ThreadPool Global
        // Il prend SystemConfig pour les threads/batchs et MCTSConfig pour initialiser les NodeEvents
        std::unique_ptr<MCTSThreadPool<GameTag>> threadPool =
            std::make_unique<MCTSThreadPool<GameTag>>(engine, std::move(neuralNets), sysConfig, mctsConfig);

        // 5. Création des instances MCTS (une par IA)
        // Elles prennent MCTSConfig pour savoir comment gérer l'arbre (taille, reuse...)
        AlignedVec<std::unique_ptr<MCTS<GameTag>>> mctsVec(reserve_only, sessionConfig.numAIs);
        for (uint8_t i = 0; i < sessionConfig.numAIs; ++i)
        {
            auto mcts = std::make_unique<MCTS<GameTag>>(engine, mctsConfig);
            mctsVec.push_back(std::move(mcts));
        }

        // 6. Setup Requester & Renderer
        auto requester = std::make_unique<RequesterT>();
        requester->setup(config, engine);

        auto renderer = std::make_unique<RendererT>();
        renderer->setup(config, engine);

        // 7. Lancement du Handler (Boucle de jeu)
        // Il prend SessionConfig pour gérer le déroulement (nb simulations, température...)
        auto handler = std::make_unique<HandlerT>();
        handler->setup(config, engine, std::move(mctsVec), std::move(threadPool),
            std::move(requester), std::move(renderer), sessionConfig);

        handler->execute();
    }
};