#pragma once
#include <iostream>
#include <memory>

#include "TypeResolver.hpp"
#include "../interfaces/IHandler.hpp"
#include "../model/ThreadPool.hpp"
#include "../model/TreeSearch.hpp"
#include "../model/NeuralNet.cuh"
#include "../bootstrap/GameConfig.hpp"

namespace Core
{
    // ========================================================================
    // 3. LE BOOTSTRAPPER (L'usine à jeux)
    // ========================================================================
    // Cette classe transforme une GameConfig (types) en un jeu qui tourne.
    template<typename GameConfig>
    class GameBootstrapper : public IGameRunner
    {
    private:
        std::string m_name;

    public:
        // Extraction des types depuis la Config (C'est là que c'est puissant !)
        using GT = typename GameConfig::GameTypes;
        using EngineT = typename GameConfig::Engine;
        using RequesterT = typename GameConfig::Requester;
        using RendererT = typename GameConfig::Renderer;

        // Pour Handler et Inference, on peut soit les prendre de la Config,
        // soit utiliser ceux par défaut du Core s'ils ne sont pas définis.
        // Ici, je suppose que tu as défini 'Handler' dans ta Config, ou on utilise le défaut.
        using HandlerT = typename GameConfig::Handler;

        explicit GameBootstrapper(const std::string& name) : m_name(name) {}

        void run(const YAML::Node& config) const override
        {
            std::cout << ">>> Bootstrapping Game: " << m_name << " <<<\n";

            // --- 1. CONFIGURATION ---
            SessionConfig<GT> sessionConfig; sessionConfig.load(config);
            TreeSearchConfig  tsConfig;      tsConfig.load(config);
            SystemConfig      sysConfig;     sysConfig.load(config);

            // --- 2. INSTANCIATION MOTEUR ---
            auto engine = std::make_shared<EngineT>();
            engine->setup(config);

            // --- 3. NEURAL NETS (Multi-GPU) ---
            AlignedVec<std::unique_ptr<NeuralNet<GT>>> neuralNets;
            neuralNets.reserve(sysConfig.numGPUs);
            for (int i = 0; i < sysConfig.numGPUs; ++i) {
                neuralNets.push_back(std::make_unique<NeuralNet<GT>>(i));
            }

            // --- 4. THREAD POOL ---
            auto threadPool = std::make_unique<ThreadPool<GT>>(
                engine, std::move(neuralNets), sysConfig, tsConfig
            );

            // --- 5. TREE SEARCH (IA Agents) ---
            AlignedVec<std::unique_ptr<TreeSearch<GT>>> treeSearches;
            treeSearches.reserve(sessionConfig.numAIs);
            for (uint8_t i = 0; i < sessionConfig.numAIs; ++i) {
                treeSearches.push_back(std::make_unique<TreeSearch<GT>>(engine, tsConfig));
            }

            // --- 6. IO (Requester / Renderer) ---
            auto requester = std::make_unique<RequesterT>();
            requester->setup(config);

            auto renderer = std::make_unique<RendererT>();
            renderer->setup(config);

            // --- 7. HANDLER (Boucle principale) ---
            auto handler = std::make_unique<HandlerT>();

            // Injection des dépendances
            handler->setup(
                config,
                engine,
                std::move(treeSearches),
                std::move(threadPool),
                std::move(requester),
                std::move(renderer),
                sessionConfig
            );

            // C'est parti !
            handler->execute();
        }
    };

    // ========================================================================
    // 4. L'AUTO-ENREGISTREUR (Le Helper statique)
    // ========================================================================
    // C'est la structure que tu utilises dans 'ChessRegistrar.cpp'.
    // Elle crée une instance statique du Bootstrapper et l'inscrit au registre.
    template<typename GameConfig>
    struct AutoGameRegister
    {
        AutoGameRegister(const std::string& name)
        {
            // Cette variable statique vit toute la durée du programme.
            // C'est elle qui contient la méthode run().
            static GameBootstrapper<GameConfig> bootstrapper(name);

            // On l'inscrit dans le singleton
            GameRegistry::instance().registerGame(name, &bootstrapper);
        }
    };
}