#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <yaml-cpp/yaml.h>

#include "TypeResolver.hpp"
#include "../interfaces/IHandler.hpp"
#include "../model/ThreadPool.hpp"
#include "../model/TreeSearch.hpp"
#include "../model/NeuralNet.hpp"
#include "GameConfig.hpp" 

#include "../handlers/SelfPlayHandler.hpp"
#include "../handlers/InferenceHandler.hpp"
#include "../handlers/MetaExportHandler.hpp"

namespace Core
{
    // SFINAE check: Determines at compile-time if a specific game has implemented 
    // a custom execution handler overriding the default play/train pipelines.
    template <typename, typename = void>
    struct has_custom_handler : std::false_type {};

    template <typename T>
    struct has_custom_handler<T, std::void_t<typename T::Handler>> : std::true_type {};

    // ========================================================================
    // GAME BOOTSTRAPPER 
    // Acts as a Dependency Injection container. It resolves concrete types 
    // based on the templated GameConfig and wires the core architecture together.
    // ========================================================================
    template<typename GameConfig>
    class GameBootstrapper : public IGameRunner
    {
    private:
        std::string m_name;

    public:
        using GT = typename GameConfig::GameTypes;
        using EngineT = typename GameConfig::Engine;
        using RequesterT = typename GameConfig::Requester;
        using RendererT = typename GameConfig::Renderer;

        explicit GameBootstrapper(const std::string& name) : m_name(name) {}

        void run(const YAML::Node& config, const std::string& mode, const std::string& modelPath) const override
        {
            std::cout << "\n[Bootstrapper] Initializing Game Module: [" << m_name << "]\n";
            std::cout << "[Bootstrapper] Target Execution Mode: [" << mode << "]\n";

            NetworkConfig        networkConfig;  networkConfig.load(config, mode);
            EngineConfig         engineConfig;   engineConfig.load(config, mode);
            TrainingConfig       trainingConfig; trainingConfig.load(config, mode);
            BackendConfig        backendConfig;  backendConfig.load(config, mode);
            SessionConfig<GameConfig> sessionConfig;  sessionConfig.load(config, mode);

            // Dynamically route to the correct execution loop based on CLI arguments
            std::unique_ptr<IHandler<GT>> handler;

            if (mode == "train") {
                handler = std::make_unique<SelfPlayHandler<GT>>();
            }
            else if (mode == "play") {
                handler = std::make_unique<InferenceHandler<GT>>();
            }
            else if (mode == "export-meta") {
                handler = std::make_unique<MetaExportHandler<GT>>();
            }
            else if (mode == "custom") {
                if constexpr (has_custom_handler<GameConfig>::value) {
                    handler = std::make_unique<typename GameConfig::Handler>();
                }
                else {
                    throw std::runtime_error("Fatal Error: Run mode 'custom' requested, but no 'Handler' type defined.");
                }
            }
            else {
                throw std::invalid_argument("Fatal Error: Invalid run_mode '" + mode + "'. Valid options are: play, train, export-meta, custom.");
            }

            auto engine = std::make_shared<EngineT>();
            engine->setup(config);

            std::unique_ptr<ThreadPool<GT>> threadPool = nullptr;
            AlignedVec<std::unique_ptr<TreeSearch<GT>>> treeSearches;

            if (mode != "export-meta")
            {
                AlignedVec<std::unique_ptr<NeuralNet<GT>>> neuralNets;
                neuralNets.reserve(backendConfig.numGPUs);
                for (uint32_t i = 0; i < backendConfig.numGPUs; ++i) {
                    neuralNets.push_back(std::make_unique<NeuralNet<GT>>(i, backendConfig.inferenceBatchSize, modelPath));
                }

                threadPool = std::make_unique<ThreadPool<GT>>(
                    engine, std::move(neuralNets), backendConfig, engineConfig
                );

                // Exact tree allocation calculation to optimize VRAM footprint:
                // Self-Play: Allocates trees for both sides across all parallel games.
                // Inference: Only allocates trees for active AI participants.
                uint32_t actualNumAIs = (mode == "play" || mode == "custom") ? sessionConfig.numAIs : GT::kNumPlayers;
                uint32_t numTreesNeeded = actualNumAIs * backendConfig.numParallelGames;

                std::cout << "[Bootstrapper] Allocating " << numTreesNeeded << " MCTS Trees...\n";

                treeSearches.reserve(numTreesNeeded);
                for (uint32_t i = 0; i < numTreesNeeded; ++i) {
                    treeSearches.push_back(std::make_unique<TreeSearch<GT>>(engine, engineConfig));
                }
            }

            // Lazy initialization for I/O bounds: Keeps training completely headless
            std::unique_ptr<RequesterT> requester = nullptr;
            std::unique_ptr<RendererT> renderer = nullptr;

            if (mode == "play" || mode == "custom")
            {
                renderer = std::make_unique<RendererT>();
                renderer->setup(config, sessionConfig);

                requester = std::make_unique<RequesterT>();
                requester->setup(config);
            }

            // Inject all dependencies into the selected pipeline handler
            handler->setup(
                config, engine, std::move(treeSearches), std::move(threadPool),
                std::move(requester), std::move(renderer), sessionConfig, engineConfig
            );

            handler->execute();
        }
    };

    // Auto-registration mechanism utilizing static initialization
    template<typename GameConfig>
    struct AutoGameRegister
    {
        AutoGameRegister(const std::string& name)
        {
            static GameBootstrapper<GameConfig> bootstrapper(name);
            GameRegistry::instance().registerGame(name, &bootstrapper);
        }
    };
}