#pragma once
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace Core
{
    // ========================================================================
    // RUNTIME ABSTRACTION
    // Erases the concrete game types, allowing the main engine binary to 
    // compile once and execute any registered game polymorphically.
    // ========================================================================
    struct IGameRunner
    {
        virtual ~IGameRunner() = default;

        // Contextual parameters injected directly via CLI to avoid mutating YAML
        virtual void run(const YAML::Node& config, const std::string& mode, const std::string& modelPath) const = 0;
    };

    // ========================================================================
    // REGISTRY
    // Thread-safe singleton holding references to all compiled game modules.
    // ========================================================================
    class GameRegistry
    {
    private:
        std::unordered_map<std::string, IGameRunner*> m_runners;

        GameRegistry() = default;

    public:
        static GameRegistry& instance()
        {
            static GameRegistry inst;
            return inst;
        }

        // Populated automatically prior to main() execution via AutoGameRegister
        void registerGame(const std::string& name, IGameRunner* runner)
        {
            if (m_runners.find(name) != m_runners.end()) {
                std::cerr << "Warning: Game '" << name << "' is already registered.\n";
                return;
            }
            m_runners[name] = runner;
        }

        IGameRunner* get(const std::string& name) const
        {
            auto it = m_runners.find(name);
            if (it == m_runners.end()) {
                throw std::runtime_error("Game registry error: No game found named '" + name + "'");
            }
            return it->second;
        }

        // Execution entry point invoked by main.cpp
        void run(const std::string& name, const YAML::Node& config, const std::string& mode, const std::string& modelPath) const
        {
            get(name)->run(config, mode, modelPath);
        }
    };
}