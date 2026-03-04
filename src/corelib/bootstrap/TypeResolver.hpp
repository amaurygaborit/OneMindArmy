#pragma once
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <yaml-cpp/yaml.h>

namespace Core
{
    // ========================================================================
    // 1. L'INTERFACE (Contrat)
    // ========================================================================
    // Tout jeu doit savoir s'exécuter à partir d'une config YAML.
    struct IGameRunner
    {
        virtual ~IGameRunner() = default;
        virtual void run(const YAML::Node& config) const = 0;
    };

    // ========================================================================
    // 2. LE REGISTRE (Conteneur Singleton)
    // ========================================================================
    // Il stocke les pointeurs vers les IGameRunner.
    class GameRegistry
    {
    private:
        std::unordered_map<std::string, IGameRunner*> m_runners;

        GameRegistry() = default; // Privé pour Singleton

    public:
        // Singleton Thread-Safe (C++11 magic static)
        static GameRegistry& instance()
        {
            static GameRegistry inst;
            return inst;
        }

        // Enregistre un jeu (appelé automatiquement au démarrage)
        void registerGame(const std::string& name, IGameRunner* runner)
        {
            if (m_runners.find(name) != m_runners.end()) {
                std::cerr << "Warning: Game '" << name << "' is already registered.\n";
                return;
            }
            m_runners[name] = runner;
        }

        // Récupère un jeu
        IGameRunner* get(const std::string& name) const
        {
            auto it = m_runners.find(name);
            if (it == m_runners.end()) {
                throw std::runtime_error("Game registry error: No game found named '" + name + "'");
            }
            return it->second;
        }

        // Helper pour lancer directement
        void run(const std::string& name, const YAML::Node& config) const
        {
            get(name)->run(config);
        }
    };
}