#pragma once
#include "../bootstrap/GameConfig.hpp"
#include "../model/GameTypes.hpp"

namespace Core
{
    // ============================================================================
    // DISPLAY INTERFACE
    // Handles state visualization and UI feedback.
    //
    // Design Intent:
    // Strictly read-only. Decouples the terminal/GUI rendering logic entirely 
    // from the core game state to prevent accidental mutations during drawing.
    // ============================================================================
    template<ValidGameTraits GT>
    class IRenderer
    {
    public:
        USING_GAME_TYPES(GT);

    protected:
        SessionConfig<GT> m_baseConfig;

        virtual void specificSetup(const YAML::Node& config) = 0;

    public:
        virtual ~IRenderer() = default;

        void setup(const YAML::Node& config,
            const SessionConfig<GT>& sessionConfig)
        {
            m_baseConfig = sessionConfig;
            specificSetup(config);
        }

        // --- REQUIRED ABSTRACTIONS ---

        virtual void renderState(const State& state) const = 0;
        virtual void renderResult(const GameResult& result) const = 0;

        // --- OPTIONAL ABSTRACTIONS ---
        // Provided as no-ops by default to simplify headless or minimal implementations.

        // Displays legal moves to assist human players.
        virtual void renderValidActions(const State& state, std::span<const Action> actionList) const {}

        // Echoes the selected action to the event log.
        virtual void renderActionPlayed(const Action& action, uint32_t player) const {}

        // Visualizes MCTS confidence, principal variations, or tree memory usage.
        virtual void renderThinking(std::span<const Action> pv,
            uint32_t nodeCount,
            uint32_t depthReached) const {
        }

        // Triggers a display refresh/wipe (e.g., ANSI terminal clear).
        virtual void clear() const {}
    };
}