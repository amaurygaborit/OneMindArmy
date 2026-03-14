#pragma once
#include "../bootstrap/GameConfig.hpp"
#include "../model/GameTypes.hpp"

// ============================================================================
// IRenderer.hpp — Display Interface
//
// IRenderer handles all human-readable output: board visualisation, legal
// move listing, move echo, and final result display.
//
// The renderer is intentionally decoupled from the game loop: it only receives
// read-only snapshots (const State&, const Action&) and never alters game state.
//
// Implementors must override:
//   specificSetup()      — load renderer-specific YAML fields
//   renderState()        — display the current board
//   renderResult()       — display the final game outcome
//
// Optional overrides (default to no-ops):
//   renderValidActions() — list legal moves for the current player
//   renderActionPlayed() — echo the move that was just played
//   renderThinking()     — display MCTS statistics / principal variation
//   clear()              — clear the terminal / canvas before redrawing
// ============================================================================

namespace Core
{
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

        // -------------------------------------------------------------------
        // SETUP
        // -------------------------------------------------------------------

        void setup(const YAML::Node& config,
            const SessionConfig<GT>& sessionConfig)
        {
            m_baseConfig = sessionConfig;
            specificSetup(config);
        }

        // -------------------------------------------------------------------
        // REQUIRED INTERFACE
        // -------------------------------------------------------------------

        /// Renders the board / game state to the output medium (terminal, file…).
        virtual void renderState(const State& state) const = 0;

        /// Renders the final scores and outcome message.
        virtual void renderResult(const GameResult& result) const = 0;

        // -------------------------------------------------------------------
        // OPTIONAL INTERFACE — default implementations are no-ops.
        // -------------------------------------------------------------------

        /// Lists all legal moves for the current player (debug / tutorial aid).
        virtual void renderValidActions(const State& state, std::span<const Action> actionList) const {}

        /// Echoes the move that was just applied to the board.
        /// @param player  Index of the player who made the move.
        virtual void renderActionPlayed(const Action& action, uint32_t player) const {}

        /// Displays search diagnostics: depth, nodes visited, principal variation.
        /// Called at the end of each MCTS simulation batch if the config flag
        /// renderThinking is set.
        ///
        /// @param pv         Principal variation (sequence of best Actions).
        /// @param nodeCount  Total MCTS nodes evaluated.
        /// @param depthReached  Maximum depth explored in this batch.
        virtual void renderThinking(std::span<const Action> pv,
            uint32_t nodeCount,
            uint32_t depthReached) const {}

        /// Clears the display medium (e.g. ANSI escape to wipe the terminal).
        /// Called before renderState() when replaceRendering == true in config.
        virtual void clear() const {}
    };

}