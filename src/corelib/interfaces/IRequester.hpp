#pragma once
#include "../bootstrap/GameConfig.hpp"
#include "../model/GameTypes.hpp"

// ============================================================================
// IRequester.hpp — Action & State Request Interface
//
// A Requester is the bridge between the game loop and an external decision
// source: a human player (via stdin / GUI), a remote UCI/protocol peer, or any
// other non-AI agent.
//
// For AI agents the InferenceHandler drives the search tree directly and does
// not go through IRequester. IRequester is only needed when at least one player
// is human or externally controlled (numHumans > 0 in config).
//
// Implementors must override:
//   specificSetup()      — load requester-specific YAML fields
//   requestAction()      — block until the controlled player submits a move
//
// Optional overrides:
//   notifyAction()       — inform the requester of a move made by another player
//                          (useful to keep a remote GUI in sync)
//   notifyResult()       — inform the requester of the final game outcome
// ============================================================================

namespace Core
{
    template<ValidGameTraits GT>
    class IRequester
    {
    public:
        USING_GAME_TYPES(GT);

    protected:
        virtual void specificSetup(const YAML::Node& config) = 0;

    public:
        virtual ~IRequester() = default;

        // -------------------------------------------------------------------
        // SETUP
        // -------------------------------------------------------------------

        void setup(const YAML::Node& config)
        {
            specificSetup(config);
        }

        // -------------------------------------------------------------------
        // REQUIRED INTERFACE
        // -------------------------------------------------------------------

        virtual void requestInitialState(const uint32_t player, State& outState) const = 0;

        /// Blocks until the externally-controlled player provides a legal move.
        ///
        /// @param state    Current game state (read-only; used to display the
        ///                 board or validate keyboard input).
        /// @param player   Index of the player who must move.
        /// @param out      Filled with the chosen Action. Must be a legal move;
        ///                 the requester is responsible for validation loops.
        virtual Action requestAction(const State& state) const = 0;

        // -------------------------------------------------------------------
        // OPTIONAL NOTIFICATION HOOKS
        // These default to no-ops. Override in requesters that maintain a
        // remote view of the game (e.g. a UCI GUI or a network peer).
        // -------------------------------------------------------------------

        /// Called by the game loop after any player (including AI) makes a move.
        /// Allows the requester to forward the move to a remote UI or log file.
        virtual void notifyAction(const Action& action, uint32_t player) const {}

        /// Called by the game loop when the game ends.
        /// Allows the requester to display the result or send it to a remote peer.
        virtual void notifyResult(const std::span<const float> result) const {}
    };
}