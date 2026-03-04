#pragma once
#include "../bootstrap/GameConfig.hpp"
#include "../model/GameTypes.hpp"
#include "../util/PovUtils.hpp"
#include "../util/Zobrist.hpp"
#include <optional>

// ============================================================================
// IEngine.hpp — Game Logic Interface
//
// IEngine is the single source of truth for all game rules. It is stateless:
// every method receives the current State as a parameter and never mutates
// internal members (all pure-virtual methods are const-qualified).
//
// Implementors must override:
//   specificSetup()    — load game-specific YAML fields
//   getInitialState()  — set up the starting position
//   getCurrentPlayer() — read the active player from State metadata
//   getValidActions()  — enumerate legal moves from a given State
//   isValidAction()    — validate a single candidate action
//   isTerminal()       — detect end-of-game and produce final scores
//   applyAction()      — mutate a State by applying a legal move
//   changePOV()        — rotate a State to a specific player's point of view
//   idxToAction()      — decode a flat network output index into an Action
//   actionToIdx()      — encode an Action into a flat network output index
//   getHash()          — return the Zobrist hash of a State (see ZobristHash.hpp)
// ============================================================================

namespace Core
{
    template<ValidGameTraits GT>
    class IEngine
    {
    public:
        USING_GAME_TYPES(GT);

    protected:
        /// Override to load game-specific settings from the YAML config node.
        /// Called by the public setup() entry point.
        virtual void specificSetup(const YAML::Node& config) = 0;

    public:
        virtual ~IEngine() = default;

        // -------------------------------------------------------------------
        // SETUP
        // Loads configuration from the YAML node. The public entry point calls
        // the protected specificSetup() hook for game-specific parameters.
        // -------------------------------------------------------------------

        void setup(const YAML::Node& config) { specificSetup(config); }

        // -------------------------------------------------------------------
        // 1. STATE INITIALISATION
        // -------------------------------------------------------------------

        /// Fills [outState] with the game's starting position.
        ///
        /// @param player POV player index. Use this to orient asymmetric games
        ///                  (e.g. always place the requesting player's pieces at
        ///                  the bottom). Ignore for symmetric games like Chess.
        /// @param outState State to initialise. The engine must call outState.clear()
        ///                  before writing so that no stale data remains.
        virtual void getInitialState(uint32_t player, State& outState) const = 0;

        // -------------------------------------------------------------------
        // 2. STATE QUERIES — All const; never mutate the engine.
        // -------------------------------------------------------------------

        /// Returns the index of the player whose turn it is.
        /// Reads the relevant Meta slot (e.g. metas[TURN].value()).
        [[nodiscard]] virtual uint32_t getCurrentPlayer(const State& state) const = 0;

        /// Fills [outActions] with every legal Action available from [state].
        /// The engine clears [outActions] before filling; do not pre-fill it.
        ///
        /// In imperfect-information mode, generate moves for all elements whose
        /// location BitsetT has at least one bit set (probability > 0).
        virtual ActionList getValidActions(const State& state, std::span<const uint64_t> hashHistory) const = 0;

        /// Returns true if [action] is legal in [state].
        /// Used to validate moves from human input or external protocols (UCI).
        [[nodiscard]] virtual bool isValidAction(const State& state, std::span<const uint64_t> hashHistory, const Action& action) const = 0;

        /// Checks whether [state] is terminal and, if so, returns the outcome.
        ///
        /// @return  std::nullopt         — game is still ongoing.
        ///          GameResult with scores — game is over; scores[i] ∈ [−1, +1]
        ///          for player i (e.g. +1 = win, 0 = draw, −1 = loss).
        [[nodiscard]] virtual std::optional<GameResult> getGameResult(const State& state, std::span<const uint64_t> hashHistory) const = 0;
        
        // -------------------------------------------------------------------
        // 3. STATE MUTATION
        // -------------------------------------------------------------------

        /// Applies [action] to [outState], advancing the game by one half-move.
        ///
        /// Contract:
        ///   — [action] must be a legal move (validated by isValidAction).
        ///   — The engine updates ALL affected Facts and Meta slots, including
        ///     turn counter, castling rights, en passant, repetition count, etc.
        ///   — The Zobrist hash (if stored in State) must be updated here.
        virtual void applyAction(const Action& action, State& outState) const = 0;

        /// Rotates [outState] so that it is expressed from [player]'s point of view.
        ///
        /// In symmetric perfect-information games this is a board flip:
        /// White's pieces appear at the bottom regardless of which player is
        /// querying (canonical input for the neural network).
        ///
        /// The parameter is named [outState] to make clear that it is both the
        /// source and the destination; the transformation is applied in-place.
        virtual void changeStatePov(uint32_t viewer, State& outState) const = 0;
        virtual void changeActionPov(uint32_t viewer, Action& outAction) const = 0;

        // -------------------------------------------------------------------
        // 4. ACTION ↔ NETWORK INDEX CONVERSION
        //
        // The neural network outputs a flat policy vector of size kActionSpace.
        // These two methods define the bijection between that flat index space
        // and the structured Action representation.
        // -------------------------------------------------------------------

        /// Encodes an Action into a flat policy index ∈ [0, kActionSpace).
        [[nodiscard]] virtual uint32_t actionToIdx(const Action& action) const = 0;
    };
}