#pragma once
#include "../bootstrap/GameConfig.hpp"
#include "../model/GameTypes.hpp"
#include "../util/PovUtils.hpp"
#include "../util/Zobrist.hpp"
#include <optional>

namespace Core
{
    // ============================================================================
    // GAME ENGINE INTERFACE
    // The single source of truth for game rules and mechanics.
    // 
    // Design Intent:
    // Strictly stateless. All methods are const-qualified and operate purely on 
    // the State passed as an argument. This guarantees thread safety during 
    // massive parallel MCTS traversal where thousands of threads mutate their 
    // own local states simultaneously.
    // ============================================================================
    template<ValidGameTraits GT>
    class IEngine
    {
    public:
        USING_GAME_TYPES(GT);

    protected:
        // Hook to load game-specific YAML parameters (e.g., max turns, board size)
        virtual void specificSetup(const YAML::Node& config) = 0;

    public:
        virtual ~IEngine() = default;

        void setup(const YAML::Node& config) { specificSetup(config); }

        // --- 1. STATE INITIALIZATION ---

        // Instantiates the starting board. Must clear outState internally to 
        // prevent leaking stale data from reused memory pools.
        virtual void getInitialState(uint32_t player, State& outState) const = 0;

        // --- 2. STATE QUERIES ---

        // Extracts the active player ID directly from the state's metadata.
        [[nodiscard]] virtual uint32_t getCurrentPlayer(const State& state) const = 0;

        // Generates the strict set of legal moves. Core bottleneck of the engine; 
        // must be heavily optimized (e.g., using bitboards).
        virtual ActionList getValidActions(const State& state, std::span<const uint64_t> hashHistory) const = 0;

        // Validates a specific move. Used primarily to sanitize external/human input.
        [[nodiscard]] virtual bool isValidAction(const State& state, std::span<const uint64_t> hashHistory, const Action& action) const = 0;

        // Evaluates terminal conditions (win/loss/draw/repetition). 
        // Returns nullopt if the game is ongoing, or the final score vector otherwise.
        [[nodiscard]] virtual std::optional<GameResult> getGameResult(const State& state, std::span<const uint64_t> hashHistory) const = 0;

        // Forces a terminal result based on a resignation trigger.
        [[nodiscard]] virtual GameResult buildResignResult(uint32_t losingPlayer) const = 0;

        // --- 3. STATE MUTATION ---

        // Applies a validated action, advancing the game state. 
        // Must update all relevant metadata (turn counters, Zobrist hashes, etc.).
        virtual void applyAction(const Action& action, State& outState) const = 0;

        // Rotates the state to the viewer's perspective (e.g., flipping the board).
        // Crucial for spatial invariances: ensures the Neural Network always evaluates 
        // positions from a canonical "bottom-up" perspective.
        virtual void changeStatePov(uint32_t viewer, State& outState) const = 0;
        virtual void changeActionPov(uint32_t viewer, Action& outAction) const = 0;

        // --- 4. TENSOR MAPPING ---

        // Defines the bijection between the engine's structured Action representation 
        // and the Neural Network's flat 1D policy output tensor.
        [[nodiscard]] virtual uint32_t actionToIdx(const Action& action) const = 0;
    };
}