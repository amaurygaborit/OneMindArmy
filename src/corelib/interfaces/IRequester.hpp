#pragma once
#include "../bootstrap/GameConfig.hpp"
#include "../model/GameTypes.hpp"

namespace Core
{
    // ============================================================================
    // EXTERNAL INPUT INTERFACE
    // Bridges the autonomous game loop with external agents (Humans, GUIs, UCI).
    //
    // Design Intent:
    // Bypassed entirely during self-play. Only instantiated when the YAML config 
    // dictates external intervention, blocking the game thread until valid input 
    // is received.
    // ============================================================================
    template<ValidGameTraits GT>
    class IRequester
    {
    public:
        USING_GAME_TYPES(GT);

    protected:
        virtual void specificSetup(const YAML::Node& config) = 0;

    public:
        virtual ~IRequester() = default;

        void setup(const YAML::Node& config)
        {
            specificSetup(config);
        }

        // --- REQUIRED ABSTRACTIONS ---

        // Allows external agents to inject custom starting positions (e.g., FEN strings).
        virtual void requestInitialState(const uint32_t player, State& outState) const = 0;

        // Halts the game loop until the external agent provides a move.
        // The requester is responsible for parsing and returning a valid Action struct.
        virtual Action requestAction(const State& state) const = 0;

        // --- OPTIONAL NOTIFICATION HOOKS ---
        // Allows the requester to push engine events to remote endpoints (e.g., UCI).

        // Broadcasts moves made by the AI to keep external GUIs synchronized.
        virtual void notifyAction(const Action& action, uint32_t player) const {}

        // Broadcasts the final game outcome to external peers.
        virtual void notifyResult(const std::span<const float> result) const {}
    };
}