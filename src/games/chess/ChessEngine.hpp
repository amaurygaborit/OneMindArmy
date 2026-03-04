#pragma once
#include "../../corelib/interfaces/IEngine.hpp"
#include "MoveGenerator.hpp"
#include "FenParser.hpp"

namespace Chess
{
    class ChessEngine : public Core::IEngine<ChessTypes>
    {
    public:
        USING_GAME_TYPES(ChessTypes);

    private:
        void stateToBB(const State& state, StateBB& out) const;

        bool isFiftyMoveRule(const State& state) const;
        bool isInsufficientMaterial(const State& state) const;

        bool ourKingInCheck(const State& state) const;

    protected:
        void specificSetup(const YAML::Node& config) override;

    public:
        ChessEngine();

        void getInitialState(const uint32_t player, State& outState) const override;
        uint32_t getCurrentPlayer(const State& state) const override;
        ActionList getValidActions(const State& state, std::span<const uint64_t> hashHistory) const override;
        bool isValidAction(const State& state, std::span<const uint64_t> hashHistory, const Action& action) const override;
        std::optional<GameResult> getGameResult(const State& state, std::span<const uint64_t> hashHistory) const override;

        void changeStatePov(uint32_t viewer, State& outState) const override;
        void changeActionPov(uint32_t viewer, Action& outAction) const override;

        void applyAction(const Action& action, State& outState) const override;

        uint32_t actionToIdx(const Action& action) const override;
    };
}