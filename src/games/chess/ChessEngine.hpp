#pragma once
#include "../../corelib/interfaces/IEngine.hpp"
#include "MoveGenerator.hpp"
#include "FenParser.hpp"

class ChessEngine : public IEngine<ChessTag>
{
private:
    bool isFiftyMoveRule(const ObsState& obsState) const;
    bool isInsufficientMaterial(const ObsState& obsState) const;

    bool ourKingInCheck(const ObsState& obsState) const;

protected:
	void specificSetup(const YAML::Node& config) override;

public:
	ChessEngine();

    void getInitialState(const size_t player, ObsState& out) const override;

    size_t getCurrentPlayer(const ObsState& obsState) const override;

    void getValidActions(const ObsState& obsState, AlignedVec<Action>& out) const override;
    bool isValidAction(const ObsState& obsState, const Action& action) const override;
    void applyAction(const Action& action, ObsState& out) const override;
    bool isTerminal(const ObsState& obsState, AlignedVec<float>& out) const override;

    void stateToFacts(const ObsState& obsState, FactState& out) const override;
    void actionToFact(const Action& action, const ObsState& obsState, FactAction& out) const override;

    void idxToAction(uint32_t idxAction, Action& out) const override;
    uint32_t actionToIdx(const Action& action) const override;
};