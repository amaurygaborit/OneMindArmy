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

    void obsToIdx(const ObsState& obsState, IdxState& out) const override;
    void idxToObs(const IdxState& idxInput, ObsState& out) const override;

    void actionToIdx(const Action& action, IdxAction& out) const override;
    void idxToAction(const IdxAction& idxAction, Action& out) const override;
};