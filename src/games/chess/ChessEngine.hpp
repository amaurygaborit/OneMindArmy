#pragma once
#include "../../corelib/interfaces/IEngine.hpp"
#include "MoveGenerator.hpp"
#include "FenParser.hpp"

class ChessEngine : public IEngine<ChessTag>
{
private:
    AlignedVec<ActionT<ChessTag>> m_validActionsBuf;

    bool isFiftyMoveRule(const ObsStateT<ChessTag>& obsState);
    bool isInsufficientMaterial(const ObsStateT<ChessTag>& obsState);

    bool ourKingInCheck(const ObsStateT<ChessTag>& obsState);

protected:
	void specificSetup(const YAML::Node& config) override;

public:
	ChessEngine();

    void getInitialState(ObsStateT<ChessTag>& out) override;
    uint8_t getCurrentPlayer(const ObsStateT<ChessTag>& obsState) override;
    void getValidActions(const ObsStateT<ChessTag>& obsState, AlignedVec<ActionT<ChessTag>>& out) override;
    bool isValidAction(const ObsStateT<ChessTag>& obsState, const ActionT<ChessTag>& action) override;
    void applyAction(const ActionT<ChessTag>& action, ObsStateT<ChessTag>& out) override;
    bool isTerminal(const ObsStateT<ChessTag>& obsState, AlignedVec<float>& out) override;

    void obsToIdx(const ObsStateT<ChessTag>& obsState, IdxStateT<ChessTag>& out) override;
    void idxToObs(const IdxStateT<ChessTag>& idxInput, ObsStateT<ChessTag>& out) override;

    void actionToIdx(const ActionT<ChessTag>& action, IdxActionT& out) override;
    void idxToAction(const IdxActionT& idxAction, ActionT<ChessTag>& out) override;
};