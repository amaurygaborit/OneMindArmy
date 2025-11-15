#pragma once
#include "ChessTraits.hpp"
#include "../../corelib/interfaces/IRequester.hpp"

class ChessRequester : public IRequester<ChessTag>
{
private:
	void convertToAction(std::string& moveStr, Action& out) const;

protected:
	void specificSetup(const YAML::Node& config) override;

public:
	void requestInitialState(const size_t player, ObsState& out) const;
	void requestAction(const ObsState& obsState, Action& out) const override;
};