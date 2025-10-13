#pragma once
#include "ChessTraits.hpp"
#include "../../corelib/interfaces/IRequester.hpp"

class ChessRequester : public IRequester<ChessTag>
{
private:
	void convertToAction(std::string& moveStr, ActionT<ChessTag>& out) const;

protected:
	void specificSetup(const YAML::Node& config) override;

public:
	void requestAction(const ObsStateT<ChessTag>& obsState, ActionT<ChessTag>& out) const override;
};