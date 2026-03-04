#pragma once
#include "../../corelib/interfaces/IRequester.hpp"
#include "ChessTypes.hpp"

namespace Chess
{
	class ChessRequester : public Core::IRequester<ChessTypes>
	{
	public:
		USING_GAME_TYPES(ChessTypes);

	private:
		Action convertToAction(const std::string& moveStr, const State state) const;

	protected:
		void specificSetup(const YAML::Node& config) override;

	public:
		void requestInitialState(const uint32_t player, State& outState) const override;
        Action requestAction(const State& state) const override;
	};
}