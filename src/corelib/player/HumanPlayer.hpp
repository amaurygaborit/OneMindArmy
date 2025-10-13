#pragma once
#include "IPlayer.hpp"
#include "../interfaces/IRequester.hpp"

template<typename GameTag>
class HumanPlayer : public IPlayer<GameTag>
{
private:
	std::shared_ptr<IRequester<GameTag>> m_requester;

public:
	HumanPlayer(std::shared_ptr<IRequester<GameTag>> requester)
		: m_requester(std::move(requester)) {};

	void chooseAction(const ObsStateT<GameTag>& obsState, ActionT<GameTag>& out) override
	{
		m_requester->requestAction(obsState, out);
	}
	void onActionPlayed(const ActionT<GameTag>& action) override
	{

	}
};