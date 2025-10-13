#pragma once
#include "IPlayer.hpp"
#include "../model/MCTS.hpp"

template<typename GameTag>
class AIPlayer : public IPlayer<GameTag>
{
private:
	MCTS<GameTag> m_mcts;
	bool m_firstMove = true;

public:
	AIPlayer(MCTS<GameTag>&& mcts)
		: m_mcts(std::move(mcts)) {};

	void chooseAction(const ObsStateT<GameTag>& obsState, ActionT<GameTag>& out) override
	{
		if (m_firstMove)
		{
			m_mcts.startSearch(obsState);
			m_firstMove = false;
		}
		m_mcts.run(1000);
		out = m_mcts.bestActionFromRoot();
		
	}
	void onActionPlayed(const ActionT<GameTag>& action) override
	{
		m_mcts.rerootByPlayedAction(action);
	}
};