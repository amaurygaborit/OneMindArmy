#pragma once
#include "../interfaces/ITraits.hpp"

template<typename GameTag>
class IPlayer
{
public:
	virtual ~IPlayer() = default;

	virtual void chooseAction(const ObsStateT<GameTag>& obsState, ActionT<GameTag>& out) = 0;
	virtual void onActionPlayed(const ActionT<GameTag>& action) = 0;
};