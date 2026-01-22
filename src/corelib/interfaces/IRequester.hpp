#pragma once
#include "IEngine.hpp"

template<typename GameTag>
class IRequester
{
protected:
	using GT = ITraits<GameTag>;
	using ObsState = typename GT::ObsState;
	using Action = typename GT::Action;

	std::shared_ptr<IEngine<GameTag>> m_engine;

protected:
	virtual void specificSetup(const YAML::Node& config) = 0;

public:
	virtual ~IRequester() = default;
	void setup(const YAML::Node& config,
		std::shared_ptr<IEngine<GameTag>> engine)
	{
		m_engine = std::move(engine);
		specificSetup(config);
	};
	
	virtual void requestInitialState(const size_t player, ObsState& out) const = 0;
	virtual void requestAction(const ObsState& obsState, Action& out) const = 0;
};