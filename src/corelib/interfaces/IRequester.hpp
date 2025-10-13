#pragma once
#include "IEngine.hpp"

template<typename GameTag>
class IRequester
{
protected:
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

	virtual void requestAction(const ObsStateT<GameTag>& obsState, ActionT<GameTag>& out) const = 0;
};