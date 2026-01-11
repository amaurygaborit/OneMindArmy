#pragma once
#include "IEngine.hpp"

template<typename GameTag>
class IRenderer
{
protected:
	using GT = ITraits<GameTag>;
	using ObsState = typename ObsStateT<GameTag>;
	using Action = typename ActionT<GameTag>;
	using IdxState = typename IdxStateT<GameTag>;
	using IdxAction = typename IdxActionT<GameTag>;

	std::shared_ptr<IEngine<GameTag>> m_engine;

	RendererConfig m_baseConfig;

protected:
	virtual void specificSetup(const YAML::Node& config) = 0;

public:
	virtual ~IRenderer() = default;
	void setup(const YAML::Node& config,
		std::shared_ptr<IEngine<GameTag>> engine)
	{
		m_engine = std::move(engine);
		m_baseConfig.load(config);

		specificSetup(config);
	};

	virtual void renderState(const ObsState& obsState) const = 0;
	virtual void renderValidActions(const ObsState& obsState) const = 0;
	virtual void renderActionPlayed(const Action& action, const size_t player) const = 0;
	virtual void renderResult(const ObsState& obsState) const = 0;
};