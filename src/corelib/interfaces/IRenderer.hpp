#pragma once
#include "IEngine.hpp"

template<typename GameTag>
class IRenderer
{
protected:
	std::shared_ptr<IEngine<GameTag>> m_engine;

	bool m_isRenderState = false;
	bool m_isRenderValidActions = false;
	bool m_isRenderActionPlayed = false;
	bool m_isRenderResult = false;

protected:
	virtual void specificSetup(const YAML::Node& config) = 0;

public:
	virtual ~IRenderer() = default;
	void setup(const YAML::Node& config,
		std::shared_ptr<IEngine<GameTag>> engine)
	{
		m_engine = std::move(engine);

		if (!config["common"]["renderer"]["renderState"])
			throw std::runtime_error("Configuration missing 'common.renderer.renderState' field");

		if (!config["common"]["renderer"]["renderValidActions"])
			throw std::runtime_error("Configuration missing 'common.renderer.renderValidActions' field");

		if (!config["common"]["renderer"]["renderActionPlayed"])
			throw std::runtime_error("Configuration missing 'common.renderer.renderActionPlayed' field");

		if (!config["common"]["renderer"]["renderResult"])
			throw std::runtime_error("Configuration missing 'common.renderer.renderResult' field");

		m_isRenderState = config["common"]["renderer"]["renderState"].as<bool>();
		m_isRenderValidActions = config["common"]["renderer"]["renderValidActions"].as<bool>();
		m_isRenderActionPlayed = config["common"]["renderer"]["renderActionPlayed"].as<bool>();
		m_isRenderResult = config["common"]["renderer"]["renderResult"].as<bool>();

		specificSetup(config);
	};

	virtual void renderState(const ObsStateT<GameTag>& obsState) const = 0;
	virtual void renderValidActions(const ObsStateT<GameTag>& obsState) const = 0;
	virtual void renderActionPlayed(const ActionT<GameTag>& action, const size_t idPlayer) const = 0;
	virtual void renderResult(const ObsStateT<GameTag>& obsState) const = 0;
};