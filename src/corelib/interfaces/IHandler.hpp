#pragma once
#include "IEngine.hpp"
#include "../player/IPlayer.hpp"
#include "IRenderer.hpp"

template<typename GameTag>
class IHandler
{
protected:
	std::shared_ptr<IEngine<GameTag>> m_engine;
	AlignedVec<std::unique_ptr<IPlayer<GameTag>>> m_players;
	std::unique_ptr<IRenderer<GameTag>> m_renderer;

protected:
	virtual void specificSetup(const YAML::Node& config) = 0;

public:
	virtual ~IHandler() = default;
	void setup(const YAML::Node& config,
		std::shared_ptr<IEngine<GameTag>> engine,
		AlignedVec<std::unique_ptr<IPlayer<GameTag>>>&& players,
		std::unique_ptr<IRenderer<GameTag>> renderer)
	{
		m_engine = std::move(engine);
		m_players = std::move(players);
		m_renderer = std::move(renderer);

		specificSetup(config);
	};

	virtual void execute() = 0;
};