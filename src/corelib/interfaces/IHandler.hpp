#pragma once
#include "../interfaces/IEngine.hpp"
#include "../interfaces/IRequester.hpp"
#include "../interfaces/IRenderer.hpp"
#include "../model/MCTSThreadPool.hpp"

template<typename GameTag>
class IHandler
{
protected:
    using GT = ITraits<GameTag>;
    using ObsState = typename ObsStateT<GameTag>;
    using Action = typename ActionT<GameTag>;

    std::shared_ptr<IEngine<GameTag>> m_engine;
    AlignedVec<std::unique_ptr<MCTS<GameTag>>> m_mcts;
    std::unique_ptr<MCTSThreadPool<GameTag>> m_threadPool;
    std::unique_ptr<IRequester<GameTag>> m_requester;
    std::unique_ptr<IRenderer<GameTag>> m_renderer;

    SessionConfig<GameTag> m_baseConfig;

    virtual void specificSetup(const YAML::Node& config) = 0;

public:
    virtual ~IHandler() = default;

    void setup(const YAML::Node& config,
        std::shared_ptr<IEngine<GameTag>> engine,
        AlignedVec<std::unique_ptr<MCTS<GameTag>>>&& mcts,
        std::unique_ptr<MCTSThreadPool<GameTag>>&& threadPool,
        std::unique_ptr<IRequester<GameTag>>&& requester,
        std::unique_ptr<IRenderer<GameTag>>&& renderer,
        const SessionConfig<GameTag>& sessionConfig)
    {
        m_engine = std::move(engine);
        m_mcts = std::move(mcts);
        m_threadPool = std::move(threadPool);
        m_requester = std::move(requester);
        m_renderer = std::move(renderer);

		m_baseConfig = sessionConfig;

        specificSetup(config);
    }

    virtual void execute() = 0;
};