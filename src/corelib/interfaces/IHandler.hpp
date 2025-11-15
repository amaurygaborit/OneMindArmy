#pragma once
#include "../interfaces/IEngine.hpp"
#include "../interfaces/IRequester.hpp"
#include "../interfaces/IRenderer.hpp"

// Forward declarations
template<typename GameTag>
class MCTS;

template<typename GameTag>
class MCTSThreadPool;

template<typename GameTag>
class IHandler
{
protected:
    using GT = ITraits<GameTag>;
    using ObsState = typename ObsStateT<GameTag>;
    using Action = typename ActionT<GameTag>;

    std::shared_ptr<IEngine<GameTag>> m_engine;
    AlignedVec<std::shared_ptr<MCTS<GameTag>>> m_mcts;
    std::shared_ptr<MCTSThreadPool<GameTag>> m_threadPool;
    std::unique_ptr<IRequester<GameTag>> m_requester;
    std::unique_ptr<IRenderer<GameTag>> m_renderer;

    size_t m_numAI;
    bool m_autoInitialState;

    virtual void specificSetup(const YAML::Node& config) = 0;

public:
    virtual ~IHandler() = default;

    void setup(const YAML::Node& config,
        std::shared_ptr<IEngine<GameTag>> engine,
        AlignedVec<std::shared_ptr<MCTS<GameTag>>>&& mcts,
        std::shared_ptr<MCTSThreadPool<GameTag>> threadPool,
        std::unique_ptr<IRequester<GameTag>>&& requester,
        std::unique_ptr<IRenderer<GameTag>>&& renderer,
        size_t numAI)
    {
        m_engine = std::move(engine);
        m_mcts = std::move(mcts);
        m_threadPool = std::move(threadPool);
        m_requester = std::move(requester);
        m_renderer = std::move(renderer);
        m_numAI = numAI;

        if (!config["common"]["model"]["autoInitialState"])
            throw std::runtime_error("Configuration missing 'common.model.autoInitialState' field.");

        m_autoInitialState = config["common"]["model"]["autoInitialState"].as<bool>();

        specificSetup(config);
    }

    virtual void execute() = 0;
};