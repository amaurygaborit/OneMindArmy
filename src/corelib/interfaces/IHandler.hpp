#pragma once
#include "../interfaces/IEngine.hpp"
#include "../interfaces/IRequester.hpp"
#include "../interfaces/IRenderer.hpp"
#include "../model/ThreadPool.hpp"

namespace Core
{
    // ============================================================================
    // SESSION ORCHESTRATOR INTERFACE
    // Drives the main execution loop (Self-Play, Match, or Protocol binding).
    //
    // Architecture:
    // Utilizes strict Dependency Injection via setup(). The handler takes ownership 
    // of all runtime components (Thread pools, UI) while keeping a shared reference 
    // to the stateless Engine.
    // ============================================================================
    template<ValidGameTraits GT>
    class IHandler
    {
    public:
        USING_GAME_TYPES(GT);

    protected:
        std::shared_ptr<IEngine<GT>>         m_engine;
        Vec<std::unique_ptr<TreeSearch<GT>>> m_treeSearch;
        std::unique_ptr<ThreadPool<GT>>      m_threadPool;
        std::unique_ptr<IRequester<GT>>      m_requester;
        std::unique_ptr<IRenderer<GT>>       m_renderer;
        SessionConfig<GT>                    m_sessionCfg;
        EngineConfig                         m_engineCfg;

        // Hook to load pipeline-specific configurations (e.g., export paths).
        virtual void specificSetup(const YAML::Node& config) = 0;

        // Safely checks if human/external interactions are enabled in this pipeline.
        [[nodiscard]] bool hasRequester() const noexcept { return m_requester != nullptr; }
        [[nodiscard]] bool hasRenderer()  const noexcept { return m_renderer != nullptr; }

    public:
        virtual ~IHandler() = default;

        // Injects all initialized dependencies to decouple component creation 
        // from the execution logic.
        void setup(const YAML::Node& config,
            std::shared_ptr<IEngine<GT>> engine,
            AlignedVec<std::unique_ptr<TreeSearch<GT>>>&& treeSearch,
            std::unique_ptr<ThreadPool<GT>>&& threadPool,
            std::unique_ptr<IRequester<GT>>&& requester,
            std::unique_ptr<IRenderer<GT>>&& renderer,
            const SessionConfig<GT>& sessionCfg,
            const EngineConfig& engineCfg)
        {
            m_engine = std::move(engine);
            m_treeSearch = std::move(treeSearch);
            m_threadPool = std::move(threadPool);
            m_requester = std::move(requester);
            m_renderer = std::move(renderer);

            m_sessionCfg = sessionCfg;
            m_engineCfg = engineCfg;

            specificSetup(config);
        }

        // Blocking call that executes the target pipeline until termination.
        virtual void execute() = 0;
    };
}