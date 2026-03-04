#pragma once
#include "../interfaces/IEngine.hpp"
#include "../interfaces/IRequester.hpp"
#include "../interfaces/IRenderer.hpp"
#include "../model/ThreadPool.hpp"

// ============================================================================
// IHandler.hpp — Session Orchestration Interface
//
// IHandler is the top-level controller for a single game session. It owns all
// runtime components (engine, tree searches, thread pool, requester, renderer)
// and drives the match loop via execute().
//
// Concrete implementations:
//   InferenceHandler<GT>  — AI vs AI or AI vs Human (MCTS-PUCT / CFR-AVG).
//   UCIHandler<GT>        — Exposes the engine over the UCI protocol.
//   TrainingHandler<GT>   — Self-play loop that emits training samples.
//
// DEPENDENCY INJECTION
// ====================
// Instead of a telescoping 7-parameter setup() call, all runtime dependencies
// are bundled into a HandlerDeps<GT> struct. This makes call sites readable,
// allows partial construction, and simplifies unit testing (just fill the
// fields you need for the test).
//
// LIFETIME MODEL
// ==============
//   — m_engine is shared (const, read-only after setup) — multiple handlers
//     or tree-search threads can reference it safely.
//   — m_treeSearch, m_threadPool, m_requester, m_renderer are exclusively
//     owned by the handler (unique_ptr / Vec<unique_ptr>).
// ============================================================================

namespace Core
{
    // ========================================================================
    // IHandler<GT>
    // ========================================================================

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
        SessionConfig<GT>                    m_baseConfig;

        // Override to read handler-specific YAML fields (e.g. numSimulations,
        // temperature schedule, draw detection horizon…).
        virtual void specificSetup(const YAML::Node& config) = 0;

        // -------------------------------------------------------------------
        // PROTECTED HELPERS — Convenience methods for concrete handlers.
        // -------------------------------------------------------------------

        /// Returns true if a human-controlled requester is attached.
        [[nodiscard]] bool hasRequester() const noexcept { return m_requester != nullptr; }

        /// Returns true if a renderer is attached (non-headless mode).
        [[nodiscard]] bool hasRenderer()  const noexcept { return m_renderer != nullptr; }

    public:
        virtual ~IHandler() = default;

        // -------------------------------------------------------------------
        // SETUP
        //
        // Acquires all dependencies and calls the game-specific hook.
        // The YAML node provides runtime configuration (time control,
        // temperature, rendering flags…).
        // -------------------------------------------------------------------

        void setup(const YAML::Node& config,
            std::shared_ptr<IEngine<GT>> engine,
            AlignedVec<std::unique_ptr<TreeSearch<GT>>>&& treeSearch,
            std::unique_ptr<ThreadPool<GT>>&& threadPool,
            std::unique_ptr<IRequester<GT>>&& requester,
            std::unique_ptr<IRenderer<GT>>&& renderer,
            const SessionConfig<GT>& sessionConfig)
        {
            m_engine = std::move(engine);
            m_treeSearch = std::move(treeSearch);
            m_threadPool = std::move(threadPool);
            m_requester = std::move(requester);
            m_renderer = std::move(renderer);

            m_baseConfig = sessionConfig;

            specificSetup(config);
        }

        // -------------------------------------------------------------------
        // EXECUTE — Runs the full session (match, self-play loop, UCI loop…).
        //
        // Blocks until the session is complete. Thread-safety is the
        // responsibility of the concrete implementation.
        // -------------------------------------------------------------------

        virtual void execute() = 0;
    };
}