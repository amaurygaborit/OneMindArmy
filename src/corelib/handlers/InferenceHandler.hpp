#pragma once
#include <chrono>
#include <optional>
#include <vector>
#include <iostream>
#include <iomanip>

#include "../interfaces/IHandler.hpp"

namespace Core
{
    // ============================================================================
    // INFERENCE HANDLER (Match / Tournament / Human vs AI)
    //
    // Runs live games without generating training data.
    //
    // Move selection: always greedy (selectMove(0.0f) = most visited child).
    //   - No temperature : the AI always plays its best move.
    //   - No Gumbel noise: gumbelK should be 0 in the inference YAML config.
    //     Exploration noise during MCTS is only useful in self-play to
    //     diversify training data. In inference it hurts playing strength.
    // ============================================================================
    template<ValidGameTraits GT>
    class InferenceHandler : public IHandler<GT>
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        void specificSetup(const YAML::Node& /*config*/) override
        {
            std::cout << "[InferenceHandler] Setup complete.\n";
        }

    public:
        InferenceHandler() = default;
        virtual ~InferenceHandler() = default;

        void execute() override
        {
            State                 currentState;
            Action                selectedAction;
            std::vector<uint64_t> realHashHistory;
            realHashHistory.reserve(512);

            // ---- Initialise state ----
            if (this->m_sessionCfg.autoInitialState)
                this->m_engine->getInitialState(0, currentState);
            else
                this->m_requester->requestInitialState(0, currentState);

            realHashHistory.push_back(currentState.hash());

            // ---- Bootstrap MCTS trees ----
            for (uint32_t p = 0; p < this->m_sessionCfg.numAIs; ++p)
                this->m_treeSearch[p]->startSearch(currentState, realHashHistory);

            std::optional<GameResult> finalOutcome =
                this->m_engine->getGameResult(currentState, realHashHistory);

            uint32_t turnCount = 0;
            double   sumTimeMs = 0.0;

            if (this->m_sessionCfg.verbose)
                std::cout << "\n[Inference] Match started. AI count: "
                << this->m_sessionCfg.numAIs << "\n";

            // ================================================================
            // MAIN GAME LOOP
            // ================================================================
            while (!finalOutcome.has_value())
            {
                ++turnCount;
                const uint32_t currentPlayer =
                    this->m_engine->getCurrentPlayer(currentState);

                this->m_renderer->renderState(currentState);
                this->m_renderer->renderValidActions(currentState,
                    this->m_engine->getValidActions(currentState, realHashHistory));

                double turnTimeMs = 0.0;

                if (currentPlayer < this->m_sessionCfg.numAIs)
                {
                    // ---- AI turn ----
                    auto t0 = std::chrono::high_resolution_clock::now();

                    this->m_threadPool->executeTreeSearch(
                        this->m_treeSearch[currentPlayer].get(),
                        this->m_engineCfg.numSimulations);

                    // Temperature = 0.0f force le coup le plus exploré/robuste
                    selectedAction = this->m_treeSearch[currentPlayer]->selectMove(0.0f);

                    auto t1 = std::chrono::high_resolution_clock::now();
                    turnTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    sumTimeMs += turnTimeMs;
                }
                else
                {
                    // ---- Human / external turn ----
                    bool valid = false;
                    do {
                        selectedAction = this->m_requester->requestAction(currentState);
                        valid = this->m_engine->isValidAction(
                            currentState, realHashHistory, selectedAction);

                        if (!valid && this->m_sessionCfg.verbose)
                            std::cout << "[Warning] Invalid action. Please try again.\n";

                    } while (!valid);
                }

                // ---- Apply move ----
                this->m_engine->applyAction(selectedAction, currentState);
                realHashHistory.push_back(currentState.hash());

                // Mettre à jour l'arbre de *toutes* les IAs de la partie (pour garder leur cache MCTS en phase)
                for (uint32_t p = 0; p < this->m_sessionCfg.numAIs; ++p)
                    this->m_treeSearch[p]->advanceRoot(selectedAction, currentState);

                this->m_renderer->renderActionPlayed(selectedAction, currentPlayer);

                // ---- Performance log ----
                if (this->m_sessionCfg.verbose &&
                    currentPlayer < this->m_sessionCfg.numAIs)
                {
                    const double meanTime =
                        sumTimeMs / std::max(1.0, static_cast<double>(turnCount));
                    const float memUsage =
                        this->m_treeSearch[currentPlayer]->getMemoryUsage() * 100.0f;

                    std::cout << std::fixed << std::setprecision(2)
                        << "[AI-" << currentPlayer << "] "
                        << "Think: " << turnTimeMs << " ms | "
                        << "Avg: " << meanTime << " ms | "
                        << "Tree: " << memUsage << "%\n";
                }

                finalOutcome =
                    this->m_engine->getGameResult(currentState, realHashHistory);
            }

            // ================================================================
            // MATCH FINISHED
            // ================================================================
            this->m_renderer->renderState(currentState);
            this->m_renderer->renderResult(finalOutcome.value());

            if (this->m_sessionCfg.verbose)
                std::cout << "[Inference] Match finished in "
                << turnCount << " turns.\n";
        }
    };
}