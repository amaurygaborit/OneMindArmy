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
    // INFERENCE HANDLER (Match / Tournament / Human Interaction)
    // Orchestrates live games. Uses the Neural Network purely for exploitation
    // and evaluation, without generating training data.
    // ============================================================================
    template<ValidGameTraits GT>
    class InferenceHandler : public IHandler<GT>
    {
    public:
        USING_GAME_TYPES(GT);

    private:
        void specificSetup(const YAML::Node& config) override
        {
            std::cout << "InferenceHandler setup called\n";
        }

    public:
        InferenceHandler() = default;
        virtual ~InferenceHandler() = default;

        void execute() override
        {
            State currentState;
            Action selectedAction;

            // 1. INITIALIZE REAL GAME STATE AND HISTORY
            std::vector<uint64_t> realHashHistory;
            realHashHistory.reserve(256);

            if (this->m_sessionCfg.autoInitialState) {
                this->m_engine->getInitialState(0, currentState);
            }
            else {
                this->m_requester->requestInitialState(0, currentState);
            }

            // Register the starting position's hash
            realHashHistory.push_back(currentState.hash());

            // 2. BOOTSTRAP THE MCTS TREES
            for (uint32_t p = 0; p < this->m_sessionCfg.numAIs; ++p) {
                this->m_treeSearch[p]->startSearch(currentState, realHashHistory);
            }

            std::optional<GameResult> finalOutcome = this->m_engine->getGameResult(currentState, realHashHistory);

            uint32_t turnCount = 0;
            double sumTimeMs = 0.0;

            if (this->m_sessionCfg.verbose) {
                std::cout << "\n[Inference] Match started. AI Count: " << this->m_sessionCfg.numAIs << "\n";
            }

            // 3. MAIN GAME LOOP
            while (!finalOutcome.has_value())
            {
                turnCount++;
                uint32_t currentPlayer = this->m_engine->getCurrentPlayer(currentState);

                // Render the current state to the console/UI
                this->m_renderer->renderState(currentState);
                this->m_renderer->renderValidActions(currentState,
                    this->m_engine->getValidActions(currentState, realHashHistory));

                double turnTimeMs = 0.0;

                // --- DECISION PHASE ---
                if (currentPlayer < this->m_sessionCfg.numAIs)
                {
                    auto t0 = std::chrono::high_resolution_clock::now();

                    // AI Turn: Trigger the asynchronous ThreadPool for a single tree
                    this->m_threadPool->executeTreeSearch(this->m_treeSearch[currentPlayer].get(),
                        this->m_engineCfg.numSimulations);

                    // Temperature scheduling (Slight exploration early, strict exploitation later)
                    float temp = (turnCount < this->m_engineCfg.temperatureDrop) ? this->m_sessionCfg.temperature : 0.01f;
                    selectedAction = this->m_treeSearch[currentPlayer]->selectMove(temp);

                    auto t1 = std::chrono::high_resolution_clock::now();
                    turnTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    sumTimeMs += turnTimeMs;
                }
                else
                {
                    // Human/External Turn: Block until a valid action is provided
                    bool valid = false;
                    do {
                        selectedAction = this->m_requester->requestAction(currentState);
                        valid = this->m_engine->isValidAction(currentState, realHashHistory, selectedAction);

                        if (!valid && this->m_sessionCfg.verbose) {
                            std::cout << "[Warning] Invalid action requested. Please try again.\n";
                        }
                    } while (!valid);
                }

                // --- REAL WORLD UPDATE ---
                this->m_engine->applyAction(selectedAction, currentState);
                realHashHistory.push_back(currentState.hash());

                // --- IMAGINARY WORLD UPDATE (Shift MCTS Root) ---
                for (uint32_t p = 0; p < this->m_sessionCfg.numAIs; ++p) {
                    this->m_treeSearch[p]->advanceRoot(selectedAction, currentState);
                }

                this->m_renderer->renderActionPlayed(selectedAction, currentPlayer);

                // --- PERFORMANCE LOGGING ---
                if (this->m_sessionCfg.verbose && currentPlayer < this->m_sessionCfg.numAIs)
                {
                    double meanTime = sumTimeMs / ((turnCount + 1) / 2.0); // Rough average for this AI
                    float memUsage = this->m_treeSearch[currentPlayer]->getMemoryUsage() * 100.0f;

                    std::cout << std::fixed << std::setprecision(2);
                    std::cout << "[AI-" << currentPlayer << "] "
                        << "Thought time: " << turnTimeMs << " ms | "
                        << "Avg: " << meanTime << " ms | "
                        << "Tree RAM: " << memUsage << " %\n";
                }

                // Check win/loss/draw conditions
                finalOutcome = this->m_engine->getGameResult(currentState, realHashHistory);
            }

            // 4. MATCH FINISHED
            this->m_renderer->renderState(currentState);
            this->m_renderer->renderResult(finalOutcome.value());

            if (this->m_sessionCfg.verbose) {
                std::cout << "[Inference] Match finished in " << turnCount << " turns.\n";
            }
        }
    };
}