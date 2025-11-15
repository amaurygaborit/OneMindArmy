#pragma once
#include "../../interfaces/IHandler.hpp"
#include "../../model/MCTS.hpp"
#include "../../model/MCTSThreadPool.hpp"

template<typename GameTag>
class InferenceHandler : public IHandler<GameTag>
{
protected:
    using GT = typename IHandler<GameTag>::GT;
    using ObsState = typename IHandler<GameTag>::ObsState;
    using Action = typename IHandler<GameTag>::Action;

    void specificSetup(const YAML::Node& config) override {
        std::cout << "Inference handler initialized" << std::endl;
    }

public:
    void execute() override {
        ObsState currentState;
        AlignedVec<float> values(GT::kNumPlayers);
        bool isTerminal = false;

        // Initialize all MCTS instances with starting state
        std::cout << "Initializing game..." << std::endl;
        for (size_t p = 0; p < this->m_numAI; ++p) {
            if (this->m_autoInitialState) {
                this->m_engine->getInitialState(p, currentState);
            }
            else {
                this->m_requester->requestInitialState(p, currentState);
            }

            try {
                this->m_mcts[p]->startSearch(currentState);
            }
            catch (const std::exception& e) {
                std::cerr << "ERROR initializing MCTS " << p << ": " << e.what() << std::endl;
                throw;
            }
        }

        // Game loop
        int moveCount = 0;
        while (!isTerminal) {
            moveCount++;
            std::cout << "\n=== Move " << moveCount << " ===" << std::endl;

            this->m_renderer->renderState(currentState);
            this->m_renderer->renderValidActions(currentState);

            size_t currentPlayer = this->m_engine->getCurrentPlayer(currentState);
            Action selectedAction;

            if (currentPlayer < this->m_numAI) {
                std::cout << "AI Player " << currentPlayer << " thinking..." << std::endl;

                try {
                    // CRITICAL FIX: Ensure workers are idle BEFORE starting search
                    if (!this->m_threadPool->waitForIdle(10000)) {
                        std::cerr << "ERROR: Workers not idle before search!" << std::endl;
                        throw std::runtime_error("Thread pool synchronization failed before search");
                    }

                    // Execute MCTS
                    this->m_mcts[currentPlayer]->run(10000, *this->m_threadPool);

                    // CRITICAL FIX: Wait for workers to be idle AFTER search
                    if (!this->m_threadPool->waitForIdle(10000)) {
                        std::cerr << "ERROR: Workers did not become idle after search!" << std::endl;
                        throw std::runtime_error("Thread pool synchronization failed after search");
                    }

                    // CRITICAL FIX: Clear worker caches AFTER ensuring they're idle
                    this->m_threadPool->clearWorkerCaches();

                    // Get best action (now safe)
                    this->m_mcts[currentPlayer]->bestActionFromRoot(selectedAction);
                }
                catch (const std::exception& e) {
                    std::cerr << "ERROR during MCTS search for player "
                        << currentPlayer << ": " << e.what() << std::endl;
                    throw;
                }
            }
            else {
                std::cout << "Human Player " << currentPlayer << " turn" << std::endl;
                this->m_requester->requestAction(currentState, selectedAction);
            }

            // CRITICAL FIX: Ensure workers are COMPLETELY idle before rerooting
            std::cout << "Waiting for complete synchronization before reroot..." << std::endl;
            if (!this->m_threadPool->waitForIdle(10000)) {
                std::cerr << "ERROR: Workers not idle before reroot!" << std::endl;
                throw std::runtime_error("Thread pool not idle before reroot");
            }

            // Apply action to game state BEFORE rerooting
            this->m_engine->applyAction(selectedAction, currentState);
            this->m_renderer->renderActionPlayed(selectedAction, currentPlayer);

            // Update all MCTS trees with played action
            std::cout << "Updating MCTS trees..." << std::endl;
            for (size_t p = 0; p < this->m_numAI; ++p) {
                try {
                    this->m_mcts[p]->rerootByPlayedAction(selectedAction);
                }
                catch (const std::exception& e) {
                    std::cerr << "WARNING: Error rerooting MCTS " << p << ": " << e.what() << std::endl;

                    // If error is for current player, it's critical
                    if (p == currentPlayer) {
                        throw;
                    }
                    // Otherwise, tree will restart from scratch
                }
            }

            // Check terminal condition
            values.clear();
            values.resize(GT::kNumPlayers);
            isTerminal = this->m_engine->isTerminal(currentState, values);
        }

        // Game over
        std::cout << "\n=== Game Over ===" << std::endl;
        this->m_renderer->renderState(currentState);
        this->m_renderer->renderResult(currentState);
    }
};