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

    void specificSetup(const YAML::Node& config) override
    {
    }

public:
    void execute() override
    {
        ObsState currentState;
        Action selectedAction;

        // Init Game
        if (this->m_baseConfig.autoInitialState)
        {
            this->m_engine->getInitialState(0, currentState);
        }
        else {
            this->m_requester->requestInitialState(0, currentState);
        }

        // Init MCTS trees
        for (size_t p = 0; p < this->m_baseConfig.numAIs; ++p)
        {
            this->m_mcts[p]->startSearch(currentState);
        }

        bool isTerminal = false;
        AlignedVec<float> values(GT::kNumPlayers);

        std::cout << "\033[s";
        while (!isTerminal) {
            size_t player = this->m_engine->getCurrentPlayer(currentState);

            this->m_renderer->renderState(currentState);
            this->m_renderer->renderValidActions(currentState);

            auto t0 = std::chrono::high_resolution_clock::now();

            if (player < this->m_baseConfig.numAIs)
            {
                // AI Turn
                this->m_threadPool->executeMCTS(this->m_mcts[player].get(), this->m_baseConfig.numSimulations);
                this->m_mcts[player]->selectMove(this->m_baseConfig.temperature, selectedAction);
            }
            else
            {
                // Human Turn
                this->m_requester->requestAction(currentState, selectedAction);
            }
            // Apply Move
            this->m_engine->applyAction(selectedAction, currentState);

            auto t1 = std::chrono::high_resolution_clock::now();

            this->m_renderer->renderActionPlayed(selectedAction, player);

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "Time: " << ms << " ms" << std::endl;

            for (size_t p = 0; p < this->m_baseConfig.numAIs; ++p)
            {
                this->m_mcts[p]->startSearch(currentState);
            }

            isTerminal = this->m_engine->isTerminal(currentState, values);
        }
        // End the Game
        this->m_renderer->renderState(currentState);
        this->m_renderer->renderResult(currentState);
    }
};