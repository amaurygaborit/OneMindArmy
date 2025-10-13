#pragma once
#include "../../interfaces/IHandler.hpp"

template<typename GameTag>
class InferenceHandler : public IHandler<GameTag>
{
protected:
	void specificSetup(const YAML::Node& config) override
    {
        std::cout << "Inference handler setup called\n";
    }

public:
    void execute() override
    {
		const uint8_t kNumPlayers = this->m_players.size();
        bool isTerminal = false;
        
        ObsStateT<GameTag> obsState;
		ActionT<GameTag> action;

        this->m_engine->getInitialState(obsState);
        AlignedVec<float> values(kNumPlayers);

        while (!isTerminal)
        {
            this->m_renderer->renderState(obsState);
            this->m_renderer->renderValidActions(obsState);

            uint8_t currentPlayer = this->m_engine->getCurrentPlayer(obsState);
			this->m_players[currentPlayer]->chooseAction(obsState, action);
            for (uint8_t p = 0; p < kNumPlayers; ++p)
                this->m_players[p]->onActionPlayed(action);
            this->m_engine->applyAction(action, obsState);

            this->m_renderer->renderActionPlayed(action, currentPlayer);

            values.reset();
            isTerminal = this->m_engine->isTerminal(obsState, values);
        }
        this->m_renderer->renderState(obsState);
		this->m_renderer->renderResult(obsState);
    }
};