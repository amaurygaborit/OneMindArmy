#pragma once
#include "../../interfaces/IHandler.hpp"
#include "../../model/MCTS.hpp"
#include "PriorityReplayBuffer.hpp"

template<typename GameTag>
class TrainingHandler : public IHandler<GameTag>
{
private:
	std::unique_ptr<PriorityReplayBuffer> m_replayBuffer;

protected:
	void specificSetup(const YAML::Node& config) override
	{
		std::cout << "Training handler setup called\n";

		if (!config["common"]["training"]["bufferSize"])
			throw std::runtime_error("Configuration missing 'common.training.bufferSize' field.");

		if (!config["common"]["training"]["batchSize"])
			throw std::runtime_error("Configuration missing 'common.training.batchSize' field.");

		uint32_t bufferSize = config["common"]["training"]["bufferSize"].as<uint32_t>();
		uint32_t batchSize = config["common"]["training"]["batchSize"].as<uint32_t>();

		m_replayBuffer = std::make_unique<PriorityReplayBuffer>(bufferSize, batchSize);
	};

public:
	void execute() override
	{
		
	};
};