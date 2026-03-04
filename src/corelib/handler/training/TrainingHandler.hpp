#pragma once
#include "../../interfaces/IHandler.hpp"
#include "PriorityReplayBuffer.hpp"

namespace Core
{
	template<ValidGameTraits GT>
	class TrainingHandler : public IHandler<GT>
	{
	public:
		USING_GAME_TYPES(GT);

	private:
		std::unique_ptr<PriorityReplayBuffer> m_replayBuffer;

	private:
		void specificSetup(const YAML::Node& config) override
		{
			std::cout << "Training handler setup called\n";

			if (!config["common"]["training"]["bufferSize"])
				throw std::runtime_error("Configuration missing 'common.training.bufferSize' field.");

			if (!config["common"]["training"]["trainingBatchSize"])
				throw std::runtime_error("Configuration missing 'common.training.trainingBatchSize' field.");

			uint32_t bufferSize = config["common"]["training"]["bufferSize"].as<uint32_t>();
			uint32_t batchSize = config["common"]["training"]["batchSize"].as<uint32_t>();

			m_replayBuffer = std::make_unique<PriorityReplayBuffer>(bufferSize, batchSize);
		};

	public:
		TrainingHandler() = default;
		virtual ~TrainingHandler() = default;

		void execute() override
		{

		};
	};
}