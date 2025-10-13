#include "PriorityReplayBuffer.hpp"

PriorityReplayBuffer::PriorityReplayBuffer(uint32_t bufferSize, uint32_t batchSize)
	: m_currentSize(0), m_kBufferSize(bufferSize), m_kBatchSize(batchSize)
{
	//buffer.reserve(bufferSize);
	//batch.reserve(batchSize);
}
/*
size_t PriorityReplayBuffer::getSize() const
{
	return currentSize;
}

void PriorityReplayBuffer::add(Experience& experience)
{
	if (currentSize < bufferSize)
	{
		buffer[currentSize] = experience;
		currentSize++;
	}
	else
	{
		size_t index = rand() % bufferSize;
		buffer[index] = experience;
	}

}

std::vector<Experience>& PriorityReplayBuffer::sampleBatch()
{
	batch.clear();
	for (size_t i = 0; i < batchSize; i++)
	{
		size_t index = rand() % currentSize;
		batch.push_back(buffer[index]);
	}
	return batch;

}
*/