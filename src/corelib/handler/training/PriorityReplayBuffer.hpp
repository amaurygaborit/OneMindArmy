#pragma once
#include <unordered_map>

#include "../../AlignedVec.hpp"

struct Experience
{
    
};

class PriorityReplayBuffer
{
private:
    //std::unordered_map<size_t, Experience> buffer;
    //std::vector<Experience> batch;

    uint32_t m_currentSize;
    const uint32_t m_kBufferSize;
    const uint32_t m_kBatchSize;

public:
    PriorityReplayBuffer(uint32_t bufferSize, uint32_t batchSize);
    ~PriorityReplayBuffer() = default;

    //size_t getSize() const;

    //void add(Experience& experience);
    //std::vector<Experience>& sampleBatch();
};