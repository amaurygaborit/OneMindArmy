#pragma once
#include "../model/GameTypes.hpp"
#include "../bootstrap/GameConfig.hpp"
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cassert>

// Headers CUDA & TensorRT
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <NvInfer.h>

namespace Core
{
#define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                          << cudaGetErrorString(err) << std::endl; \
                std::terminate(); \
            } \
        } while (0)

    class TRTLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cerr << "[TensorRT] " << msg << std::endl;
            }
        }
    };

    // Le 'inline' est obligatoire en C++17/20 pour les variables statiques globales dans les headers !
    inline static TRTLogger g_logger;

    // ========================================================================
    // KERNELS CUDA
    // ========================================================================
    namespace Kernels
    {
        template<typename GT>
        __global__ void tokenizeStatesKernel(
            const Core::State<GT>* d_states,
            float* d_nnInputBuffer,
            uint32_t batchSize,
            uint32_t tensorSizePerState)
        {
            uint32_t batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (batchIdx >= batchSize) return;

            float* myTensor = d_nnInputBuffer + (batchIdx * tensorSizePerState);
            const auto& state = d_states[batchIdx];

            uint32_t offset = 0;
            for (uint32_t i = 0; i < GT::kMaxElems; ++i) {
                const auto& fact = state.getElem(i);
                if (fact.exists()) {
                    myTensor[offset++] = static_cast<float>(fact.factId());
                    myTensor[offset++] = static_cast<float>(fact.ownerId());
                    myTensor[offset++] = static_cast<float>(fact.pos());
                }
                else {
                    myTensor[offset++] = -1.0f;
                    myTensor[offset++] = -1.0f;
                    myTensor[offset++] = -1.0f;
                }
            }
        }
    }

    // ========================================================================
    // STRUCTURES ET CLASSE NEURALNET
    // ========================================================================
    template<typename GT>
    struct ModelResultsT
    {
        USING_GAME_TYPES(GT);
        std::array<float, Defs::kNumPlayers> values{};
        std::array<float, Defs::kActionSpace> policy{};
        ModelResultsT() noexcept = default;
    };

    template<typename GT>
    class NeuralNet
    {
    private:
        USING_GAME_TYPES(GT);
        using ModelResults = ModelResultsT<GT>;

        int m_deviceId;
        uint32_t m_maxBatchSize;
        cudaStream_t m_stream;

        nvinfer1::IRuntime* m_runtime = nullptr;
        nvinfer1::ICudaEngine* m_engine = nullptr;
        nvinfer1::IExecutionContext* m_context = nullptr;

        State* d_statesBuffer = nullptr;
        float* d_nnInputBuffer = nullptr;
        float* d_nnOutputValues = nullptr;
        float* d_nnOutputPolicy = nullptr;
        float* h_outputValues = nullptr;
        float* h_outputPolicy = nullptr;

        void loadTensorRTEngine(const std::string& enginePath)
        {
            std::ifstream file(enginePath, std::ios::binary);
            if (!file.good()) throw std::runtime_error("Erreur lecture engine TensorRT.");

            file.seekg(0, file.end);
            size_t size = file.tellg();
            file.seekg(0, file.beg);

            std::vector<char> trtModelStream(size);
            file.read(trtModelStream.data(), size);
            file.close();

            m_runtime = nvinfer1::createInferRuntime(g_logger);
            m_engine = m_runtime->deserializeCudaEngine(trtModelStream.data(), size);
            m_context = m_engine->createExecutionContext();
        }

    public:
        NeuralNet(int deviceId, uint32_t maxBatchSize, const std::string& enginePath)
            : m_deviceId(deviceId), m_maxBatchSize(maxBatchSize)
        {
            CUDA_CHECK(cudaSetDevice(m_deviceId));
            CUDA_CHECK(cudaStreamCreate(&m_stream));

            loadTensorRTEngine(enginePath);

            constexpr uint32_t kTensorSizePerState = GT::kMaxElems * 3;

            CUDA_CHECK(cudaMalloc(&d_statesBuffer, maxBatchSize * sizeof(State)));
            CUDA_CHECK(cudaMalloc(&d_nnInputBuffer, maxBatchSize * kTensorSizePerState * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_nnOutputValues, maxBatchSize * Defs::kNumPlayers * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_nnOutputPolicy, maxBatchSize * Defs::kActionSpace * sizeof(float)));

            CUDA_CHECK(cudaMallocHost(&h_outputValues, maxBatchSize * Defs::kNumPlayers * sizeof(float)));
            CUDA_CHECK(cudaMallocHost(&h_outputPolicy, maxBatchSize * Defs::kActionSpace * sizeof(float)));

            m_context->setTensorAddress("input_state", d_nnInputBuffer);
            m_context->setTensorAddress("output_value", d_nnOutputValues);
            m_context->setTensorAddress("output_policy", d_nnOutputPolicy);
        }

        ~NeuralNet()
        {
            CUDA_CHECK(cudaSetDevice(m_deviceId));
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
            CUDA_CHECK(cudaStreamDestroy(m_stream));

            cudaFree(d_statesBuffer);
            cudaFree(d_nnInputBuffer);
            cudaFree(d_nnOutputValues);
            cudaFree(d_nnOutputPolicy);
            cudaFreeHost(h_outputValues);
            cudaFreeHost(h_outputPolicy);

            if (m_context) delete m_context;
            if (m_engine) delete m_engine;
            if (m_runtime) delete m_runtime;
        }

        void forwardBatch(const AlignedVec<State>& states,
            const AlignedVec<AlignedVec<Action>>& histories,
            AlignedVec<ModelResults>& results)
        {
            uint32_t batchSize = states.size();
            if (batchSize == 0) return;

            constexpr uint32_t kTensorSizePerState = GT::kMaxElems * 3;
            if (results.size() < batchSize) results.resize(batchSize);

            CUDA_CHECK(cudaSetDevice(m_deviceId));

            CUDA_CHECK(cudaMemcpyAsync(
                d_statesBuffer, states.data(), batchSize * sizeof(State),
                cudaMemcpyHostToDevice, m_stream
            ));

            int threadsPerBlock = 256;
            int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
            Kernels::tokenizeStatesKernel<GT> << <blocksPerGrid, threadsPerBlock, 0, m_stream >> > (
                d_statesBuffer, d_nnInputBuffer, batchSize, kTensorSizePerState
                );

            m_context->enqueueV3(m_stream);

            CUDA_CHECK(cudaMemcpyAsync(
                h_outputValues, d_nnOutputValues, batchSize * Defs::kNumPlayers * sizeof(float),
                cudaMemcpyDeviceToHost, m_stream
            ));

            CUDA_CHECK(cudaMemcpyAsync(
                h_outputPolicy, d_nnOutputPolicy, batchSize * Defs::kActionSpace * sizeof(float),
                cudaMemcpyDeviceToHost, m_stream
            ));

            CUDA_CHECK(cudaStreamSynchronize(m_stream));

            for (size_t b = 0; b < batchSize; ++b) {
                auto& res = results[b];
                for (size_t p = 0; p < Defs::kNumPlayers; ++p) {
                    res.values[p] = h_outputValues[b * Defs::kNumPlayers + p];
                }
                for (size_t a = 0; a < Defs::kActionSpace; ++a) {
                    res.policy[a] = h_outputPolicy[b * Defs::kActionSpace + a];
                }
            }
        }
    };
}