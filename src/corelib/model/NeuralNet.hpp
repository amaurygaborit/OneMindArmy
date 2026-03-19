#pragma once

#include <array>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "../bootstrap/GameConfig.hpp"
#include "../util/AlignedVec.hpp"

namespace Core
{
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::cerr << "[CUDA Error] " << __FILE__ << ":" << __LINE__        \
                      << " — " << cudaGetErrorString(_err) << "\n";            \
            std::terminate();                                                   \
        }                                                                       \
    } while (0)

    class TRTLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override
        {
            if (severity <= Severity::kWARNING)
                std::cerr << "[TensorRT] " << msg << "\n";
        }
    };
    inline static TRTLogger g_logger;

    // ========================================================================
    // MODEL RESULTS
    //
    // values : kNumPlayers * 3 WDL probabilities per player
    //   [p*3+0] = P(Win), [p*3+1] = P(Draw), [p*3+2] = P(Loss)
    //
    // This matches GameResult::wdl and NodeEvent::nnWDL exactly.
    // All three are std::array<float, kNumPlayers * 3>.
    //
    // NOTE: GameResult is a STRUCT { wdl: array<float,kNumPlayers*3>; reason: uint32_t }.
    //       ModelResultsT only holds the float payload — no reason field needed.
    // ========================================================================
    template<ValidGameTraits GT>
    struct ModelResultsT
    {
        USING_GAME_TYPES(GT);

        std::array<float, Defs::kNumPlayers * 3> values{};  // WDL per player
        std::array<float, Defs::kActionSpace>    policy{};  // post-softmax

        ModelResultsT() noexcept = default;
    };

    // ========================================================================
    // NEURAL NET (TensorRT Asynchronous GPU Wrapper)
    // ========================================================================
    template<ValidGameTraits GT>
    class NeuralNet
    {
    private:
        USING_GAME_TYPES(GT);
        using ModelResults = ModelResultsT<GT>;

        // Floats in the value output per sample = kNumPlayers * 3
        static constexpr uint32_t kValueOutSize = Defs::kNumPlayers * 3;

        int      m_deviceId;
        uint32_t m_inferenceBatchSize;

        cudaStream_t                 m_stream = nullptr;
        nvinfer1::IRuntime* m_runtime = nullptr;
        nvinfer1::ICudaEngine* m_engine = nullptr;
        nvinfer1::IExecutionContext* m_context = nullptr;

        float* d_input = nullptr;  // GPU: NN input
        float* d_values = nullptr;  // GPU: WDL output (kNumPlayers * 3 per sample)
        float* d_policy = nullptr;  // GPU: policy output

        float* h_input = nullptr;  // pinned CPU: NN input
        float* h_values = nullptr;  // pinned CPU: WDL output
        float* h_policy = nullptr;  // pinned CPU: policy output

        void loadEngine(const std::string& path)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open())
                throw std::runtime_error("NeuralNet: cannot open engine: " + path);

            file.seekg(0, std::ios::end);
            const size_t size = static_cast<size_t>(file.tellg());
            if (size == 0)
                throw std::runtime_error("NeuralNet: engine file is empty: " + path);
            file.seekg(0, std::ios::beg);

            AlignedVec<char> buf(reserve_only, size);
            file.read(buf.data(), static_cast<std::streamsize>(size));
            file.close();

            m_runtime = nvinfer1::createInferRuntime(g_logger);
            if (!m_runtime) throw std::runtime_error("NeuralNet: createInferRuntime failed.");

            m_engine = m_runtime->deserializeCudaEngine(buf.data(), size);
            if (!m_engine) throw std::runtime_error("NeuralNet: deserializeCudaEngine failed.");

            m_context = m_engine->createExecutionContext();
            if (!m_context) throw std::runtime_error("NeuralNet: createExecutionContext failed.");
        }

    public:
        NeuralNet(int deviceId, uint32_t inferenceBatchSize,
            const std::string& enginePath)
            : m_deviceId(deviceId)
            , m_inferenceBatchSize(inferenceBatchSize)
        {
            CUDA_CHECK(cudaSetDevice(m_deviceId));
            CUDA_CHECK(cudaStreamCreate(&m_stream));
            loadEngine(enginePath);

            const uint32_t B = inferenceBatchSize;

            // GPU buffers
            CUDA_CHECK(cudaMalloc(&d_input, B * Defs::kNNInputSize * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_values, B * kValueOutSize * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_policy, B * Defs::kActionSpace * sizeof(float)));

            // Pinned CPU buffers (direct DMA, no paging)
            CUDA_CHECK(cudaMallocHost(&h_input, B * Defs::kNNInputSize * sizeof(float)));
            CUDA_CHECK(cudaMallocHost(&h_values, B * kValueOutSize * sizeof(float)));
            CUDA_CHECK(cudaMallocHost(&h_policy, B * Defs::kActionSpace * sizeof(float)));

            // Tensor name bindings — must match the ONNX export in train.py
            m_context->setTensorAddress("input_state", d_input);
            m_context->setTensorAddress("value_output", d_values);
            m_context->setTensorAddress("policy_output", d_policy);
        }

        ~NeuralNet()
        {
            CUDA_CHECK(cudaSetDevice(m_deviceId));
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
            CUDA_CHECK(cudaStreamDestroy(m_stream));

            cudaFree(d_input);  cudaFree(d_values);  cudaFree(d_policy);
            cudaFreeHost(h_input); cudaFreeHost(h_values); cudaFreeHost(h_policy);

            if (m_context) { delete m_context; m_context = nullptr; }
            if (m_engine) { delete m_engine;  m_engine = nullptr; }
            if (m_runtime) { delete m_runtime; m_runtime = nullptr; }
        }

        NeuralNet(const NeuralNet&) = delete;
        NeuralNet& operator=(const NeuralNet&) = delete;

        // ----------------------------------------------------------------
        // BATCH INFERENCE
        // ----------------------------------------------------------------
        void forwardBatch(
            const AlignedVec<std::array<float, Defs::kNNInputSize>>& batchInputs,
            AlignedVec<ModelResults>& results)
        {
            const int32_t batchSize = static_cast<int32_t>(batchInputs.size());
            if (batchSize == 0) return;

            if (results.size() < static_cast<size_t>(batchSize))
                results.resize(static_cast<size_t>(batchSize));

            CUDA_CHECK(cudaSetDevice(m_deviceId));

            // Dynamic shape
            m_context->setInputShape(
                "input_state",
                nvinfer1::Dims2{ batchSize, static_cast<int32_t>(Defs::kNNInputSize) });

            // H2D
            const size_t inputBytes = static_cast<size_t>(batchSize) * Defs::kNNInputSize * sizeof(float);
            std::memcpy(h_input, batchInputs.data(), inputBytes);
            CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, inputBytes, cudaMemcpyHostToDevice, m_stream));

            // Async inference
            m_context->enqueueV3(m_stream);

            // D2H — kValueOutSize = kNumPlayers * 3 floats per sample
            const size_t valueBytes = static_cast<size_t>(batchSize) * kValueOutSize * sizeof(float);
            const size_t policyBytes = static_cast<size_t>(batchSize) * Defs::kActionSpace * sizeof(float);

            CUDA_CHECK(cudaMemcpyAsync(h_values, d_values, valueBytes, cudaMemcpyDeviceToHost, m_stream));
            CUDA_CHECK(cudaMemcpyAsync(h_policy, d_policy, policyBytes, cudaMemcpyDeviceToHost, m_stream));
            CUDA_CHECK(cudaStreamSynchronize(m_stream));

            // Unpack
            for (int32_t b = 0; b < batchSize; ++b) {
                ModelResults& res = results[static_cast<size_t>(b)];

                std::memcpy(res.values.data(),
                    h_values + static_cast<size_t>(b) * kValueOutSize,
                    kValueOutSize * sizeof(float));

                std::memcpy(res.policy.data(),
                    h_policy + static_cast<size_t>(b) * Defs::kActionSpace,
                    Defs::kActionSpace * sizeof(float));
            }
        }
    };
}