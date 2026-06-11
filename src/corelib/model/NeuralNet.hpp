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
    // Values: Stores exactly 3 probabilities (Win, Draw, Loss) per player.
    // Maps identically to the GameResult::wdl schema.
    // ========================================================================
    template<ValidGameTraits GT>
    struct ModelResultsT
    {
        USING_GAME_TYPES(GT);

        std::array<float, Defs::kNumPlayers * 3> values{};  // WDL per player
        std::array<float, Defs::kActionSpace>    policy{};  // Post-softmax probabilities

        ModelResultsT() noexcept = default;
    };

    // ========================================================================
    // TENSORRT WRAPPER 
    // High-throughput, asynchronous GPU inference engine.
    // 
    // Design Intent:
    // Leverages CUDA Streams and Pinned Memory (cudaMallocHost) to execute 
    // Direct Memory Access (DMA) transfers without blocking the CPU thread.
    // ========================================================================
    template<ValidGameTraits GT>
    class NeuralNet
    {
    private:
        USING_GAME_TYPES(GT);
        using ModelResults = ModelResultsT<GT>;

        static constexpr uint32_t kValueOutSize = Defs::kNumPlayers * 3;

        int      m_deviceId;
        uint32_t m_inferenceBatchSize;

        cudaStream_t                 m_stream = nullptr;
        nvinfer1::IRuntime* m_runtime = nullptr;
        nvinfer1::ICudaEngine* m_engine = nullptr;
        nvinfer1::IExecutionContext* m_context = nullptr;

        float* d_input = nullptr;  // Device VRAM: NN input
        float* d_values = nullptr; // Device VRAM: WDL output 
        float* d_policy = nullptr; // Device VRAM: Policy output

        float* h_input = nullptr;  // Pinned RAM: NN input
        float* h_values = nullptr; // Pinned RAM: WDL output
        float* h_policy = nullptr; // Pinned RAM: Policy output

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
            try {
                loadEngine(enginePath);

                const uint32_t B = inferenceBatchSize;

                // Allocate VRAM
                CUDA_CHECK(cudaMalloc(&d_input, B * Defs::kNNInputSize * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_values, B * kValueOutSize * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_policy, B * Defs::kActionSpace * sizeof(float)));

                // Allocate Paged-Locked Host Memory (Prevents OS swapping, allows async DMA)
                CUDA_CHECK(cudaMallocHost(&h_input, B * Defs::kNNInputSize * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_values, B * kValueOutSize * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_policy, B * Defs::kActionSpace * sizeof(float)));

                // Map I/O tensors (Names MUST match the python ONNX exporter)
                m_context->setTensorAddress("input_state", d_input);
                m_context->setTensorAddress("value_output", d_values);
                m_context->setTensorAddress("policy_output", d_policy);
            }
            catch (...) {
                if (m_stream) cudaStreamDestroy(m_stream);
                throw;
            }
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
        // BATCH INFERENCE PIPELINE
        // ----------------------------------------------------------------
        void forwardBatch(
            const AlignedVec<const std::array<float, Defs::kNNInputSize>*>& batchPtrs,
            AlignedVec<ModelResults>& results)
        {
            const int32_t batchSize = static_cast<int32_t>(batchPtrs.size());
            if (batchSize == 0) return;

            if (results.size() < static_cast<size_t>(batchSize))
                results.resize(static_cast<size_t>(batchSize));

            CUDA_CHECK(cudaSetDevice(m_deviceId));

            // Dynamically adjust execution context for the current batch size
            m_context->setInputShape(
                "input_state",
                nvinfer1::Dims2{ batchSize, static_cast<int32_t>(Defs::kNNInputSize) });

            // OPTIMIZATION: Consolidate sparse MCTS tensors directly into contiguous Pinned Memory.
            for (int32_t b = 0; b < batchSize; ++b) {
                std::memcpy(
                    h_input + b * Defs::kNNInputSize,
                    batchPtrs[b]->data(),
                    Defs::kNNInputSize * sizeof(float)
                );
            }

            const size_t inputBytes = static_cast<size_t>(batchSize) * Defs::kNNInputSize * sizeof(float);

            // Asynchronous Execution: H2D -> Compute -> D2H
            CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, inputBytes, cudaMemcpyHostToDevice, m_stream));
            m_context->enqueueV3(m_stream);

            const size_t valueBytes = static_cast<size_t>(batchSize) * kValueOutSize * sizeof(float);
            const size_t policyBytes = static_cast<size_t>(batchSize) * Defs::kActionSpace * sizeof(float);

            CUDA_CHECK(cudaMemcpyAsync(h_values, d_values, valueBytes, cudaMemcpyDeviceToHost, m_stream));
            CUDA_CHECK(cudaMemcpyAsync(h_policy, d_policy, policyBytes, cudaMemcpyDeviceToHost, m_stream));

            // Block CPU thread until the GPU stream fully completes
            CUDA_CHECK(cudaStreamSynchronize(m_stream));

            // Unpack flat host buffers back into structured results
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