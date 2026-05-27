#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <numeric>

using namespace nvinfer1;

// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

std::vector<char> loadEngine(const std::string& engine_file) {
    std::ifstream file(engine_file, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engine_file << std::endl;
        std::exit(EXIT_FAILURE);
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_file>" << std::endl;
        return 1;
    }

    std::string engine_path = argv[1];
    auto engine_data = loadEngine(engine_path);

    std::unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
    std::unique_ptr<ICudaEngine> engine{
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
    };
    std::unique_ptr<IExecutionContext> context{engine->createExecutionContext()};

    int num_io = engine->getNbIOTensors();
    std::unordered_map<std::string, void*> device_ptrs;

    for (int i = 0; i < num_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto dtype = engine->getTensorDataType(name);
        auto dims = engine->getTensorShape(name);

        int element_size = sizeof(float);
        int total_count = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            total_count *= dims.d[j];
        }

        void* device_ptr = nullptr;
        cudaMalloc(&device_ptr, total_count * element_size);
        device_ptrs[name] = device_ptr;

        if (mode == TensorIOMode::kINPUT) {
            std::vector<float> dummy_input(total_count, 1.0f);
            cudaMemcpy(device_ptr, dummy_input.data(), total_count * element_size, cudaMemcpyHostToDevice);
            std::cout << "Set input tensor: " << name << " (" << total_count << " elements)" << std::endl;
        } else {
            std::cout << "Set output tensor: " << name << " (" << total_count << " elements)" << std::endl;
        }

        context->setTensorAddress(name, device_ptr);
    }

    // GPU warm-up
    context->enqueueV3(nullptr);
    cudaDeviceSynchronize();

    // 10 runs of inference
    constexpr int N = 10;
    std::vector<double> times;
    for (int i = 0; i < N; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        context->enqueueV3(nullptr);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Average inference time over " << N << " runs: " << avg << " ms" << std::endl;

    for (int i = 0; i < num_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) != TensorIOMode::kOUTPUT)
            continue;

        auto dims = engine->getTensorShape(name);
        int total_count = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            total_count *= dims.d[j];
        }

        std::vector<float> host_output(total_count);
        cudaMemcpy(host_output.data(), device_ptrs[name], total_count * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Output tensor [" << name << "] first value: " << host_output[0] << std::endl;
    }

    for (auto& [name, ptr] : device_ptrs) {
        cudaFree(ptr);
    }

    return 0;
}
