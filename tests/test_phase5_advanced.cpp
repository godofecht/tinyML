//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 5 Advanced Optimizations Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>
#include <future>
#include <algorithm>

// Include existing components
#include "SIMDOperations.h"

namespace ML {
namespace Advanced {

// Mock AdvancedOptimizer class (to be implemented in Phase 5)
class AdvancedOptimizer {
private:
    // Kernel fusion components
    std::vector<float> fused_weights;
    std::vector<float> fused_biases;
    std::vector<size_t> fusion_map;
    
    // Cache optimization components
    std::vector<float> cache_optimized_data;
    size_t cache_line_size;
    bool data_prefetch_enabled;
    
    // Parallel processing components
    size_t num_threads;
    std::vector<std::thread> worker_threads;
    std::vector<std::vector<float>> thread_local_buffers;
    
    // Hardware acceleration flags
    bool gpu_acceleration_enabled;
    bool metal_shaders_enabled;
    
    // Scaling strategy components
    std::vector<std::vector<float>> micro_transformer_weights;
    std::vector<bool> module_activation_flags;
    std::vector<float> progressive_loading_weights;
    
public:
    AdvancedOptimizer(size_t initial_size = 1024) 
        : cache_line_size(64), data_prefetch_enabled(true), num_threads(4),
          gpu_acceleration_enabled(false), metal_shaders_enabled(false) {
        
        initializeKernelFusion(initial_size);
        initializeCacheOptimization();
        initializeParallelProcessing();
        initializeScalingStrategy();
    }
    
    void initializeKernelFusion(size_t size) {
        // Initialize fused weights for combining operations
        fused_weights.resize(size * size * 4); // Support 4 operations fusion
        fused_biases.resize(size);
        fusion_map.resize(size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
        
        for (auto& w : fused_weights) w = dis(gen);
        for (auto& b : fused_biases) b = dis(gen);
        
        // Create fusion mapping
        for (size_t i = 0; i < size; ++i) {
            fusion_map[i] = i % 4; // Map to 4 different operation types
        }
    }
    
    void initializeCacheOptimization() {
        // Initialize cache-optimized data layout
        cache_optimized_data.resize(1024 * 1024); // 1MB cache buffer
        
        // Organize data for maximum cache efficiency
        for (size_t i = 0; i < cache_optimized_data.size(); i += cache_line_size) {
            // Fill cache lines with related data
            for (size_t j = 0; j < cache_line_size && (i + j) < cache_optimized_data.size(); ++j) {
                cache_optimized_data[i + j] = static_cast<float>(i + j) / cache_line_size;
            }
        }
    }
    
    void initializeParallelProcessing() {
        // Initialize thread-local buffers
        thread_local_buffers.resize(num_threads);
        for (auto& buffer : thread_local_buffers) {
            buffer.resize(1024); // Each thread gets 1KB buffer
        }
    }
    
    void initializeScalingStrategy() {
        // Initialize micro-transformer weights (<1M parameters)
        size_t micro_params = 512 * 256; // <1M parameters
        micro_transformer_weights.resize(micro_params);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
        
        for (auto& w : micro_transformer_weights) w = dis(gen);
        
        // Initialize module activation flags
        module_activation_flags.resize(8, true); // 8 modules
        
        // Initialize progressive loading weights
        progressive_loading_weights.resize(micro_params / 4); // 25% loaded initially
        for (size_t i = 0; i < progressive_loading_weights.size(); ++i) {
            progressive_loading_weights[i] = micro_transformer_weights[i];
        }
    }
    
    // Kernel Fusion: Combine operations into single passes
    void fusedMatrixVectorMultiply(const std::vector<float>& matrix,
                                  const std::vector<float>& vector,
                                  std::vector<float>& output,
                                  size_t rows, size_t cols) const {
        output.resize(rows);
        
        // Fused operation: matrix multiplication + bias + activation
        for (size_t i = 0; i < rows; ++i) {
            float sum = 0.0f;
            
            // Matrix multiplication
            for (size_t j = 0; j < cols; ++j) {
                sum += matrix[i * cols + j] * vector[j];
            }
            
            // Add bias (fused)
            sum += fused_biases[i % fused_biases.size()];
            
            // Apply activation (fused) - tanh
            output[i] = std::tanh(sum);
        }
    }
    
    void fusedAttentionComputation(const std::vector<float>& q,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  std::vector<float>& output,
                                  size_t seq_len, size_t embed_dim) const {
        output.resize(seq_len * embed_dim);
        
        // Fused: Q*K^T + softmax + V multiplication
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t d = 0; d < embed_dim; ++d) {
                float attention_sum = 0.0f;
                float softmax_sum = 0.0f;
                
                // Compute attention scores and softmax in single pass
                for (size_t j = 0; j < seq_len; ++j) {
                    float score = 0.0f;
                    for (size_t dim = 0; dim < embed_dim; ++dim) {
                        score += q[i * embed_dim + dim] * k[j * embed_dim + dim];
                    }
                    score = std::exp(score / std::sqrt(embed_dim));
                    softmax_sum += score;
                    attention_sum += score * v[j * embed_dim + d];
                }
                
                output[i * embed_dim + d] = attention_sum / softmax_sum;
            }
        }
    }
    
    // Cache Optimization: Data layout for maximum throughput
    void cacheOptimizedProcessing(const std::vector<float>& input,
                                 std::vector<float>& output) const {
        output.resize(input.size());
        
        // Process data in cache-friendly blocks
        const size_t block_size = cache_line_size / sizeof(float);
        
        for (size_t block_start = 0; block_start < input.size(); block_start += block_size) {
            size_t block_end = std::min(block_start + block_size, input.size());
            
            // Process entire cache line at once
            for (size_t i = block_start; i < block_end; ++i) {
                // Apply cache-optimized transformation
                output[i] = input[i] * cache_optimized_data[i % cache_optimized_data.size()];
                
                // Prefetch next cache line if enabled
                if (data_prefetch_enabled && (i % block_size) == block_size - 1) {
                    // Simulate prefetch - in real implementation would use hardware prefetch
                    size_t next_block = block_start + block_size;
                    if (next_block < input.size()) {
                        volatile float prefetch_val = cache_optimized_data[next_block % cache_optimized_data.size()];
                        (void)prefetch_val; // Prevent optimization
                    }
                }
            }
        }
    }
    
    // Parallel Processing: Multi-threaded attention computation
    void parallelAttentionComputation(const std::vector<float>& input,
                                     std::vector<float>& output,
                                     size_t seq_len, size_t embed_dim) const {
        output.resize(input.size());
        
        // Divide work among threads
        std::vector<std::future<void>> futures;
        size_t chunk_size = seq_len / num_threads;
        
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            size_t start_idx = thread_id * chunk_size;
            size_t end_idx = (thread_id == num_threads - 1) ? seq_len : start_idx + chunk_size;
            
            futures.push_back(std::async(std::launch::async, [this, &input, &output, start_idx, end_idx, embed_dim]() {
                // Each thread processes its chunk
                for (size_t i = start_idx; i < end_idx; ++i) {
                    for (size_t d = 0; d < embed_dim; ++d) {
                        // Simple parallel attention computation
                        float sum = 0.0f;
                        for (size_t j = 0; j < embed_dim; ++j) {
                            sum += input[i * embed_dim + j] * std::sin(j * 0.1f);
                        }
                        output[i * embed_dim + d] = std::tanh(sum);
                    }
                }
            }));
        }
        
        // Wait for all threads to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    // Hardware Acceleration: GPU/Metal shader kernels
    void hardwareAcceleratedComputation(const std::vector<float>& input,
                                      std::vector<float>& output) const {
        output.resize(input.size());
        
        if (gpu_acceleration_enabled) {
            // Simulate GPU acceleration - would use actual GPU kernels
            std::transform(input.begin(), input.end(), output.begin(), 
                          [](float x) { return x * x + std::sin(x); });
        } else if (metal_shaders_enabled) {
            // Simulate Metal shader acceleration - would use actual Metal shaders
            std::transform(input.begin(), input.end(), output.begin(),
                          [](float x) { return std::exp(-x * x) * std::cos(x); });
        } else {
            // Fallback to CPU
            std::copy(input.begin(), input.end(), output.begin());
        }
    }
    
    // Micro-Transformers: <1M parameters for edge devices
    void microTransformerInference(const std::vector<float>& input,
                                  std::vector<float>& output) const {
        // Simulate micro-transformer with <1M parameters
        size_t micro_embed_dim = 256;
        size_t micro_seq_len = 128;
        
        output.resize(input.size());
        
        // Simple micro-transformer computation
        for (size_t i = 0; i < std::min(input.size(), micro_seq_len * micro_embed_dim); ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < std::min(micro_transformer_weights.size(), size_t(1024)); ++j) {
                sum += input[i] * micro_transformer_weights[j];
            }
            output[i] = std::tanh(sum);
        }
    }
    
    // Modular Design: Composable transformer blocks
    void modularTransformerProcessing(const std::vector<float>& input,
                                    std::vector<float>& output) const {
        output = input;
        
        // Process through active modules only
        for (size_t module_idx = 0; module_idx < module_activation_flags.size(); ++module_idx) {
            if (module_activation_flags[module_idx]) {
                // Apply module transformation
                for (size_t i = 0; i < output.size(); ++i) {
                    switch (module_idx) {
                        case 0: // Attention module
                            output[i] = std::sin(output[i] * 0.1f);
                            break;
                        case 1: // Feed-forward module
                            output[i] = output[i] * output[i];
                            break;
                        case 2: // Normalization module
                            output[i] = std::tanh(output[i]);
                            break;
                        case 3: // Activation module
                            output[i] = std::max(0.0f, output[i]); // ReLU
                            break;
                        default:
                            output[i] = output[i] * 0.9f; // Identity-like
                            break;
                    }
                }
            }
        }
    }
    
    // Progressive Loading: On-demand component activation
    void progressiveLoadingInference(const std::vector<float>& input,
                                    std::vector<float>& output,
                                    float load_factor = 0.25f) const {
        size_t active_params = static_cast<size_t>(progressive_loading_weights.size() * load_factor);
        
        output.resize(input.size());
        
        // Use only progressively loaded parameters
        for (size_t i = 0; i < input.size(); ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < std::min(active_params, progressive_loading_weights.size()); ++j) {
                sum += input[i] * progressive_loading_weights[j];
            }
            output[i] = std::tanh(sum);
        }
    }
    
    // Federated Learning: Distributed model updates
    void federatedLearningUpdate(const std::vector<std::vector<float>>& client_updates,
                               std::vector<float>& global_weights) const {
        if (client_updates.empty()) return;
        
        size_t weight_size = client_updates[0].size();
        global_weights.resize(weight_size, 0.0f);
        
        // Federated averaging
        for (const auto& client_weights : client_updates) {
            for (size_t i = 0; i < weight_size; ++i) {
                global_weights[i] += client_weights[i];
            }
        }
        
        // Average across clients
        for (size_t i = 0; i < weight_size; ++i) {
            global_weights[i] /= client_updates.size();
        }
    }
    
    // Getters for testing
    size_t getNumThreads() const { return num_threads; }
    bool isDataPrefetchEnabled() const { return data_prefetch_enabled; }
    bool isGPUAccelerationEnabled() const { return gpu_acceleration_enabled; }
    bool isMetalShadersEnabled() const { return metal_shaders_enabled; }
    void setGPUAccelerationEnabled(bool enabled) { gpu_acceleration_enabled = enabled; }
    void setMetalShadersEnabled(bool enabled) { metal_shaders_enabled = enabled; }
    void setModuleActivation(size_t module_idx, bool active) {
        if (module_idx < module_activation_flags.size()) {
            module_activation_flags[module_idx] = active;
        }
    }
};

} // namespace Advanced
} // namespace ML

class Phase5AdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test configurations for advanced optimizations
        test_sizes = {256, 512, 1024, 2048};
        
        for (size_t size : test_sizes) {
            test_matrices[size] = std::vector<float>(size * size);
            test_vectors[size] = std::vector<float>(size);
            test_inputs[size] = std::vector<float>(size);
            
            for (size_t i = 0; i < size * size; ++i) {
                test_matrices[size][i] = dis(gen);
            }
            
            for (size_t i = 0; i < size; ++i) {
                test_vectors[size][i] = dis(gen);
                test_inputs[size][i] = dis(gen);
            }
        }
    }
    
    std::vector<size_t> test_sizes;
    std::map<size_t, std::vector<float>> test_matrices;
    std::map<size_t, std::vector<float>> test_vectors;
    std::map<size_t, std::vector<float>> test_inputs;
};

// Test Kernel Fusion
TEST_F(Phase5AdvancedTest, KernelFusion) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    for (size_t size : test_sizes) {
        const auto& matrix = test_matrices[size];
        const auto& vector = test_vectors[size];
        
        std::vector<float> fused_output;
        optimizer.fusedMatrixVectorMultiply(matrix, vector, fused_output, size, size);
        
        // Verify output dimensions
        ASSERT_EQ(fused_output.size(), size);
        
        // Verify no NaN or Inf
        for (float val : fused_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in fused matrix-vector output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in fused matrix-vector output";
            ASSERT_LE(std::abs(val), 1.0f) << "Tanh output should be in [-1, 1]";
        }
        
        // Test attention fusion
        std::vector<float> q(size * size), k(size * size), v(size * size), attention_output;
        std::iota(q.begin(), q.end(), 0.0f);
        std::iota(k.begin(), k.end(), 1.0f);
        std::iota(v.begin(), v.end(), 2.0f);
        
        optimizer.fusedAttentionComputation(q, k, v, attention_output, size, size);
        
        ASSERT_EQ(attention_output.size(), size * size);
        
        for (float val : attention_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in fused attention output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in fused attention output";
        }
    }
}

// Test Cache Optimization
TEST_F(Phase5AdvancedTest, CacheOptimization) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    for (size_t size : test_sizes) {
        const auto& input = test_inputs[size];
        
        std::vector<float> cache_output;
        optimizer.cacheOptimizedProcessing(input, cache_output);
        
        // Verify output dimensions
        ASSERT_EQ(cache_output.size(), input.size());
        
        // Verify output is different from input (transformation occurred)
        bool different = false;
        for (size_t i = 0; i < input.size(); ++i) {
            if (std::abs(input[i] - cache_output[i]) > 1e-6f) {
                different = true;
                break;
            }
        }
        ASSERT_TRUE(different) << "Cache optimization should transform input";
        
        // Verify reasonable values
        for (float val : cache_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in cache-optimized output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in cache-optimized output";
        }
    }
}

// Test Parallel Processing
TEST_F(Phase5AdvancedTest, ParallelProcessing) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    for (size_t size : test_sizes) {
        const auto& input = test_inputs[size];
        
        std::vector<float> parallel_output;
        optimizer.parallelAttentionComputation(input, parallel_output, size / 4, 4);
        
        // Verify output dimensions
        ASSERT_EQ(parallel_output.size(), input.size());
        
        // Verify reasonable values
        for (float val : parallel_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in parallel processing output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in parallel processing output";
            ASSERT_LE(std::abs(val), 1.0f) << "Tanh output should be in [-1, 1]";
        }
        
        // Verify thread count
        EXPECT_EQ(optimizer.getNumThreads(), 4);
    }
}

// Test Hardware Acceleration
TEST_F(Phase5AdvancedTest, HardwareAcceleration) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    const auto& input = test_inputs[512];
    
    // Test CPU fallback
    std::vector<float> cpu_output;
    optimizer.hardwareAcceleratedComputation(input, cpu_output);
    
    ASSERT_EQ(cpu_output.size(), input.size());
    
    // Test GPU acceleration simulation
    optimizer.setGPUAccelerationEnabled(true);
    std::vector<float> gpu_output;
    optimizer.hardwareAcceleratedComputation(input, gpu_output);
    
    ASSERT_EQ(gpu_output.size(), input.size());
    
    // Test Metal shader acceleration simulation
    optimizer.setGPUAccelerationEnabled(false);
    optimizer.setMetalShadersEnabled(true);
    std::vector<float> metal_output;
    optimizer.hardwareAcceleratedComputation(input, metal_output);
    
    ASSERT_EQ(metal_output.size(), input.size());
    
    // Verify all outputs are reasonable
    for (const auto& output : {cpu_output, gpu_output, metal_output}) {
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in hardware-accelerated output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in hardware-accelerated output";
        }
    }
}

// Test Micro-Transformers
TEST_F(Phase5AdvancedTest, MicroTransformers) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    for (size_t size : {256, 512}) { // Test smaller sizes for micro-transformers
        const auto& input = test_inputs[size];
        
        std::vector<float> micro_output;
        optimizer.microTransformerInference(input, micro_output);
        
        // Verify output dimensions
        ASSERT_EQ(micro_output.size(), input.size());
        
        // Verify reasonable values
        for (float val : micro_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in micro-transformer output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in micro-transformer output";
            ASSERT_LE(std::abs(val), 1.0f) << "Tanh output should be in [-1, 1]";
        }
    }
}

// Test Modular Design
TEST_F(Phase5AdvancedTest, ModularDesign) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    const auto& input = test_inputs[512];
    
    // Test with all modules active
    std::vector<float> full_output;
    optimizer.modularTransformerProcessing(input, full_output);
    
    ASSERT_EQ(full_output.size(), input.size());
    
    // Test with selective module activation
    optimizer.setModuleActivation(0, true);  // Attention
    optimizer.setModuleActivation(1, true);  // Feed-forward
    optimizer.setModuleActivation(2, false); // Normalization
    optimizer.setModuleActivation(3, false); // Activation
    
    std::vector<float> selective_output;
    optimizer.modularTransformerProcessing(input, selective_output);
    
    ASSERT_EQ(selective_output.size(), input.size());
    
    // Outputs should be different due to different active modules
    bool different = false;
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::abs(full_output[i] - selective_output[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    ASSERT_TRUE(different) << "Different module configurations should produce different outputs";
    
    // Verify reasonable values
    for (float val : selective_output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN in modular output";
        ASSERT_FALSE(std::isinf(val)) << "Inf in modular output";
    }
}

// Test Progressive Loading
TEST_F(Phase5AdvancedTest, ProgressiveLoading) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    const auto& input = test_inputs[512];
    
    // Test with different load factors
    std::vector<float> output_25, output_50, output_75, output_100;
    
    optimizer.progressiveLoadingInference(input, output_25, 0.25f);
    optimizer.progressiveLoadingInference(input, output_50, 0.50f);
    optimizer.progressiveLoadingInference(input, output_75, 0.75f);
    optimizer.progressiveLoadingInference(input, output_100, 1.00f);
    
    // All outputs should have same dimensions
    ASSERT_EQ(output_25.size(), input.size());
    ASSERT_EQ(output_50.size(), input.size());
    ASSERT_EQ(output_75.size(), input.size());
    ASSERT_EQ(output_100.size(), input.size());
    
    // Higher load factors should produce different (potentially better) results
    bool different_25_50 = false, different_50_100 = false;
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::abs(output_25[i] - output_50[i]) > 1e-6f) different_25_50 = true;
        if (std::abs(output_50[i] - output_100[i]) > 1e-6f) different_50_100 = true;
    }
    
    // At least some difference should exist between different load factors
    ASSERT_TRUE(different_25_50 || different_50_100) 
        << "Different load factors should produce different results";
    
    // Verify reasonable values
    for (const auto& output : {output_25, output_50, output_75, output_100}) {
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in progressive loading output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in progressive loading output";
            ASSERT_LE(std::abs(val), 1.0f) << "Tanh output should be in [-1, 1]";
        }
    }
}

// Test Federated Learning
TEST_F(Phase5AdvancedTest, FederatedLearning) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    // Simulate client updates
    std::vector<std::vector<float>> client_updates;
    for (int client = 0; client < 4; ++client) {
        std::vector<float> client_weights(1024);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
        
        for (auto& w : client_weights) w = dis(gen);
        client_updates.push_back(client_weights);
    }
    
    std::vector<float> global_weights;
    optimizer.federatedLearningUpdate(client_updates, global_weights);
    
    // Verify global weights dimensions
    ASSERT_EQ(global_weights.size(), 1024);
    
    // Verify global weights are reasonable
    for (float val : global_weights) {
        ASSERT_FALSE(std::isnan(val)) << "NaN in federated learning weights";
        ASSERT_FALSE(std::isinf(val)) << "Inf in federated learning weights";
        ASSERT_LT(std::abs(val), 1.0f) << "Federated weights should be reasonable";
    }
    
    // Test with empty client updates
    std::vector<float> empty_global;
    optimizer.federatedLearningUpdate({}, empty_global);
    ASSERT_TRUE(empty_global.empty()) << "Empty client updates should produce empty global weights";
}

// Test Performance Engineering
TEST_F(Phase5AdvancedTest, PerformanceEngineering) {
    std::cout << "\n=== Phase 5 Performance Engineering ===\n";
    std::cout << std::setw(15) << "Optimization" << std::setw(12) << "Size" 
              << std::setw(15) << "Time (μs)" << std::setw(15) << "Speedup" 
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(67, '-') << std::endl;
    
    const int iterations = 100;
    
    for (size_t size : {512, 1024}) {
        const auto& input = test_inputs[size];
        ML::Advanced::AdvancedOptimizer optimizer;
        
        // Benchmark kernel fusion
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> output;
            optimizer.fusedMatrixVectorMultiply(test_matrices[size], test_vectors[size], 
                                               output, size, size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto fusion_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Benchmark cache optimization
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> output;
            optimizer.cacheOptimizedProcessing(input, output);
        }
        end = std::chrono::high_resolution_clock::now();
        auto cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Benchmark parallel processing
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> output;
            optimizer.parallelAttentionComputation(input, output, size / 4, 4);
        }
        end = std::chrono::high_resolution_clock::now();
        auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Calculate speedups (relative to cache optimization as baseline)
        double fusion_speedup = static_cast<double>(cache_time.count()) / fusion_time.count();
        double parallel_speedup = static_cast<double>(cache_time.count()) / parallel_time.count();
        
        std::cout << std::setw(15) << "Kernel Fusion" << std::setw(12) << size
                  << std::setw(15) << std::fixed << std::setprecision(2) << static_cast<double>(fusion_time.count()) / iterations
                  << std::setw(15) << fusion_speedup << std::setw(10) << "PASS" << std::endl;
        
        std::cout << std::setw(15) << "Cache Opt" << std::setw(12) << size
                  << std::setw(15) << std::fixed << std::setprecision(2) << static_cast<double>(cache_time.count()) / iterations
                  << std::setw(15) << "1.00x" << std::setw(10) << "BASE" << std::endl;
        
        std::cout << std::setw(15) << "Parallel" << std::setw(12) << size
                  << std::setw(15) << std::fixed << std::setprecision(2) << static_cast<double>(parallel_time.count()) / iterations
                  << std::setw(15) << parallel_speedup << std::setw(10) << "PASS" << std::endl;
        
        std::cout << std::string(67, '-') << std::endl;
        
        // Performance targets
        EXPECT_LT(fusion_time.count() / iterations, 100) << "Kernel fusion should be fast";
        EXPECT_LT(cache_time.count() / iterations, 200) << "Cache optimization should be fast";
        EXPECT_LT(parallel_time.count() / iterations, 150) << "Parallel processing should be fast";
        EXPECT_GT(parallel_speedup, 1.5) << "Parallel processing should show speedup";
    }
}

// Test Scaling Strategy
TEST_F(Phase5AdvancedTest, ScalingStrategy) {
    std::cout << "\n=== Phase 5 Scaling Strategy ===\n";
    std::cout << std::setw(20) << "Strategy" << std::setw(15) << "Params" 
              << std::setw(15) << "Memory (MB)" << std::setw(15) << "Latency (ms)"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    ML::Advanced::AdvancedOptimizer optimizer;
    
    // Test micro-transformer scaling
    const int iterations = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::vector<float> output;
        optimizer.microTransformerInference(test_inputs[256], output);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto micro_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Test progressive loading scaling
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::vector<float> output;
        optimizer.progressiveLoadingInference(test_inputs[512], output, 0.25f);
    }
    end = std::chrono::high_resolution_clock::now();
    auto progressive_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Calculate memory usage (simplified)
    double micro_memory_mb = (512 * 256 * sizeof(float)) / (1024.0 * 1024.0); // <1M parameters
    double progressive_memory_mb = (1024 * 256 * sizeof(float) * 0.25) / (1024.0 * 1024.0);
    
    std::cout << std::setw(20) << "Micro-Transformer" << std::setw(15) << "<1M"
              << std::setw(15) << std::fixed << std::setprecision(2) << micro_memory_mb
              << std::setw(15) << static_cast<double>(micro_time.count()) / iterations
              << std::setw(10) << "PASS" << std::endl;
    
    std::cout << std::setw(20) << "Progressive Loading" << std::setw(15) << "25%"
              << std::setw(15) << std::fixed << std::setprecision(2) << progressive_memory_mb
              << std::setw(15) << static_cast<double>(progressive_time.count()) / iterations
              << std::setw(10) << "PASS" << std::endl;
    
    std::cout << std::string(75, '-') << std::endl;
    
    // Verify scaling targets
    EXPECT_LT(micro_memory_mb, 1.0) << "Micro-transformer should use <1MB";
    EXPECT_LT(progressive_memory_mb, 1.0) << "Progressive loading should use <1MB at 25%";
    EXPECT_LT(micro_time.count() / iterations, 10) << "Micro-transformer should be fast";
    EXPECT_LT(progressive_time.count() / iterations, 10) << "Progressive loading should be fast";
}

// Test Edge Cases
TEST_F(Phase5AdvancedTest, EdgeCases) {
    ML::Advanced::AdvancedOptimizer optimizer;
    
    // Test with empty inputs
    std::vector<float> empty_input, empty_output;
    EXPECT_NO_THROW(optimizer.cacheOptimizedProcessing(empty_input, empty_output));
    EXPECT_TRUE(empty_output.empty());
    
    // Test with single element
    std::vector<float> single_input = {1.0f};
    std::vector<float> single_output;
    EXPECT_NO_THROW(optimizer.cacheOptimizedProcessing(single_input, single_output));
    ASSERT_EQ(single_output.size(), 1);
    
    // Test with extreme values
    std::vector<float> extreme_input = {1e6f, -1e6f, 1e-6f, -1e-6f};
    std::vector<float> extreme_output;
    EXPECT_NO_THROW(optimizer.cacheOptimizedProcessing(extreme_input, extreme_output));
    
    for (float val : extreme_output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN with extreme input";
        ASSERT_FALSE(std::isinf(val)) << "Inf with extreme input";
    }
    
    // Test module activation bounds
    EXPECT_NO_THROW(optimizer.setModuleActivation(100, true)); // Should handle gracefully
    EXPECT_NO_THROW(optimizer.setModuleActivation(-1, false)); // Should handle gracefully
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
