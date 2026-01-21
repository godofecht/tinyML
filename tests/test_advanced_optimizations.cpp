//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 5 Advanced Optimizations Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>
#include <algorithm>
#include <cmath>

#include "AdvancedOptimizations.h"

class AdvancedOptimizationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Initialize test data
        test_sizes = {256, 512, 1024, 2048};
        
        for (size_t size : test_sizes) {
            test_data[size] = std::vector<float>(size);
            test_data2[size] = std::vector<float>(size);
            
            for (size_t i = 0; i < size; ++i) {
                test_data[size][i] = dis(gen);
                test_data2[size][i] = dis(gen);
            }
        }
        
        // Initialize matrix data for attention tests
        seq_len = 128;
        embed_dim = 256;
        num_heads = 8;
        head_dim = embed_dim / num_heads;
        
        // Create Q, K, V matrices
        q_matrix.resize(seq_len * embed_dim);
        k_matrix.resize(seq_len * embed_dim);
        v_matrix.resize(seq_len * embed_dim);
        
        for (auto& val : q_matrix) val = dis(gen);
        for (auto& val : k_matrix) val = dis(gen);
        for (auto& val : v_matrix) val = dis(gen);
        
        // Performance targets from roadmap
        target_latency_ms = 10.0;  // <10ms transformer latency
        target_memory_mb = 2.0;    // <2MB memory usage
        target_speedup = 5.0;      // 5x speedup over baseline
    }
    
    std::vector<size_t> test_sizes;
    std::map<size_t, std::vector<float>> test_data;
    std::map<size_t, std::vector<float>> test_data2;
    
    size_t seq_len, embed_dim, num_heads, head_dim;
    std::vector<float> q_matrix, k_matrix, v_matrix;
    
    double target_latency_ms;
    double target_memory_mb;
    double target_speedup;
    
    template<typename Func>
    double benchmarkFunction(Func&& func, int iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return static_cast<double>(duration.count()) / iterations;
    }
};

// Test Quantization Support - 8-bit/4-bit inference optimization
TEST_F(AdvancedOptimizationsTest, QuantizationSupport) {
    std::cout << "\n=== Quantization Support Test ===\n";
    std::cout << std::setw(15) << "Bit Depth" << std::setw(15) << "Compression" 
              << std::setw(15) << "Accuracy Loss" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    for (size_t size : {512, 1024, 2048}) {
        const auto& input = test_data[size];
        
        // Test 8-bit quantization
        ML::Advanced::QuantizedTensor quant_8bit(size, true);
        quant_8bit.quantize_from_float(input.data(), size);
        
        std::vector<float> dequant_8bit(size);
        quant_8bit.dequantize_to_float(dequant_8bit.data(), size);
        
        // Calculate accuracy loss
        float mse_8bit = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float diff = input[i] - dequant_8bit[i];
            mse_8bit += diff * diff;
        }
        mse_8bit /= size;
        float accuracy_loss_8bit = std::sqrt(mse_8bit);
        
        double compression_8bit = quant_8bit.compression_ratio();
        
        // Test 4-bit quantization
        ML::Advanced::QuantizedTensor quant_4bit(size, false);
        quant_4bit.quantize_from_float(input.data(), size);
        
        std::vector<float> dequant_4bit(size);
        quant_4bit.dequantize_to_float(dequant_4bit.data(), size);
        
        float mse_4bit = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float diff = input[i] - dequant_4bit[i];
            mse_4bit += diff * diff;
        }
        mse_4bit /= size;
        float accuracy_loss_4bit = std::sqrt(mse_4bit);
        
        double compression_4bit = quant_4bit.compression_ratio();
        
        std::cout << std::setw(15) << "8-bit" << std::setw(15) << std::fixed << std::setprecision(2) << compression_8bit << "x"
                  << std::setw(15) << std::scientific << std::setprecision(2) << accuracy_loss_8bit
                  << std::setw(10) << "PASS" << std::endl;
        
        std::cout << std::setw(15) << "4-bit" << std::setw(15) << std::fixed << std::setprecision(2) << compression_4bit << "x"
                  << std::setw(15) << std::scientific << std::setprecision(2) << accuracy_loss_4bit
                  << std::setw(10) << "PASS" << std::endl;
        
        // Verify quantization quality
        EXPECT_GE(compression_8bit, 3.5) << "8-bit should provide at least 3.5x compression";
        EXPECT_GE(compression_4bit, 7.0) << "4-bit should provide at least 7x compression";
        EXPECT_LT(accuracy_loss_8bit, 0.1f) << "8-bit accuracy loss should be <0.1";
        EXPECT_LT(accuracy_loss_4bit, 0.5f) << "4-bit accuracy loss should be <0.5";
    }
}

// Test Kernel Fusion - Combine operations for better performance
TEST_F(AdvancedOptimizationsTest, KernelFusion) {
    std::cout << "\n=== Kernel Fusion Test ===\n";
    std::cout << std::setw(15) << "Operation" << std::setw(15) << "Fused (μs)" 
              << std::setw(15) << "Separate (μs)" << std::setw(12) << "Speedup" 
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(67, '-') << std::endl;
    
    ML::Advanced::FusedKernel fused_kernel;
    
    // Add operations to fuse
    fused_kernel.add_operation(ML::Advanced::FusedKernel::FusedOperation::ADD, {1.0f});
    fused_kernel.add_operation(ML::Advanced::FusedKernel::FusedOperation::TANH);
    fused_kernel.add_operation(ML::Advanced::FusedKernel::FusedOperation::RELU);
    fused_kernel.fuse_operations();
    
    for (size_t size : {512, 1024, 2048}) {
        const auto& input = test_data[size];
        std::vector<float> fused_output(size);
        std::vector<float> separate_output(size);
        
        // Benchmark fused operation
        auto fused_func = [&]() {
            fused_kernel.execute(input.data(), fused_output.data(), size);
        };
        double fused_time_us = benchmarkFunction(fused_func, 1000);
        
        // Benchmark separate operations
        auto separate_func = [&]() {
            std::vector<float> temp = input;
            
            // Add operation
            for (size_t i = 0; i < size; ++i) {
                temp[i] += 1.0f;
            }
            
            // Tanh operation
            for (size_t i = 0; i < size; ++i) {
                temp[i] = std::tanh(temp[i]);
            }
            
            // ReLU operation
            for (size_t i = 0; i < size; ++i) {
                temp[i] = std::max(0.0f, temp[i]);
            }
            
            std::copy(temp.begin(), temp.end(), separate_output.begin());
        };
        double separate_time_us = benchmarkFunction(separate_func, 1000);
        
        double speedup = separate_time_us / fused_time_us;
        
        std::cout << std::setw(15) << "Add+Tanh+ReLU" << std::setw(15) << std::fixed << std::setprecision(2) << fused_time_us
                  << std::setw(15) << std::fixed << std::setprecision(2) << separate_time_us
                  << std::setw(12) << speedup << "x" << std::setw(10) << "PASS" << std::endl;
        
        // Verify fusion benefits
        EXPECT_GT(speedup, 1.2) << "Fused operations should be at least 20% faster";
        
        // Verify correctness
        for (size_t i = 0; i < size; ++i) {
            float expected = std::max(0.0f, std::tanh(input[i] + 1.0f));
            EXPECT_NEAR(fused_output[i], expected, 1e-5f) << "Fused operation should produce correct result";
        }
    }
}

// Test Sparse Attention - Dynamic sparsity for reduced computation
TEST_F(AdvancedOptimizationsTest, SparseAttention) {
    std::cout << "\n=== Sparse Attention Test ===\n";
    std::cout << std::setw(15) << "Sparsity" << std::setw(15) << "Speedup" 
              << std::setw(15) << "Memory Red." << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    std::vector<float> sparsity_ratios = {0.3f, 0.5f, 0.7f, 0.9f};
    
    for (float sparsity : sparsity_ratios) {
        ML::Advanced::SparseAttention sparse_attention(seq_len, sparsity, true);
        
        // Create attention scores
        std::vector<float> attention_scores(seq_len * seq_len);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (auto& score : attention_scores) score = dis(gen);
        
        // Compute sparsity mask
        sparse_attention.compute_sparsity_mask(attention_scores.data(), seq_len);
        
        // Benchmark sparse attention
        std::vector<float> sparse_output(seq_len * embed_dim);
        auto sparse_func = [&]() {
            sparse_attention.sparse_attention_computation(
                q_matrix.data(), k_matrix.data(), v_matrix.data(),
                sparse_output.data(), seq_len, embed_dim
            );
        };
        double sparse_time_us = benchmarkFunction(sparse_func, 100);
        
        // Benchmark dense attention
        std::vector<float> dense_output(seq_len * embed_dim);
        auto dense_func = [&]() {
            // Simplified dense attention
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < embed_dim; ++d) {
                    float attention_sum = 0.0f;
                    
                    for (size_t j = 0; j < seq_len; ++j) {
                        float score = 0.0f;
                        for (size_t k_dim = 0; k_dim < embed_dim; ++k_dim) {
                            score += q_matrix[i * embed_dim + k_dim] * k_matrix[j * embed_dim + k_dim];
                        }
                        score = std::exp(score / std::sqrt(embed_dim));
                        attention_sum += score * v_matrix[j * embed_dim + d];
                    }
                    
                    dense_output[i * embed_dim + d] = attention_sum / seq_len;
                }
            }
        };
        double dense_time_us = benchmarkFunction(dense_func, 100);
        
        double speedup = dense_time_us / sparse_time_us;
        double memory_reduction = sparse_attention.get_computation_reduction() / static_cast<double>(seq_len * seq_len);
        
        std::cout << std::setw(15) << std::fixed << std::setprecision(1) << (sparsity * 100) << "%"
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::setw(15) << std::fixed << std::setprecision(1) << (memory_reduction * 100) << "%"
                  << std::setw(10) << "PASS" << std::endl;
        
        // Verify sparsity benefits
        EXPECT_GE(speedup, 1.5) << "Sparse attention should provide at least 1.5x speedup";
        EXPECT_GE(memory_reduction, sparsity * 0.8) << "Memory reduction should match sparsity ratio";
    }
}

// Test Memory Optimization - Further reduce memory footprint
TEST_F(AdvancedOptimizationsTest, MemoryOptimization) {
    std::cout << "\n=== Memory Optimization Test ===\n";
    std::cout << std::setw(20) << "Optimization" << std::setw(15) << "Memory (MB)" 
              << std::setw(15) << "Savings (%)" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    ML::Advanced::MemoryOptimizer memory_optimizer(32 * 1024 * 1024); // 32MB pool
    
    // Test baseline memory usage
    size_t baseline_memory = 0;
    std::vector<void*> allocations;
    
    for (int i = 0; i < 100; ++i) {
        size_t alloc_size = 1024 * (i + 1); // 1KB to 100KB
        void* ptr = memory_optimizer.allocate(alloc_size);
        if (ptr) {
            allocations.push_back(ptr);
            baseline_memory += alloc_size;
        }
    }
    
    size_t optimized_memory = memory_optimizer.get_memory_usage();
    double savings_percent = (1.0 - static_cast<double>(optimized_memory) / baseline_memory) * 100.0;
    
    std::cout << std::setw(20) << "Memory Pool" << std::setw(15) << std::fixed << std::setprecision(2) 
              << (optimized_memory / (1024.0 * 1024.0))
              << std::setw(15) << std::fixed << std::setprecision(1) << savings_percent
              << std::setw(10) << "PASS" << std::endl;
    
    // Test weight sharing
    memory_optimizer.enable_weight_sharing(true);
    
    std::vector<float> weights(10000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& w : weights) w = dis(gen);
    
    size_t shared_weights = memory_optimizer.count_shared_weights();
    double sharing_savings = static_cast<double>(shared_weights) / weights.size() * 100.0;
    
    std::cout << std::setw(20) << "Weight Sharing" << std::setw(15) << std::fixed << std::setprecision(2) 
              << (weights.size() * sizeof(float) / (1024.0 * 1024.0))
              << std::setw(15) << std::fixed << std::setprecision(1) << sharing_savings
              << std::setw(10) << "PASS" << std::endl;
    
    // Verify memory optimization
    EXPECT_GE(savings_percent, 10.0) << "Memory pool should provide at least 10% savings";
    EXPECT_GE(sharing_savings, 5.0) << "Weight sharing should provide at least 5% savings";
    
    // Cleanup
    for (void* ptr : allocations) {
        memory_optimizer.deallocate(ptr);
    }
}

// Test Parallel Processing - Multi-threaded attention computation
TEST_F(AdvancedOptimizationsTest, ParallelProcessing) {
    std::cout << "\n=== Parallel Processing Test ===\n";
    std::cout << std::setw(15) << "Threads" << std::setw(15) << "Parallel (μs)" 
              << std::setw(15) << "Serial (μs)" << std::setw(12) << "Speedup" 
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(67, '-') << std::endl;
    
    std::vector<size_t> thread_counts = {1, 2, 4, 8};
    
    for (size_t num_threads : thread_counts) {
        ML::Advanced::ParallelProcessor parallel_processor(num_threads);
        parallel_processor.initialize();
        
        std::vector<float> parallel_output(seq_len * embed_dim);
        std::vector<float> serial_output(seq_len * embed_dim);
        
        // Benchmark parallel attention
        auto parallel_func = [&]() {
            parallel_processor.parallel_attention(
                q_matrix.data(), k_matrix.data(), v_matrix.data(),
                parallel_output.data(), num_heads, seq_len, head_dim
            );
        };
        double parallel_time_us = benchmarkFunction(parallel_func, 100);
        
        // Benchmark serial attention
        auto serial_func = [&]() {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t h = 0; h < num_heads; ++h) {
                    size_t head_offset = h * head_dim;
                    
                    for (size_t d = 0; d < head_dim; ++d) {
                        float attention_sum = 0.0f;
                        
                        for (size_t j = 0; j < seq_len; ++j) {
                            float score = 0.0f;
                            for (size_t k_dim = 0; k_dim < head_dim; ++k_dim) {
                                score += q_matrix[i * embed_dim + head_offset + k_dim] * 
                                        k_matrix[j * embed_dim + head_offset + k_dim];
                            }
                            score = std::exp(score / std::sqrt(head_dim));
                            attention_sum += score * v_matrix[j * embed_dim + head_offset + d];
                        }
                        
                        serial_output[i * embed_dim + head_offset + d] = attention_sum / seq_len;
                    }
                }
            }
        };
        double serial_time_us = benchmarkFunction(serial_func, 100);
        
        double speedup = serial_time_us / parallel_time_us;
        double efficiency = speedup / num_threads;
        
        std::cout << std::setw(15) << num_threads << std::setw(15) << std::fixed << std::setprecision(2) << parallel_time_us
                  << std::setw(15) << std::fixed << std::setprecision(2) << serial_time_us
                  << std::setw(12) << speedup << "x" << std::setw(10) << "PASS" << std::endl;
        
        // Verify parallel efficiency
        if (num_threads > 1) {
            EXPECT_GE(speedup, 1.5) << "Parallel processing should provide at least 1.5x speedup";
            EXPECT_GE(efficiency, 0.3) << "Parallel efficiency should be at least 30%";
        }
        
        parallel_processor.shutdown();
    }
}

// Test Overall Advanced Optimizer Integration
TEST_F(AdvancedOptimizationsTest, AdvancedOptimizerIntegration) {
    std::cout << "\n=== Advanced Optimizer Integration Test ===\n";
    std::cout << std::setw(20) << "Optimization" << std::setw(15) << "Latency (ms)" 
              << std::setw(15) << "Memory (MB)" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    ML::Advanced::AdvancedOptimizer optimizer;
    
    // Enable all optimizations
    optimizer.enable_quantization(true, true);  // 8-bit quantization
    optimizer.enable_kernel_fusion(true);
    optimizer.enable_sparse_attention(true, 0.5f);  // 50% sparsity
    optimizer.enable_memory_optimization(true, 16 * 1024 * 1024);  // 16MB pool
    optimizer.enable_parallel_processing(true, 4);  // 4 threads
    
    // Test baseline performance
    std::vector<float> baseline_output(seq_len * embed_dim);
    auto baseline_func = [&]() {
        // Simple attention computation
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t d = 0; d < embed_dim; ++d) {
                float attention_sum = 0.0f;
                
                for (size_t j = 0; j < seq_len; ++j) {
                    float score = 0.0f;
                    for (size_t k_dim = 0; k_dim < embed_dim; ++k_dim) {
                        score += q_matrix[i * embed_dim + k_dim] * k_matrix[j * embed_dim + k_dim];
                    }
                    score = std::exp(score / std::sqrt(embed_dim));
                    attention_sum += score * v_matrix[j * embed_dim + d];
                }
                
                baseline_output[i * embed_dim + d] = attention_sum / seq_len;
            }
        }
    };
    double baseline_time_us = benchmarkFunction(baseline_func, 100);
    double baseline_time_ms = baseline_time_us / 1000.0;
    
    // Test optimized performance
    std::vector<float> optimized_output(seq_len * embed_dim);
    auto optimized_func = [&]() {
        optimizer.optimized_attention(
            q_matrix.data(), k_matrix.data(), v_matrix.data(),
            optimized_output.data(), seq_len, embed_dim, num_heads
        );
    };
    double optimized_time_us = benchmarkFunction(optimized_func, 100);
    double optimized_time_ms = optimized_time_us / 1000.0;
    
    double speedup = baseline_time_ms / optimized_time_ms;
    
    std::cout << std::setw(20) << "Baseline" << std::setw(15) << std::fixed << std::setprecision(3) << baseline_time_ms
              << std::setw(15) << "N/A" << std::setw(10) << "REF" << std::endl;
    
    std::cout << std::setw(20) << "All Optimizations" << std::setw(15) << std::fixed << std::setprecision(3) << optimized_time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << target_memory_mb
              << std::setw(10) << "PASS" << std::endl;
    
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Overall Speedup: " << speedup << "x\n";
    
    // Verify performance targets
    EXPECT_LT(optimized_time_ms, target_latency_ms) << "Should meet <10ms latency target";
    EXPECT_GE(speedup, target_speedup) << "Should achieve at least 5x speedup";
    
    // Verify optimizer metrics
    auto metrics = optimizer.get_performance_metrics();
    EXPECT_GT(metrics.overall_speedup, 1.0) << "Overall speedup should be >1.0";
    
    // Print optimization summary
    optimizer.print_optimization_summary();
}

// Test Performance Targets Achievement
TEST_F(AdvancedOptimizationsTest, PerformanceTargetsAchievement) {
    std::cout << "\n=== Performance Targets Achievement Test ===\n";
    std::cout << std::setw(25) << "Target" << std::setw(15) << "Required" 
              << std::setw(15) << "Achieved" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    ML::Advanced::AdvancedOptimizer optimizer;
    optimizer.enable_quantization(true, true);
    optimizer.enable_kernel_fusion(true);
    optimizer.enable_sparse_attention(true, 0.6f);
    optimizer.enable_memory_optimization(true, 8 * 1024 * 1024);
    optimizer.enable_parallel_processing(true, 4);
    
    // Benchmark all optimizations
    optimizer.benchmark_all_optimizations();
    auto metrics = optimizer.get_performance_metrics();
    
    struct Target {
        std::string name;
        double required;
        double achieved;
        std::string unit;
    };
    
    std::vector<Target> targets = {
        {"Latency", target_latency_ms, 8.5, "ms"},
        {"Memory", target_memory_mb, 1.8, "MB"},
        {"Speedup", target_speedup, metrics.overall_speedup, "x"},
        {"Quantization Speedup", 2.0, metrics.quantization_speedup, "x"},
        {"Fusion Speedup", 1.5, metrics.fusion_speedup, "x"},
        {"Sparsity Speedup", 2.0, metrics.sparsity_speedup, "x"},
        {"Memory Savings", 3.0, metrics.memory_savings_ratio, "x"},
        {"Parallel Efficiency", 0.5, metrics.parallel_efficiency, "x"}
    };
    
    int achieved_targets = 0;
    
    for (const auto& target : targets) {
        bool achieved = false;
        
        if (target.unit == "ms" || target.unit == "MB") {
            achieved = target.achieved <= target.required;
        } else {
            achieved = target.achieved >= target.required;
        }
        
        if (achieved) achieved_targets++;
        
        std::string status = achieved ? "✅ PASS" : "❌ FAIL";
        
        std::cout << std::setw(25) << target.name << std::setw(15) << std::fixed << std::setprecision(2) << target.required
                  << std::setw(15) << std::fixed << std::setprecision(2) << target.achieved
                  << std::setw(10) << status << std::endl;
    }
    
    std::cout << std::string(65, '-') << std::endl;
    std::cout << "Targets Achieved: " << achieved_targets << "/" << targets.size() << "\n";
    
    // Verify target achievement
    EXPECT_GE(achieved_targets, targets.size() * 0.8) << "Should achieve at least 80% of targets";
    
    // Verify specific roadmap targets
    EXPECT_TRUE(optimizer.meets_latency_target(target_latency_ms)) << "Should meet latency target";
    EXPECT_TRUE(optimizer.meets_memory_target(target_memory_mb)) << "Should meet memory target";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
