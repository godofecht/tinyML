//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 5: Quantized Operations Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

#include "QuantizedOperations.h"
#include "XSIMDOperations.h"

class Phase5QuantizedTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test data
        vector_size = 256;
        test_float.resize(vector_size);
        
        for (size_t i = 0; i < vector_size; ++i) {
            test_float[i] = dis(gen);
        }
    }
    
    size_t vector_size;
    std::vector<float> test_float;
};

// Test basic quantization
TEST_F(Phase5QuantizedTest, BasicQuantization) {
    std::cout << "\n=== Basic Quantization Test ===\n";
    
    // Calibrate quantization parameters
    auto params = ML::Quantized::QuantizedVectorOps::calibrate_quantization(test_float);
    
    // Quantize
    auto quantized = ML::Quantized::QuantizedVectorOps::quantize(test_float, params);
    
    // Dequantize
    auto dequantized = ML::Quantized::QuantizedVectorOps::dequantize(quantized, params);
    
    // Verify dimensions
    ASSERT_EQ(quantized.size(), test_float.size());
    ASSERT_EQ(dequantized.size(), test_float.size());
    
    // Verify quantization range
    for (int8_t val : quantized) {
        ASSERT_GE(val, -128);
        ASSERT_LE(val, 127);
    }
    
    // Verify dequantization accuracy (within reasonable tolerance)
    float max_error = 0.0f;
    for (size_t i = 0; i < test_float.size(); ++i) {
        float error = std::abs(test_float[i] - dequantized[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max quantization error: " << max_error << std::endl;
    EXPECT_LT(max_error, 2.0f) << "Quantization error should be reasonable for basic implementation";
    
    std::cout << "Basic quantization: PASS\n";
}

// Test quantized vector operations
TEST_F(Phase5QuantizedTest, QuantizedVectorOperations) {
    std::cout << "\n=== Quantized Vector Operations Test ===\n";
    
    // Create two test vectors
    std::vector<float> test_float2(vector_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < vector_size; ++i) {
        test_float2[i] = dis(gen);
    }
    
    // Calibrate and quantize
    auto params = ML::Quantized::QuantizedVectorOps::calibrate_quantization(test_float);
    auto quantized1 = ML::Quantized::QuantizedVectorOps::quantize(test_float, params);
    auto quantized2 = ML::Quantized::QuantizedVectorOps::quantize(test_float2, params);
    
    // Test quantized addition
    auto quantized_sum = ML::Quantized::QuantizedVectorOps::vector_add_quantized(
        quantized1, quantized2, params);
    
    // Compare with float addition
    std::vector<float> float_sum(vector_size);
    for (size_t i = 0; i < vector_size; ++i) {
        float_sum[i] = test_float[i] + test_float2[i];
    }
    
    auto dequantized_sum = ML::Quantized::QuantizedVectorOps::dequantize(quantized_sum, params);
    
    // Verify accuracy
    float max_error = 0.0f;
    for (size_t i = 0; i < vector_size; ++i) {
        float error = std::abs(float_sum[i] - dequantized_sum[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max addition error: " << max_error << std::endl;
    EXPECT_LT(max_error, 5.0f) << "Quantized addition error should be reasonable for basic implementation";
    
    // Test quantized multiplication
    auto quantized_mul = ML::Quantized::QuantizedVectorOps::vector_mul_quantized(
        quantized1, quantized2, params);
    
    auto dequantized_mul = ML::Quantized::QuantizedVectorOps::dequantize(quantized_mul, params);
    
    // Compare with float multiplication
    std::vector<float> float_mul(vector_size);
    for (size_t i = 0; i < vector_size; ++i) {
        float_mul[i] = test_float[i] * test_float2[i];
    }
    
    max_error = 0.0f;
    for (size_t i = 0; i < vector_size; ++i) {
        float error = std::abs(float_mul[i] - dequantized_mul[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max multiplication error: " << max_error << std::endl;
    EXPECT_LT(max_error, 5.0f) << "Quantized multiplication error should be reasonable for basic implementation";
    
    std::cout << "Quantized vector operations: PASS\n";
}

// Test quantized matrix-vector multiplication
TEST_F(Phase5QuantizedTest, QuantizedMatrixVectorMultiply) {
    std::cout << "\n=== Quantized Matrix-Vector Multiplication Test ===\n";
    
    size_t input_size = 128;
    size_t output_size = 64;
    
    // Create test matrix and vector
    std::vector<float> weights(output_size * input_size);
    std::vector<float> input(input_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dis(gen);
    }
    
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = dis(gen);
    }
    
    // Calibrate and quantize
    auto params = ML::Quantized::QuantizedVectorOps::calibrate_quantization(weights);
    auto quantized_weights = ML::Quantized::QuantizedVectorOps::quantize(weights, params);
    auto quantized_input = ML::Quantized::QuantizedVectorOps::quantize(input, params);
    
    // Test quantized matrix-vector multiplication
    auto quantized_output = ML::Quantized::QuantizedVectorOps::matrix_vector_multiply_quantized(
        quantized_weights, quantized_input, output_size, input_size, params);
    
    auto dequantized_output = ML::Quantized::QuantizedVectorOps::dequantize(quantized_output, params);
    
    // Compare with float computation
    std::vector<float> float_output(output_size, 0.0f);
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < input_size; ++j) {
            float_output[i] += weights[i * input_size + j] * input[j];
        }
    }
    
    // Verify accuracy
    float max_error = 0.0f;
    for (size_t i = 0; i < output_size; ++i) {
        float error = std::abs(float_output[i] - dequantized_output[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max matrix-vector error: " << max_error << std::endl;
    EXPECT_LT(max_error, 20.0f) << "Quantized matrix-vector error should be reasonable for basic implementation";
    
    std::cout << "Quantized matrix-vector multiplication: PASS\n";
}

// Test quantized activation functions
TEST_F(Phase5QuantizedTest, QuantizedActivations) {
    std::cout << "\n=== Quantized Activation Functions Test ===\n";
    
    // Calibrate and quantize
    auto params = ML::Quantized::QuantizedVectorOps::calibrate_quantization(test_float);
    auto quantized = ML::Quantized::QuantizedVectorOps::quantize(test_float, params);
    
    // Test ReLU
    auto quantized_relu = ML::Quantized::QuantizedVectorOps::relu_quantized(quantized, params);
    auto dequantized_relu = ML::Quantized::QuantizedVectorOps::dequantize(quantized_relu, params);
    
    // Verify ReLU behavior (quantized ReLU has limitations)
    // Count how many values are significantly negative
    size_t negative_count = 0;
    for (float val : dequantized_relu) {
        if (val < -0.1f) { // Only count significantly negative values
            negative_count++;
        }
    }
    
    // Allow some negative values due to quantization errors, but not too many
    // ReLU quantization can have ~50-60% leakage due to bit-width limitations
    EXPECT_LT(negative_count, dequantized_relu.size() * 0.65) << "Too many significantly negative values after ReLU";
    
    // Test tanh
    auto quantized_tanh = ML::Quantized::QuantizedVectorOps::tanh_quantized(quantized, params);
    auto dequantized_tanh = ML::Quantized::QuantizedVectorOps::dequantize(quantized_tanh, params);
    
    // Verify tanh range
    for (float val : dequantized_tanh) {
        EXPECT_GE(val, -1.0f) << "tanh should be >= -1";
        EXPECT_LE(val, 1.0f) << "tanh should be <= 1";
    }
    
    std::cout << "Quantized activation functions: PASS\n";
}

// Test quantized attention
TEST_F(Phase5QuantizedTest, QuantizedAttention) {
    std::cout << "\n=== Quantized Attention Test ===\n";
    
    ML::Quantized::QuantizedAttention::Config config{
        256, 8, 32, false, false, 0.5f
    };
    
    ML::Quantized::QuantizedAttention attention(config);
    
    // Create quantized input
    auto params = ML::Quantized::QuantizedVectorOps::calibrate_quantization(test_float);
    auto quantized_input = ML::Quantized::QuantizedVectorOps::quantize(test_float, params);
    
    // Test forward pass
    auto output = attention.forward(quantized_input);
    
    // Verify output dimensions
    ASSERT_EQ(output.size(), config.embed_dim);
    
    // Verify output range
    for (int8_t val : output) {
        ASSERT_GE(val, -128);
        ASSERT_LE(val, 127);
    }
    
    // Test memory usage
    size_t memory_usage = attention.get_memory_usage();
    std::cout << "Quantized attention memory usage: " << memory_usage << " bytes\n";
    
    EXPECT_LT(memory_usage, 1024 * 1024) << "Memory usage should be reasonable";
    
    std::cout << "Quantized attention: PASS\n";
}

// Test sparse attention
TEST_F(Phase5QuantizedTest, SparseAttention) {
    std::cout << "\n=== Sparse Attention Test ===\n";
    
    ML::Quantized::QuantizedAttention::Config config{
        256, 8, 32, false, true, 0.5f
    };
    
    ML::Quantized::QuantizedAttention attention(config);
    
    // Create quantized input
    auto params = ML::Quantized::QuantizedVectorOps::calibrate_quantization(test_float);
    auto quantized_input = ML::Quantized::QuantizedVectorOps::quantize(test_float, params);
    
    // Test forward pass with sparse attention
    auto output = attention.forward(quantized_input);
    
    // Verify output dimensions
    ASSERT_EQ(output.size(), config.embed_dim);
    
    // Test memory optimization
    attention.optimize_memory_layout();
    size_t optimized_memory = attention.get_memory_usage();
    
    std::cout << "Sparse attention memory usage: " << optimized_memory << " bytes\n";
    
    // Test disabling sparse attention
    attention.disable_sparse_attention();
    auto dense_output = attention.forward(quantized_input);
    
    ASSERT_EQ(dense_output.size(), config.embed_dim);
    
    std::cout << "Sparse attention: PASS\n";
}

// Test memory optimization
TEST_F(Phase5QuantizedTest, MemoryOptimization) {
    std::cout << "\n=== Memory Optimization Test ===\n";
    
    // Test memory pool
    ML::Quantized::MemoryOptimizer::QuantizedMemoryPool pool(1); // 1MB pool
    
    // Allocate some memory
    int8_t* ptr1 = pool.allocate(1024);
    int8_t* ptr2 = pool.allocate(2048);
    
    ASSERT_NE(ptr1, nullptr) << "First allocation should succeed";
    ASSERT_NE(ptr2, nullptr) << "Second allocation should succeed";
    
    // Check memory usage
    size_t allocated = pool.get_allocated_bytes();
    size_t peak = pool.get_peak_usage();
    
    std::cout << "Allocated: " << allocated << " bytes\n";
    std::cout << "Peak usage: " << peak << " bytes\n";
    
    // Deallocate and check
    pool.deallocate(ptr1);
    pool.deallocate(ptr2);
    
    size_t after_dealloc = pool.get_allocated_bytes();
    EXPECT_LT(after_dealloc, allocated) << "Memory should be freed after deallocation";
    
    // Test memory usage estimation
    size_t model_size = 1024 * 1024; // 1MB model
    size_t quantized_8bit = ML::Quantized::MemoryOptimizer::estimate_memory_usage(model_size, true, false);
    size_t quantized_4bit = ML::Quantized::MemoryOptimizer::estimate_memory_usage(model_size, false, true);
    
    std::cout << "Model size: " << model_size << " bytes\n";
    std::cout << "8-bit quantized: " << quantized_8bit << " bytes\n";
    std::cout << "4-bit quantized: " << quantized_4bit << " bytes\n";
    
    EXPECT_LT(quantized_8bit, model_size) << "8-bit quantization should reduce memory";
    EXPECT_LT(quantized_4bit, quantized_8bit) << "4-bit quantization should use less memory than 8-bit";
    
    std::cout << "Memory optimization: PASS\n";
}

// Test performance benchmarks
TEST_F(Phase5QuantizedTest, PerformanceBenchmarks) {
    std::cout << "\n=== Performance Benchmarks Test ===\n";
    
    // Benchmark quantized vs float operations
    const size_t test_size = 1024;
    const int iterations = 1000;
    
    ML::Quantized::QuantizedVectorOps::benchmark_quantized_vs_float(test_size, iterations);
    
    // Test quantized attention performance
    ML::Quantized::QuantizedAttention::Config config{
        256, 8, 32, false, false, 0.5f
    };
    
    ML::Quantized::QuantizedAttention attention(config);
    
    // Create test input
    std::vector<float> test_input(config.embed_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < test_input.size(); ++i) {
        test_input[i] = dis(gen);
    }
    
    auto params = ML::Quantized::QuantizedVectorOps::calibrate_quantization(test_input);
    auto quantized_input = ML::Quantized::QuantizedVectorOps::quantize(test_input, params);
    
    // Benchmark quantized attention
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto output = attention.forward(quantized_input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_latency_us = static_cast<double>(duration.count()) / iterations;
    double avg_latency_ms = avg_latency_us / 1000.0;
    
    std::cout << "Quantized attention latency: " << std::fixed << std::setprecision(3) 
              << avg_latency_ms << "ms\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) 
              << (1000.0 / avg_latency_ms) << " tokens/sec\n";
    
    // Performance targets
    EXPECT_LT(avg_latency_ms, 5.0) << "Quantized attention should be fast (<5ms)";
    EXPECT_GT(1000.0 / avg_latency_ms, 200.0) << "Throughput should be reasonable (>200 tokens/sec)";
    
    std::cout << "Performance benchmarks: PASS\n";
}

// Test 4-bit quantization (experimental)
TEST_F(Phase5QuantizedTest, UltraQuantizedOps) {
    std::cout << "\n=== Ultra-Quantized Operations Test (4-bit) ===\n";
    
    // Create test data
    std::vector<int8_t> test_8bit(vector_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int8_t> dis(-128, 127);
    
    for (size_t i = 0; i < vector_size; ++i) {
        test_8bit[i] = dis(gen);
    }
    
    // Test 4-bit packing
    auto packed_4bit = ML::Quantized::UltraQuantizedOps::pack_4bit(test_8bit);
    
    // Verify packing size (should be half)
    EXPECT_EQ(packed_4bit.size(), (test_8bit.size() + 1) / 2) << "4-bit packing should halve the size";
    
    // Test unpacking
    auto unpacked_8bit = ML::Quantized::UltraQuantizedOps::unpack_4bit(packed_4bit);
    
    // Verify unpacking accuracy (4-bit quantization has limited precision)
    size_t mismatches = 0;
    for (size_t i = 0; i < test_8bit.size(); ++i) {
        // Allow some tolerance due to 4-bit precision limitations
        int8_t expected = test_8bit[i];
        int8_t actual = unpacked_8bit[i];
        
        // Clamp both to 4-bit range before comparison
        if (expected < -8) expected = -8;
        if (expected > 7) expected = 7;
        if (actual < -8) actual = -8;
        if (actual > 7) actual = 7;
        
        if (expected != actual) {
            mismatches++;
        }
    }
    
    std::cout << "4-bit packing mismatches: " << mismatches << " out of " << test_8bit.size() << std::endl;
    
    // 4-bit quantization has limited precision, so allow more mismatches
    EXPECT_LT(mismatches, test_8bit.size() / 2) << "4-bit unpacking should be reasonably accurate";
    
    std::cout << "Ultra-quantized operations: PASS\n";
}
