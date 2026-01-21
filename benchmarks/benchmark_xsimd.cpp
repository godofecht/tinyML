//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * XSIMD Performance Benchmarking Suite
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include "XSIMDOperations.h"
#include <random>
#include <iostream>
#include <iomanip>

class XSIMDBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test different sizes to show SIMD benefits
        test_sizes = {64, 256, 512, 1024, 4096, 8192};
        
        for (size_t size : test_sizes) {
            test_data[size] = std::vector<float>(size);
            test_data2[size] = std::vector<float>(size);
            result_data[size] = std::vector<float>(size);
            
            for (size_t i = 0; i < size; ++i) {
                test_data[size][i] = dis(gen);
                test_data2[size][i] = dis(gen);
            }
        }
    }
    
    std::vector<size_t> test_sizes;
    std::map<size_t, std::vector<float>> test_data;
    std::map<size_t, std::vector<float>> test_data2;
    std::map<size_t, std::vector<float>> result_data;
};

// Test XSIMD functionality
TEST_F(XSIMDBenchmark, BasicOperations) {
    std::cout << "\n=== XSIMD Basic Operations Test ===\n";
    std::cout << "SIMD batch size: " << ML::XSIMD::XSIMDVector::simd_batch_size() << std::endl;
    std::cout << "Has SIMD support: " << (ML::XSIMD::XSIMDVector::has_simd_support() ? "Yes" : "No") << std::endl;
    
    // Test vector operations
    for (size_t size : test_sizes) {
        ML::XSIMD::XSIMDVector vec1(test_data[size]);
        ML::XSIMD::XSIMDVector vec2(test_data2[size]);
        
        // Test addition
        auto result = vec1 + vec2;
        ASSERT_EQ(result.size(), size);
        
        // Test scalar multiplication
        auto scaled = vec1 * 2.0f;
        ASSERT_EQ(scaled.size(), size);
        
        // Test dot product
        float dot = vec1.dot_product(vec2);
        
        // Verify against scalar implementation
        float expected_dot = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            expected_dot += test_data[size][i] * test_data2[size][i];
        }
        ASSERT_NEAR(dot, expected_dot, 1e-5f);
    }
}

// Test activation functions
TEST_F(XSIMDBenchmark, ActivationFunctions) {
    std::cout << "\n=== XSIMD Activation Functions Test ===\n";
    
    for (size_t size : test_sizes) {
        ML::XSIMD::XSIMDVector vec(test_data[size]);
        
        // Test tanh
        vec.apply_tanh();
        
        // Verify all values are in valid range [-1, 1]
        for (size_t i = 0; i < size; ++i) {
            ASSERT_GE(vec[i], -1.0f);
            ASSERT_LE(vec[i], 1.0f);
        }
        
        // Test ReLU
        vec.apply_relu();
        
        // Verify all values are non-negative
        for (size_t i = 0; i < size; ++i) {
            ASSERT_GE(vec[i], 0.0f);
        }
        
        // Test softmax
        vec.apply_softmax();
        
        // Verify softmax properties (all positive, sum to 1)
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            ASSERT_GT(vec[i], 0.0f);
            sum += vec[i];
        }
        ASSERT_NEAR(sum, 1.0f, 1e-5f);
    }
}

// Test matrix-vector multiplication
TEST_F(XSIMDBenchmark, MatrixVectorMultiplication) {
    std::cout << "\n=== XSIMD Matrix-Vector Multiplication Test ===\n";
    
    std::vector<std::pair<size_t, size_t>> matrix_sizes = {
        {64, 64}, {128, 128}, {256, 256}, {512, 512}
    };
    
    for (auto [rows, cols] : matrix_sizes) {
        // Create test matrix and vector
        std::vector<float> matrix(rows * cols);
        std::vector<float> vector(cols);
        std::vector<float> result(rows);
        std::vector<float> expected_result(rows);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < rows * cols; ++i) matrix[i] = dis(gen);
        for (size_t i = 0; i < cols; ++i) vector[i] = dis(gen);
        
        // XSIMD implementation
        ML::XSIMD::VectorOps::matrix_vector_multiply(matrix.data(), vector.data(), 
                                                     result.data(), rows, cols);
        
        // Scalar implementation for verification
        for (size_t i = 0; i < rows; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                sum += matrix[i * cols + j] * vector[j];
            }
            expected_result[i] = sum;
        }
        
        // Verify results
        for (size_t i = 0; i < rows; ++i) {
            ASSERT_NEAR(result[i], expected_result[i], 1e-5f);
        }
        
        std::cout << "Matrix " << rows << "x" << cols << ": PASSED" << std::endl;
    }
}

// Test attention mechanism
TEST_F(XSIMDBenchmark, AttentionMechanism) {
    std::cout << "\n=== XSIMD Attention Mechanism Test ===\n";
    
    size_t d_model = 256;
    size_t n_heads = 8;
    size_t seq_len = 64;
    
    ML::XSIMD::XSIMDAttention attention(d_model, n_heads);
    
    // Create test input (simplified)
    std::vector<float> input(seq_len * d_model);
    std::vector<float> output(seq_len * d_model);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < seq_len * d_model; ++i) {
        input[i] = dis(gen);
    }
    
    // Test attention computation
    attention.forward(input, output);
    
    // Verify output size and reasonable values
    ASSERT_EQ(output.size(), seq_len * d_model);
    
    for (float val : output) {
        ASSERT_TRUE(std::isfinite(val));
    }
    
    std::cout << "Attention mechanism (" << d_model << "x" << n_heads << " heads): PASSED" << std::endl;
}

// Performance comparison
TEST_F(XSIMDBenchmark, PerformanceComparison) {
    std::cout << "\n=== XSIMD Performance Comparison ===\n";
    
    auto results = ML::XSIMD::XSIMDPerformance::run_benchmarks();
    ML::XSIMD::XSIMDPerformance::print_benchmark_results(results);
    
    // Verify we got results for all test sizes
    ASSERT_GT(results.size(), 0);
    
    // Check that we have speedup data
    bool has_speedup = false;
    for (const auto& result : results) {
        if (result.speedup > 1.0) {
            has_speedup = true;
            break;
        }
    }
    
    if (ML::XSIMD::XSIMDVector::has_simd_support()) {
        EXPECT_TRUE(has_speedup) << "Expected some speedup with SIMD support";
    }
}

// Test reduction operations
TEST_F(XSIMDBenchmark, ReductionOperations) {
    std::cout << "\n=== XSIMD Reduction Operations Test ===\n";
    
    for (size_t size : test_sizes) {
        ML::XSIMD::XSIMDVector vec(test_data[size]);
        
        // Test sum
        float xsimd_sum = vec.sum();
        float expected_sum = 0.0f;
        for (float val : test_data[size]) {
            expected_sum += val;
        }
        ASSERT_NEAR(xsimd_sum, expected_sum, 1e-4f);
        
        // Test max
        float xsimd_max = vec.max();
        float expected_max = *std::max_element(test_data[size].begin(), test_data[size].end());
        ASSERT_NEAR(xsimd_max, expected_max, 1e-5f);
        
        // Test mean
        float xsimd_mean = vec.mean();
        float expected_mean = expected_sum / static_cast<float>(size);
        ASSERT_NEAR(xsimd_mean, expected_mean, 1e-5f);
        
        std::cout << "Reductions for size " << size << ": PASSED" << std::endl;
    }
}

// Test softmax implementation
TEST_F(XSIMDBenchmark, SoftmaxImplementation) {
    std::cout << "\n=== XSIMD Softmax Test ===\n";
    
    for (size_t size : test_sizes) {
        std::vector<float> input(test_data[size]);
        std::vector<float> output(size);
        
        // Apply softmax
        ML::XSIMD::VectorOps::softmax(input.data(), output.data(), size);
        
        // Verify softmax properties
        float sum = 0.0f;
        for (float val : output) {
            ASSERT_GT(val, 0.0f);
            ASSERT_LT(val, 1.0f);
            sum += val;
        }
        ASSERT_NEAR(sum, 1.0f, 1e-5f);
        
        std::cout << "Softmax for size " << size << ": PASSED" << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
