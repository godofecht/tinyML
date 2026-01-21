//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * SIMD Performance Benchmarking Suite
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include "SIMDOperations.h"
#include "VectorOperations.h"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

class SIMDBenchmark : public ::testing::Test {
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
    
    template<typename Func>
    double benchmark_function(Func&& func, int iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return static_cast<double>(duration.count()) / iterations;
    }
    
    std::vector<size_t> test_sizes;
    std::map<size_t, std::vector<float>> test_data;
    std::map<size_t, std::vector<float>> test_data2;
    std::map<size_t, std::vector<float>> result_data;
};

// Benchmark vector-scalar addition
TEST_F(SIMDBenchmark, VectorScalarAddition) {
    std::cout << "\n=== Vector-Scalar Addition Benchmark ===\n";
    std::cout << std::setw(10) << "Size" << std::setw(15) << "Scalar (μs)" 
              << std::setw(15) << "SIMD (μs)" << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    for (size_t size : test_sizes) {
        const float scalar = 2.0f;
        
        // Benchmark scalar implementation
        double scalar_time = benchmark_function([&]() {
            ML::Scalar::vector_add_scalar(test_data[size].data(), scalar, result_data[size].data(), size);
        });
        
        // Benchmark SIMD implementation
        double simd_time = benchmark_function([&]() {
            ML::SIMD::VectorOps::vector_add_scalar(test_data[size].data(), scalar, result_data[size].data(), size);
        });
        
        double speedup = scalar_time / simd_time;
        
        std::cout << std::setw(10) << size << std::setw(15) << std::fixed << std::setprecision(2) << scalar_time
                  << std::setw(15) << simd_time << std::setw(12) << speedup << "x" << std::endl;
        
        // Verify results are the same
        std::vector<float> scalar_result(size);
        std::vector<float> simd_result(size);
        
        ML::Scalar::vector_add_scalar(test_data[size].data(), scalar, scalar_result.data(), size);
        ML::SIMD::VectorOps::vector_add_scalar(test_data[size].data(), scalar, simd_result.data(), size);
        
        for (size_t i = 0; i < size; ++i) {
            ASSERT_NEAR(scalar_result[i], simd_result[i], 1e-6f);
        }
    }
}

// Benchmark vector-vector multiplication
TEST_F(SIMDBenchmark, VectorVectorMultiplication) {
    std::cout << "\n=== Vector-Vector Multiplication Benchmark ===\n";
    std::cout << std::setw(10) << "Size" << std::setw(15) << "Scalar (μs)" 
              << std::setw(15) << "SIMD (μs)" << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    for (size_t size : test_sizes) {
        // Benchmark scalar implementation
        double scalar_time = benchmark_function([&]() {
            ML::Scalar::vector_mul_vector(test_data[size].data(), test_data2[size].data(), 
                                         result_data[size].data(), size);
        });
        
        // Benchmark SIMD implementation
        double simd_time = benchmark_function([&]() {
            ML::SIMD::VectorOps::vector_mul_vector(test_data[size].data(), test_data2[size].data(), 
                                                 result_data[size].data(), size);
        });
        
        double speedup = scalar_time / simd_time;
        
        std::cout << std::setw(10) << size << std::setw(15) << std::fixed << std::setprecision(2) << scalar_time
                  << std::setw(15) << simd_time << std::setw(12) << speedup << "x" << std::endl;
        
        // Verify results
        std::vector<float> scalar_result(size);
        std::vector<float> simd_result(size);
        
        ML::Scalar::vector_mul_vector(test_data[size].data(), test_data2[size].data(), scalar_result.data(), size);
        ML::SIMD::VectorOps::vector_mul_vector(test_data[size].data(), test_data2[size].data(), simd_result.data(), size);
        
        for (size_t i = 0; i < size; ++i) {
            ASSERT_NEAR(scalar_result[i], simd_result[i], 1e-6f);
        }
    }
}

// Benchmark matrix-vector multiplication (neural network feed-forward)
TEST_F(SIMDBenchmark, MatrixVectorMultiplication) {
    std::cout << "\n=== Matrix-Vector Multiplication Benchmark ===\n";
    std::cout << std::setw(12) << "Matrix" << std::setw(15) << "Scalar (μs)" 
              << std::setw(15) << "SIMD (μs)" << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    std::vector<std::pair<size_t, size_t>> matrix_sizes = {
        {64, 64}, {128, 128}, {256, 256}, {512, 512}, {256, 1024}, {1024, 256}
    };
    
    for (auto [rows, cols] : matrix_sizes) {
        // Create test matrix and vector
        std::vector<float> matrix(rows * cols);
        std::vector<float> vector(cols);
        std::vector<float> result_scalar(rows);
        std::vector<float> result_simd(rows);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < rows * cols; ++i) matrix[i] = dis(gen);
        for (size_t i = 0; i < cols; ++i) vector[i] = dis(gen);
        
        // Benchmark scalar implementation
        double scalar_time = benchmark_function([&]() {
            ML::Scalar::matrix_vector_multiply(matrix.data(), vector.data(), result_scalar.data(), rows, cols);
        }, 100); // Fewer iterations for larger operations
        
        // Benchmark SIMD implementation
        double simd_time = benchmark_function([&]() {
            ML::SIMD::VectorOps::matrix_vector_multiply(matrix.data(), vector.data(), result_simd.data(), rows, cols);
        }, 100);
        
        double speedup = scalar_time / simd_time;
        
        std::cout << std::setw(5) << rows << "x" << std::setw(6) << cols 
                  << std::setw(15) << std::fixed << std::setprecision(2) << scalar_time
                  << std::setw(15) << simd_time << std::setw(12) << speedup << "x" << std::endl;
        
        // Verify results
        for (size_t i = 0; i < rows; ++i) {
            ASSERT_NEAR(result_scalar[i], result_simd[i], 1e-5f);
        }
    }
}

// Benchmark activation functions
TEST_F(SIMDBenchmark, ActivationFunctions) {
    std::cout << "\n=== Tanh Activation Function Benchmark ===\n";
    std::cout << std::setw(10) << "Size" << std::setw(15) << "Scalar (μs)" 
              << std::setw(15) << "SIMD (μs)" << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    for (size_t size : test_sizes) {
        // Benchmark scalar implementation
        double scalar_time = benchmark_function([&]() {
            ML::Scalar::tanh_batch(test_data[size].data(), result_data[size].data(), size);
        });
        
        // Benchmark SIMD implementation
        double simd_time = benchmark_function([&]() {
            ML::SIMD::VectorOps::tanh_batch(test_data[size].data(), result_data[size].data(), size);
        });
        
        double speedup = scalar_time / simd_time;
        
        std::cout << std::setw(10) << size << std::setw(15) << std::fixed << std::setprecision(2) << scalar_time
                  << std::setw(15) << simd_time << std::setw(12) << speedup << "x" << std::endl;
        
        // Verify results
        std::vector<float> scalar_result(size);
        std::vector<float> simd_result(size);
        
        ML::Scalar::tanh_batch(test_data[size].data(), scalar_result.data(), size);
        ML::SIMD::VectorOps::tanh_batch(test_data[size].data(), simd_result.data(), size);
        
        for (size_t i = 0; i < size; ++i) {
            ASSERT_NEAR(scalar_result[i], simd_result[i], 1e-5f);
        }
    }
}

// Benchmark dot product
TEST_F(SIMDBenchmark, DotProduct) {
    std::cout << "\n=== Dot Product Benchmark ===\n";
    std::cout << std::setw(10) << "Size" << std::setw(15) << "Scalar (μs)" 
              << std::setw(15) << "SIMD (μs)" << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    for (size_t size : test_sizes) {
        // Benchmark scalar implementation
        float scalar_result = 0.0f;
        double scalar_time = benchmark_function([&]() {
            scalar_result = ML::Scalar::dot_product(test_data[size].data(), test_data2[size].data(), size);
        });
        
        // Benchmark SIMD implementation
        float simd_result = 0.0f;
        double simd_time = benchmark_function([&]() {
            simd_result = ML::SIMD::VectorOps::dot_product(test_data[size].data(), test_data2[size].data(), size);
        });
        
        double speedup = scalar_time / simd_time;
        
        std::cout << std::setw(10) << size << std::setw(15) << std::fixed << std::setprecision(2) << scalar_time
                  << std::setw(15) << simd_time << std::setw(12) << speedup << "x" << std::endl;
        
        // Verify results
        ASSERT_NEAR(scalar_result, simd_result, 1e-5f);
    }
}

// Benchmark original VectorOperations.h vs new SIMD implementation
TEST_F(SIMDBenchmark, OriginalVsNewImplementation) {
    std::cout << "\n=== Original vs New Implementation Comparison ===\n";
    std::cout << std::setw(10) << "Size" << std::setw(18) << "Original (μs)" 
              << std::setw(15) << "New SIMD (μs)" << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    for (size_t size : test_sizes) {
        // Benchmark original implementation
        double original_time = benchmark_function([&]() {
            auto result = test_data[size] + 2.0f; // Uses original operator+
        });
        
        // Benchmark new SIMD implementation
        double new_time = benchmark_function([&]() {
            ML::SIMD::SIMDVector vec(test_data[size]);
            auto result = vec + 2.0f;
        });
        
        double speedup = original_time / new_time;
        
        std::cout << std::setw(10) << size << std::setw(18) << std::fixed << std::setprecision(2) << original_time
                  << std::setw(15) << new_time << std::setw(12) << speedup << "x" << std::endl;
    }
}

// Memory bandwidth test
TEST_F(SIMDBenchmark, MemoryBandwidth) {
    std::cout << "\n=== Memory Bandwidth Test ===\n";
    std::cout << std::setw(10) << "Size" << std::setw(15) << "Copy (GB/s)" 
              << std::setw(15) << "Add (GB/s)" << std::setw(15) << "Mul (GB/s)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (size_t size : test_sizes) {
        size_t bytes = size * sizeof(float);
        
        // Copy bandwidth
        double copy_time = benchmark_function([&]() {
            ML::SIMD::VectorOps::copy_vector(test_data[size].data(), result_data[size].data(), size);
        });
        double copy_bandwidth = (bytes * 1e-6) / copy_time; // MB/s
        
        // Add bandwidth (reads 2 arrays, writes 1)
        double add_time = benchmark_function([&]() {
            ML::SIMD::VectorOps::vector_add_vector(test_data[size].data(), test_data2[size].data(), 
                                                 result_data[size].data(), size);
        });
        double add_bandwidth = (bytes * 3e-6) / add_time; // 3x memory traffic
        
        // Mul bandwidth
        double mul_time = benchmark_function([&]() {
            ML::SIMD::VectorOps::vector_mul_vector(test_data[size].data(), test_data2[size].data(), 
                                                 result_data[size].data(), size);
        });
        double mul_bandwidth = (bytes * 3e-6) / mul_time;
        
        std::cout << std::setw(10) << size << std::setw(15) << std::fixed << std::setprecision(1) << copy_bandwidth
                  << std::setw(15) << add_bandwidth << std::setw(15) << mul_bandwidth << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
