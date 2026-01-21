//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * XSIMD-based Vector Operations Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "XSIMDOperations.h"
#include <cstring>
#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace ML {
namespace XSIMD {

// Vector Operations Implementation
void VectorOps::vector_add_scalar(const float* src, float scalar, float* dst, size_t size) {
    const batch_type scalar_batch(scalar);
    size_t aligned_size = size - (size % batch_size);
    
    // Process aligned data with SIMD
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        batch_type result_batch = data_batch + scalar_batch;
        result_batch.store_unaligned(&dst[i]);
    }
    
    // Handle remaining elements
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = src[i] + scalar;
    }
}

void VectorOps::vector_mul_scalar(const float* src, float scalar, float* dst, size_t size) {
    const batch_type scalar_batch(scalar);
    size_t aligned_size = size - (size % batch_size);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        batch_type result_batch = data_batch * scalar_batch;
        result_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = src[i] * scalar;
    }
}

void VectorOps::vector_add_vector(const float* src1, const float* src2, float* dst, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data1_batch = xsimd::load_unaligned(&src1[i]);
        batch_type data2_batch = xsimd::load_unaligned(&src2[i]);
        batch_type result_batch = data1_batch + data2_batch;
        result_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

void VectorOps::vector_mul_vector(const float* src1, const float* src2, float* dst, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data1_batch = xsimd::load_unaligned(&src1[i]);
        batch_type data2_batch = xsimd::load_unaligned(&src2[i]);
        batch_type result_batch = data1_batch * data2_batch;
        result_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

void VectorOps::matrix_vector_multiply(const float* matrix, const float* vector, 
                                      float* result, size_t rows, size_t cols) {
    // For each row in matrix
    for (size_t i = 0; i < rows; ++i) {
        const float* matrix_row = &matrix[i * cols];
        float sum = 0.0f;

        size_t aligned_cols = cols - (cols % batch_size);

        // Process aligned columns with SIMD, reduce per-batch in scalar order
        // to minimize numeric drift vs. the reference scalar loop.
        for (size_t j = 0; j < aligned_cols; j += batch_size) {
            batch_type matrix_batch = xsimd::load_unaligned(&matrix_row[j]);
            batch_type vector_batch = xsimd::load_unaligned(&vector[j]);
            batch_type prod_batch = matrix_batch * vector_batch;
            alignas(64) float prod_values[batch_size];
            prod_batch.store_unaligned(prod_values);
            for (size_t k = 0; k < batch_size; ++k) {
                sum += prod_values[k];
            }
        }

        // Handle remaining columns
        for (size_t j = aligned_cols; j < cols; ++j) {
            sum += matrix_row[j] * vector[j];
        }

        result[i] = sum;
    }
}

void VectorOps::tanh_batch(const float* src, float* dst, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        batch_type result_batch = xsimd::tanh(data_batch);
        result_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = std::tanh(src[i]);
    }
}

void VectorOps::relu_batch(const float* src, float* dst, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    const batch_type zero_batch(0.0f);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        batch_type result_batch = xsimd::max(data_batch, zero_batch);
        result_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = std::max(0.0f, src[i]);
    }
}

void VectorOps::sigmoid_batch(const float* src, float* dst, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    const batch_type one_batch(1.0f);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        batch_type neg_data = -data_batch;
        batch_type exp_neg = xsimd::exp(neg_data);
        batch_type result_batch = one_batch / (one_batch + exp_neg);
        result_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}

void VectorOps::gelu_batch(const float* src, float* dst, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    const batch_type one_batch(1.0f);
    const batch_type sqrt_2_over_pi_batch(0.7978845608f); // sqrt(2/pi)
    const batch_type coeff_batch(0.044715f);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type x_batch = xsimd::load_unaligned(&src[i]);
        batch_type x_cubed = x_batch * x_batch * x_batch;
        batch_type tanh_arg = sqrt_2_over_pi_batch * x_batch * (one_batch + coeff_batch * x_cubed);
        batch_type tanh_result = xsimd::tanh(tanh_arg);
        batch_type result_batch = x_batch * 0.5f * (one_batch + tanh_result);
        result_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        float x = src[i];
        dst[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * x * (1.0f + 0.044715f * x * x * x)));
    }
}

float VectorOps::dot_product(const float* vec1, const float* vec2, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    batch_type sum_batch = batch_type(0.0f);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data1_batch = xsimd::load_unaligned(&vec1[i]);
        batch_type data2_batch = xsimd::load_unaligned(&vec2[i]);
        batch_type prod_batch = data1_batch * data2_batch;
        sum_batch += prod_batch;
    }
    
    float sum = xsimd::reduce_add(sum_batch);
    
    // Handle remaining elements
    for (size_t i = aligned_size; i < size; ++i) {
        sum += vec1[i] * vec2[i];
    }
    
    return sum;
}

void VectorOps::copy_vector(const float* src, float* dst, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        data_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = src[i];
    }
}

void VectorOps::fill_vector(float* dst, float value, size_t size) {
    const batch_type value_batch(value);
    size_t aligned_size = size - (size % batch_size);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        value_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = value;
    }
}

float VectorOps::reduce_sum(const float* src, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    batch_type sum_batch = batch_type(0.0f);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        sum_batch += data_batch;
    }
    
    float sum = xsimd::reduce_add(sum_batch);
    
    for (size_t i = aligned_size; i < size; ++i) {
        sum += src[i];
    }
    
    return sum;
}

float VectorOps::reduce_max(const float* src, size_t size) {
    size_t aligned_size = size - (size % batch_size);
    batch_type max_batch = xsimd::load_unaligned(&src[0]);
    
    for (size_t i = batch_size; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        max_batch = xsimd::max(max_batch, data_batch);
    }
    
    float max_val = xsimd::reduce_max(max_batch);
    
    for (size_t i = aligned_size; i < size; ++i) {
        max_val = std::max(max_val, src[i]);
    }
    
    return max_val;
}

void VectorOps::softmax(const float* src, float* dst, size_t size) {
    // Find max for numerical stability
    float max_val = reduce_max(src, size);
    
    // Compute exp and sum
    std::vector<float> exp_values(size);
    float sum = 0.0f;
    
    size_t aligned_size = size - (size % batch_size);
    const batch_type max_batch(max_val);
    batch_type sum_batch = batch_type(0.0f);
    
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type data_batch = xsimd::load_unaligned(&src[i]);
        batch_type shifted = data_batch - max_batch;
        batch_type exp_batch = xsimd::exp(shifted);
        exp_batch.store_unaligned(&exp_values[i]);
        sum_batch += exp_batch;
    }
    
    float exp_sum = xsimd::reduce_add(sum_batch);
    
    // Handle remaining elements
    for (size_t i = aligned_size; i < size; ++i) {
        exp_values[i] = std::exp(src[i] - max_val);
        exp_sum += exp_values[i];
    }
    
    // Normalize
    const batch_type sum_inv_batch(1.0f / exp_sum);
    for (size_t i = 0; i < aligned_size; i += batch_size) {
        batch_type exp_batch = xsimd::load_unaligned(&exp_values[i]);
        batch_type result_batch = exp_batch * sum_inv_batch;
        result_batch.store_unaligned(&dst[i]);
    }
    
    for (size_t i = aligned_size; i < size; ++i) {
        dst[i] = exp_values[i] / exp_sum;
    }
}

// XSIMDVector Implementation
XSIMDVector::XSIMDVector(size_t size) : data(size) {}

XSIMDVector::XSIMDVector(const std::vector<float>& vec) : data(vec) {}

XSIMDVector XSIMDVector::operator+(float scalar) const {
    XSIMDVector result(data.size());
    VectorOps::vector_add_scalar(data.data(), scalar, result.data.data(), data.size());
    return result;
}

XSIMDVector XSIMDVector::operator*(float scalar) const {
    XSIMDVector result(data.size());
    VectorOps::vector_mul_scalar(data.data(), scalar, result.data.data(), data.size());
    return result;
}

XSIMDVector XSIMDVector::operator+(const XSIMDVector& other) const {
    XSIMDVector result(data.size());
    VectorOps::vector_add_vector(data.data(), other.data.data(), result.data.data(), data.size());
    return result;
}

XSIMDVector XSIMDVector::operator*(const XSIMDVector& other) const {
    XSIMDVector result(data.size());
    VectorOps::vector_mul_vector(data.data(), other.data.data(), result.data.data(), data.size());
    return result;
}

XSIMDVector& XSIMDVector::operator+=(float scalar) {
    VectorOps::vector_add_scalar(data.data(), scalar, data.data(), data.size());
    return *this;
}

XSIMDVector& XSIMDVector::operator*=(float scalar) {
    VectorOps::vector_mul_scalar(data.data(), scalar, data.data(), data.size());
    return *this;
}

void XSIMDVector::apply_tanh() {
    VectorOps::tanh_batch(data.data(), data.data(), data.size());
}

void XSIMDVector::apply_relu() {
    VectorOps::relu_batch(data.data(), data.data(), data.size());
}

void XSIMDVector::apply_softmax() {
    VectorOps::softmax(data.data(), data.data(), data.size());
}

float XSIMDVector::dot_product(const XSIMDVector& other) const {
    return VectorOps::dot_product(data.data(), other.data.data(), data.size());
}

float XSIMDVector::sum() const {
    return VectorOps::reduce_sum(data.data(), data.size());
}

float XSIMDVector::max() const {
    return VectorOps::reduce_max(data.data(), data.size());
}

float XSIMDVector::mean() const {
    return sum() / static_cast<float>(data.size());
}

// XSIMDAttention Implementation
XSIMDAttention::XSIMDAttention(size_t d_model, size_t n_heads) 
    : d_model_(d_model), n_heads_(n_heads), head_dim_(d_model / n_heads) {
    
    // Initialize weights (simplified - in practice would be random initialization)
    q_weight_.resize(d_model_ * d_model_);
    k_weight_.resize(d_model_ * d_model_);
    v_weight_.resize(d_model_ * d_model_);
    o_weight_.resize(d_model_ * d_model_);
    
    // Initialize buffers
    q_buffer_.resize(d_model_);
    k_buffer_.resize(d_model_);
    v_buffer_.resize(d_model_);
    attention_buffer_.resize(1); // Will be resized as needed
}

void XSIMDAttention::forward(const std::vector<float>& input, 
                             std::vector<float>& output) {
    // Simple forward pass - in practice would compute QKV projections
    output = input; // Placeholder
}

// Performance Benchmarking Implementation
template<typename Func>
double XSIMDPerformance::benchmark_function(Func&& func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return static_cast<double>(duration.count()) / iterations;
}

std::vector<XSIMDPerformance::BenchmarkResult> XSIMDPerformance::run_benchmarks() {
    std::vector<BenchmarkResult> results;
    
    // Test different sizes
    std::vector<size_t> test_sizes = {64, 256, 512, 1024, 4096};
    
    // Generate test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t size : test_sizes) {
        std::vector<float> data1(size), data2(size), result(size);
        
        for (size_t i = 0; i < size; ++i) {
            data1[i] = dis(gen);
            data2[i] = dis(gen);
        }
        
        // Benchmark vector addition
        double scalar_time = benchmark_function([&]() {
            for (size_t i = 0; i < size; ++i) {
                result[i] = data1[i] + data2[i];
            }
        });
        
        double xsimd_time = benchmark_function([&]() {
            VectorOps::vector_add_vector(data1.data(), data2.data(), result.data(), size);
        });
        
        results.push_back({scalar_time, xsimd_time, scalar_time / xsimd_time, size, "vector_add"});
        
        // Benchmark dot product
        scalar_time = benchmark_function([&]() {
            float sum = 0.0f;
            for (size_t i = 0; i < size; ++i) {
                sum += data1[i] * data2[i];
            }
        });
        
        xsimd_time = benchmark_function([&]() {
            VectorOps::dot_product(data1.data(), data2.data(), size);
        });
        
        results.push_back({scalar_time, xsimd_time, scalar_time / xsimd_time, size, "dot_product"});
    }
    
    return results;
}

void XSIMDPerformance::print_benchmark_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=== XSIMD Performance Benchmarks ===\n";
    std::cout << std::setw(12) << "Operation" << std::setw(8) << "Size" 
              << std::setw(15) << "Scalar (μs)" << std::setw(15) << "XSIMD (μs)" 
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(12) << result.operation << std::setw(8) << result.data_size
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.scalar_time_us
                  << std::setw(15) << result.xsimd_time_us 
                  << std::setw(11) << result.speedup << "x" << std::endl;
    }
}

} // namespace XSIMD
} // namespace ML
