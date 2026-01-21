//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * XSIMD-based Vector Operations for Cross-Platform SIMD
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef XSIMD_OPERATIONS_H
#define XSIMD_OPERATIONS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <xsimd/xsimd.hpp>

namespace ML {
namespace XSIMD {

using batch_type = xsimd::batch<float>;
constexpr std::size_t batch_size = batch_type::size;

// Cross-platform SIMD vector operations using xsimd
class VectorOps {
public:
    // Vector-scalar operations
    static void vector_add_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_sub_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_mul_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_div_scalar(const float* src, float scalar, float* dst, size_t size);
    
    // Vector-vector operations
    static void vector_add_vector(const float* src1, const float* src2, float* dst, size_t size);
    static void vector_sub_vector(const float* src1, const float* src2, float* dst, size_t size);
    static void vector_mul_vector(const float* src1, const float* src2, float* dst, size_t size);
    static void vector_div_vector(const float* src1, const float* src2, float* dst, size_t size);
    
    // Matrix-vector multiplication (optimized for feed-forward)
    static void matrix_vector_multiply(const float* matrix, const float* vector, 
                                      float* result, size_t rows, size_t cols);
    
    // Batched activation functions
    static void tanh_batch(const float* src, float* dst, size_t size);
    static void relu_batch(const float* src, float* dst, size_t size);
    static void sigmoid_batch(const float* src, float* dst, size_t size);
    static void gelu_batch(const float* src, float* dst, size_t size);
    
    // Dot product for gradient calculations
    static float dot_product(const float* vec1, const float* vec2, size_t size);
    
    // Memory operations
    static void copy_vector(const float* src, float* dst, size_t size);
    static void fill_vector(float* dst, float value, size_t size);
    
    // Reduction operations
    static float reduce_sum(const float* src, size_t size);
    static float reduce_max(const float* src, size_t size);
    static float reduce_min(const float* src, size_t size);
    
    // Neural network specific operations
    static void softmax(const float* src, float* dst, size_t size);
    static void layer_norm(const float* input, float* output, 
                          const float* gamma, const float* beta, 
                          size_t size, float epsilon = 1e-6f);
};

// High-level vector wrapper using xsimd
class XSIMDVector {
private:
    std::vector<float> data;
    
public:
    XSIMDVector(size_t size = 0);
    XSIMDVector(const std::vector<float>& vec);
    
    // Element access
    float& operator[](size_t index) { return data[index]; }
    const float& operator[](size_t index) const { return data[index]; }
    
    // Vector operations
    XSIMDVector operator+(float scalar) const;
    XSIMDVector operator-(float scalar) const;
    XSIMDVector operator*(float scalar) const;
    XSIMDVector operator/(float scalar) const;
    
    XSIMDVector operator+(const XSIMDVector& other) const;
    XSIMDVector operator-(const XSIMDVector& other) const;
    XSIMDVector operator*(const XSIMDVector& other) const;
    XSIMDVector operator/(const XSIMDVector& other) const;
    
    // In-place operations
    XSIMDVector& operator+=(float scalar);
    XSIMDVector& operator*=(float scalar);
    XSIMDVector& operator+=(const XSIMDVector& other);
    XSIMDVector& operator*=(const XSIMDVector& other);
    
    // Neural network specific operations
    void apply_tanh();
    void apply_relu();
    void apply_sigmoid();
    void apply_gelu();
    void apply_softmax();
    float dot_product(const XSIMDVector& other) const;
    
    // Reduction operations
    float sum() const;
    float max() const;
    float min() const;
    float mean() const;
    
    // Utility
    size_t size() const { return data.size(); }
    float* data_ptr() { return data.data(); }
    const float* data_ptr() const { return data.data(); }
    void resize(size_t new_size) { data.resize(new_size); }
    void fill(float value) { std::fill(data.begin(), data.end(), value); }
    
    // Conversion
    std::vector<float> to_std_vector() const { return data; }
    
    // SIMD info
    static std::size_t simd_batch_size() { return batch_size; }
    static bool has_simd_support() { return batch_size > 1; }
};

// Attention mechanism using xsimd
class XSIMDAttention {
public:
    XSIMDAttention(size_t d_model, size_t n_heads);
    
    // Efficient attention computation
    void forward(const std::vector<float>& input, 
                std::vector<float>& output);
    
    // Scaled dot-product attention
    void scaled_dot_product_attention(const std::vector<float>& q,
                                     const std::vector<float>& k,
                                     const std::vector<float>& v,
                                     std::vector<float>& output,
                                     size_t seq_len);
    
private:
    size_t d_model_;
    size_t n_heads_;
    size_t head_dim_;
    
    // Weight matrices
    std::vector<float> q_weight_;
    std::vector<float> k_weight_;
    std::vector<float> v_weight_;
    std::vector<float> o_weight_;
    
    // Temporary buffers
    mutable std::vector<float> q_buffer_;
    mutable std::vector<float> k_buffer_;
    mutable std::vector<float> v_buffer_;
    mutable std::vector<float> attention_buffer_;
    
    // Internal operations
    void compute_qkv(const float* input, float* q, float* k, float* v) const;
};

// Transformer block using xsimd
class XSIMDTransformerBlock {
public:
    XSIMDTransformerBlock(size_t d_model, size_t n_heads, size_t d_ff, float dropout = 0.1f);
    
    void forward(const std::vector<float>& input, 
                std::vector<float>& output);
    
    // Memory-efficient forward pass
    void memory_efficient_forward(const std::vector<float>& input,
                                  std::vector<float>& output);
    
private:
    std::unique_ptr<XSIMDAttention> attention_;
    std::vector<float> ff_weight1_;
    std::vector<float> ff_weight2_;
    std::vector<float> ff_bias1_;
    std::vector<float> ff_bias2_;
    std::vector<float> norm1_weight_;
    std::vector<float> norm1_bias_;
    std::vector<float> norm2_weight_;
    std::vector<float> norm2_bias_;
    
    // Temporary buffers
    std::vector<float> attention_output_;
    std::vector<float> ff_output_;
    std::vector<float> norm_output_;
    
    // Internal operations
    void feed_forward(const float* input, float* output) const;
    void layer_norm(const float* input, const float* weight, const float* bias,
                   float* output, size_t size) const;
};

// Performance monitoring
class XSIMDPerformance {
public:
    struct BenchmarkResult {
        double scalar_time_us;
        double xsimd_time_us;
        double speedup;
        size_t data_size;
        std::string operation;
    };
    
    static std::vector<BenchmarkResult> run_benchmarks();
    static void print_benchmark_results(const std::vector<BenchmarkResult>& results);
    
private:
    template<typename Func>
    static double benchmark_function(Func&& func, int iterations = 1000);
};

} // namespace XSIMD
} // namespace ML

#endif // XSIMD_OPERATIONS_H
