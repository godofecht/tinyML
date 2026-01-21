//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * ARM NEON SIMD Operations for Mobile Devices
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef NEON_OPERATIONS_H
#define NEON_OPERATIONS_H

#include <vector>
#include <arm_neon.h>
#include <algorithm>
#include <cmath>

namespace ML {
namespace NEON {

// ARM NEON-optimized vector operations for mobile devices
class VectorOps {
public:
    // Vector-scalar operations with NEON
    static void vector_add_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_sub_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_mul_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_div_scalar(const float* src, float scalar, float* dst, size_t size);
    
    // Vector-vector operations with NEON
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
    
    // Dot product for gradient calculations
    static float dot_product(const float* vec1, const float* vec2, size_t size);
    
    // Memory operations
    static void copy_vector(const float* src, float* dst, size_t size);
    static void fill_vector(float* dst, float value, size_t size);
    
    // Specialized neural network operations
    static void weighted_sum(const float* inputs, const float* weights, 
                           float* output, size_t size);
    static void apply_dropout(float* data, float dropout_rate, size_t size);
    static void layer_norm(const float* input, float* output, 
                          const float* gamma, const float* beta, 
                          size_t size, float epsilon);
};

// High-level NEON vector wrapper
class NEONVector {
private:
    std::vector<float> data;
    
public:
    NEONVector(size_t size = 0);
    NEONVector(const std::vector<float>& vec);
    
    // Element access
    float& operator[](size_t index) { return data[index]; }
    const float& operator[](size_t index) const { return data[index]; }
    
    // Vector operations
    NEONVector operator+(float scalar) const;
    NEONVector operator-(float scalar) const;
    NEONVector operator*(float scalar) const;
    NEONVector operator/(float scalar) const;
    
    NEONVector operator+(const NEONVector& other) const;
    NEONVector operator-(const NEONVector& other) const;
    NEONVector operator*(const NEONVector& other) const;
    NEONVector operator/(const NEONVector& other) const;
    
    // Neural network specific operations
    void apply_tanh();
    void apply_relu();
    void apply_sigmoid();
    float dot_product(const NEONVector& other) const;
    
    // Utility
    size_t size() const { return data.size(); }
    float* data_ptr() { return data.data(); }
    const float* data_ptr() const { return data.data(); }
    void resize(size_t new_size) { data.resize(new_size); }
    
    // Conversion
    std::vector<float> to_std_vector() const { return data; }
};

// Mobile-optimized attention mechanisms
class MobileAttention {
public:
    MobileAttention(size_t d_model, size_t n_heads);
    
    // Efficient attention computation for mobile
    void forward(const std::vector<float>& input, 
                std::vector<float>& output);
    
    // Lightweight attention with reduced memory footprint
    void lightweight_attention(const std::vector<float>& q,
                              const std::vector<float>& k,
                              const std::vector<float>& v,
                              std::vector<float>& output);
    
private:
    size_t d_model_;
    size_t n_heads_;
    size_t head_dim_;
    
    // Quantized weights for mobile efficiency
    std::vector<int8_t> q_weight_q_;
    std::vector<int8_t> k_weight_q_;
    std::vector<int8_t> v_weight_q_;
    std::vector<float> scales_;
    
    // NEON-optimized attention computation
    void compute_qkv_neon(const float* input, float* q, float* k, float* v);
    void scaled_dot_product_neon(const float* q, const float* k, const float* v,
                                float* output, size_t seq_len);
};

// Mobile-optimized transformer block
class MobileTransformerBlock {
public:
    MobileTransformerBlock(size_t d_model, size_t n_heads, size_t d_ff);
    
    void forward(const std::vector<float>& input, 
                std::vector<float>& output);
    
    // Memory-efficient forward pass for mobile
    void memory_efficient_forward(const std::vector<float>& input,
                                  std::vector<float>& output);
    
private:
    std::unique_ptr<MobileAttention> attention_;
    std::unique_ptr<NEONVector> feed_forward_weights_;
    std::unique_ptr<NEONVector> layer_norm_weights_;
    
    // Temporary buffers to avoid allocations
    std::vector<float> attention_buffer_;
    std::vector<float> ff_buffer_;
    std::vector<float> norm_buffer_;
};

// Quantization support for mobile inference
class QuantizedOps {
public:
    // 8-bit quantized matrix multiplication
    static void quantized_matmul(const int8_t* A, const int8_t* B, int32_t* C,
                               float scale_a, float scale_b, float scale_c,
                               size_t M, size_t N, size_t K);
    
    // 8-bit to 32-bit dequantization
    static void dequantize(const int8_t* quantized, float* dequantized,
                          float scale, size_t size);
    
    // 32-bit to 8-bit quantization
    static void quantize(const float* input, int8_t* output,
                        float& scale, size_t size);
    
    // Quantized activation functions
    static void quantized_relu(const int8_t* input, int8_t* output, size_t size);
    static void quantized_tanh_lut(const int8_t* input, int8_t* output, size_t size);
};

// Power management for mobile devices
class PowerManager {
public:
    static void set_performance_level(int level); // 0-3 (power saving to max performance)
    static int get_current_performance_level();
    static bool is_thermal_throttling_active();
    static void enable_adaptive_scaling(bool enable);
    
    // Monitor power consumption
    static float get_current_power_mw();
    static float get_average_power_mw();
    
private:
    static int current_performance_level_;
    static bool adaptive_scaling_enabled_;
};

} // namespace NEON
} // namespace ML

#endif // NEON_OPERATIONS_H
