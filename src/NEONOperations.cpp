//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * ARM NEON SIMD Operations Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "NEONOperations.h"
#include <cstring>

namespace ML {
namespace NEON {

// NEON Vector Operations Implementation
void VectorOps::vector_add_scalar(const float* src, float scalar, float* dst, size_t size) {
    const float32x4_t scalar_vec = vdupq_n_f32(scalar);
    size_t neon_size = size & ~3; // Process 4 elements at a time
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data = vld1q_f32(&src[i]);
        float32x4_t result = vaddq_f32(data, scalar_vec);
        vst1q_f32(&dst[i], result);
    }
    
    // Handle remaining elements
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = src[i] + scalar;
    }
}

void VectorOps::vector_mul_scalar(const float* src, float scalar, float* dst, size_t size) {
    const float32x4_t scalar_vec = vdupq_n_f32(scalar);
    size_t neon_size = size & ~3;
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data = vld1q_f32(&src[i]);
        float32x4_t result = vmulq_f32(data, scalar_vec);
        vst1q_f32(&dst[i], result);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = src[i] * scalar;
    }
}

void VectorOps::vector_add_vector(const float* src1, const float* src2, float* dst, size_t size) {
    size_t neon_size = size & ~3;
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data1 = vld1q_f32(&src1[i]);
        float32x4_t data2 = vld1q_f32(&src2[i]);
        float32x4_t result = vaddq_f32(data1, data2);
        vst1q_f32(&dst[i], result);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

void VectorOps::vector_mul_vector(const float* src1, const float* src2, float* dst, size_t size) {
    size_t neon_size = size & ~3;
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data1 = vld1q_f32(&src1[i]);
        float32x4_t data2 = vld1q_f32(&src2[i]);
        float32x4_t result = vmulq_f32(data1, data2);
        vst1q_f32(&dst[i], result);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

void VectorOps::matrix_vector_multiply(const float* matrix, const float* vector, 
                                      float* result, size_t rows, size_t cols) {
    // For each row in matrix
    for (size_t i = 0; i < rows; ++i) {
        const float* matrix_row = &matrix[i * cols];
        float sum = 0.0f;
        
        size_t neon_size = cols & ~3;
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        
        // Process 4 elements at a time
        for (size_t j = 0; j < neon_size; j += 4) {
            float32x4_t matrix_data = vld1q_f32(&matrix_row[j]);
            float32x4_t vector_data = vld1q_f32(&vector[j]);
            float32x4_t prod = vmulq_f32(matrix_data, vector_data);
            sum_vec = vaddq_f32(sum_vec, prod);
        }
        
        // Horizontal sum of the 4 floats
        float sum_array[4];
        vst1q_f32(sum_array, sum_vec);
        for (int k = 0; k < 4; ++k) {
            sum += sum_array[k];
        }
        
        // Handle remaining elements
        for (size_t j = neon_size; j < cols; ++j) {
            sum += matrix_row[j] * vector[j];
        }
        
        result[i] = sum;
    }
}

void VectorOps::tanh_batch(const float* src, float* dst, size_t size) {
    size_t neon_size = size & ~3;
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data = vld1q_f32(&src[i]);
        
        // Fast tanh approximation using NEON
        // tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
        float32x4_t x2 = vmulq_f32(data, data);
        float32x4_t twenty_seven = vdupq_n_f32(27.0f);
        float32x4_t nine = vdupq_n_f32(9.0f);
        
        float32x4_t numerator = vmulq_f32(data, vaddq_f32(twenty_seven, x2));
        float32x4_t denominator = vaddq_f32(twenty_seven, vmulq_f32(nine, x2));
        float32x4_t result = vdivq_f32(numerator, denominator);
        
        vst1q_f32(&dst[i], result);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = std::tanh(src[i]);
    }
}

void VectorOps::relu_batch(const float* src, float* dst, size_t size) {
    size_t neon_size = size & ~3;
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data = vld1q_f32(&src[i]);
        float32x4_t result = vmaxq_f32(data, zero_vec);
        vst1q_f32(&dst[i], result);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = std::max(0.0f, src[i]);
    }
}

void VectorOps::sigmoid_batch(const float* src, float* dst, size_t size) {
    size_t neon_size = size & ~3;
    float32x4_t one_vec = vdupq_n_f32(1.0f);
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data = vld1q_f32(&src[i]);
        
        // Fast sigmoid approximation: sigmoid(x) ≈ 0.5 * tanh(x/2) + 0.5
        float32x4_t half_data = vmulq_f32(data, vdupq_n_f32(0.5f));
        
        // Use tanh approximation
        float32x4_t x2 = vmulq_f32(half_data, half_data);
        float32x4_t twenty_seven = vdupq_n_f32(27.0f);
        float32x4_t nine = vdupq_n_f32(9.0f);
        
        float32x4_t numerator = vmulq_f32(half_data, vaddq_f32(twenty_seven, x2));
        float32x4_t denominator = vaddq_f32(twenty_seven, vmulq_f32(nine, x2));
        float32x4_t tanh_approx = vdivq_f32(numerator, denominator);
        
        float32x4_t result = vaddq_f32(vdupq_n_f32(0.5f), vmulq_f32(vdupq_n_f32(0.5f), tanh_approx));
        vst1q_f32(&dst[i], result);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}

float VectorOps::dot_product(const float* vec1, const float* vec2, size_t size) {
    size_t neon_size = size & ~3;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data1 = vld1q_f32(&vec1[i]);
        float32x4_t data2 = vld1q_f32(&vec2[i]);
        float32x4_t prod = vmulq_f32(data1, data2);
        sum_vec = vaddq_f32(sum_vec, prod);
    }
    
    // Horizontal sum
    float sum_array[4];
    vst1q_f32(sum_array, sum_vec);
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += sum_array[i];
    }
    
    // Handle remaining elements
    for (size_t i = neon_size; i < size; ++i) {
        sum += vec1[i] * vec2[i];
    }
    
    return sum;
}

void VectorOps::copy_vector(const float* src, float* dst, size_t size) {
    size_t neon_size = size & ~3;
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t data = vld1q_f32(&src[i]);
        vst1q_f32(&dst[i], data);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = src[i];
    }
}

void VectorOps::fill_vector(float* dst, float value, size_t size) {
    float32x4_t value_vec = vdupq_n_f32(value);
    size_t neon_size = size & ~3;
    
    for (size_t i = 0; i < neon_size; i += 4) {
        vst1q_f32(&dst[i], value_vec);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        dst[i] = value;
    }
}

void VectorOps::weighted_sum(const float* inputs, const float* weights, 
                           float* output, size_t size) {
    size_t neon_size = size & ~3;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t input_data = vld1q_f32(&inputs[i]);
        float32x4_t weight_data = vld1q_f32(&weights[i]);
        float32x4_t prod = vmulq_f32(input_data, weight_data);
        sum_vec = vaddq_f32(sum_vec, prod);
    }
    
    // Horizontal sum and store result
    float sum_array[4];
    vst1q_f32(sum_array, sum_vec);
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += sum_array[i];
    }
    
    // Handle remaining elements
    for (size_t i = neon_size; i < size; ++i) {
        sum += inputs[i] * weights[i];
    }
    
    output[0] = sum; // Assuming single output
}

void VectorOps::layer_norm(const float* input, float* output, 
                          const float* gamma, const float* beta, 
                          size_t size, float epsilon) {
    // Compute mean
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += input[i];
    }
    float mean = sum / size;
    
    // Compute variance
    float var_sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / size;
    float std_dev = std::sqrt(variance + epsilon);
    
    // Normalize and apply gamma/beta
    float32x4_t mean_vec = vdupq_n_f32(mean);
    float32x4_t std_dev_vec = vdupq_n_f32(std_dev);
    size_t neon_size = size & ~3;
    
    for (size_t i = 0; i < neon_size; i += 4) {
        float32x4_t input_data = vld1q_f32(&input[i]);
        float32x4_t gamma_data = vld1q_f32(&gamma[i]);
        float32x4_t beta_data = vld1q_f32(&beta[i]);
        
        float32x4_t normalized = vdivq_f32(vsubq_f32(input_data, mean_vec), std_dev_vec);
        float32x4_t result = vaddq_f32(vmulq_f32(normalized, gamma_data), beta_data);
        
        vst1q_f32(&output[i], result);
    }
    
    for (size_t i = neon_size; i < size; ++i) {
        output[i] = ((input[i] - mean) / std_dev) * gamma[i] + beta[i];
    }
}

// NEONVector Implementation
NEONVector::NEONVector(size_t size) : data(size) {}

NEONVector::NEONVector(const std::vector<float>& vec) : data(vec) {}

NEONVector NEONVector::operator+(float scalar) const {
    NEONVector result(data.size());
    VectorOps::vector_add_scalar(data.data(), scalar, result.data.data(), data.size());
    return result;
}

NEONVector NEONVector::operator*(float scalar) const {
    NEONVector result(data.size());
    VectorOps::vector_mul_scalar(data.data(), scalar, result.data.data(), data.size());
    return result;
}

NEONVector NEONVector::operator+(const NEONVector& other) const {
    NEONVector result(data.size());
    VectorOps::vector_add_vector(data.data(), other.data.data(), result.data.data(), data.size());
    return result;
}

NEONVector NEONVector::operator*(const NEONVector& other) const {
    NEONVector result(data.size());
    VectorOps::vector_mul_vector(data.data(), other.data.data(), result.data.data(), data.size());
    return result;
}

void NEONVector::apply_tanh() {
    VectorOps::tanh_batch(data.data(), data.data(), data.size());
}

void NEONVector::apply_relu() {
    VectorOps::relu_batch(data.data(), data.data(), data.size());
}

float NEONVector::dot_product(const NEONVector& other) const {
    return VectorOps::dot_product(data.data(), other.data.data(), data.size());
}

// MobileAttention Implementation
MobileAttention::MobileAttention(size_t d_model, size_t n_heads) 
    : d_model_(d_model), n_heads_(n_heads), head_dim_(d_model / n_heads) {
    // Initialize quantized weights (simplified)
    q_weight_q_.resize(d_model_ * d_model_);
    k_weight_q_.resize(d_model_ * d_model_);
    v_weight_q_.resize(d_model_ * d_model_);
    scales_.resize(3); // One scale per weight matrix
}

void MobileAttention::lightweight_attention(const std::vector<float>& q,
                                          const std::vector<float>& k,
                                          const std::vector<float>& v,
                                          std::vector<float>& output) {
    size_t seq_len = q.size() / d_model_;
    
    // Compute attention scores using NEON
    std::vector<float> scores(seq_len * seq_len);
    
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            float score = VectorOps::dot_product(&q[i * d_model_], &k[j * d_model_], d_model_);
            scores[i * seq_len + j] = score / std::sqrt(static_cast<float>(head_dim_));
        }
    }
    
    // Apply softmax (simplified for mobile)
    for (size_t i = 0; i < seq_len; ++i) {
        // Find max for numerical stability
        float max_score = scores[i * seq_len];
        for (size_t j = 1; j < seq_len; ++j) {
            max_score = std::max(max_score, scores[i * seq_len + j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < seq_len; ++j) {
            scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_score);
            sum += scores[i * seq_len + j];
        }
        
        // Normalize
        for (size_t j = 0; j < seq_len; ++j) {
            scores[i * seq_len + j] /= sum;
        }
    }
    
    // Apply attention to values
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            float weighted_sum = 0.0f;
            for (size_t k = 0; k < seq_len; ++k) {
                weighted_sum += scores[i * seq_len + k] * v[k * d_model_ + j];
            }
            output[i * d_model_ + j] = weighted_sum;
        }
    }
}

// QuantizedOps Implementation
void QuantizedOps::quantize(const float* input, int8_t* output, float& scale, size_t size) {
    // Find min and max values
    float min_val = input[0], max_val = input[0];
    for (size_t i = 1; i < size; ++i) {
        min_val = std::min(min_val, input[i]);
        max_val = std::max(max_val, input[i]);
    }
    
    // Calculate scale
    scale = (max_val - min_val) / 255.0f;
    if (scale < 1e-6f) scale = 1e-6f;
    
    // Quantize
    for (size_t i = 0; i < size; ++i) {
        output[i] = static_cast<int8_t>((input[i] - min_val) / scale);
    }
}

void QuantizedOps::dequantize(const int8_t* quantized, float* dequantized,
                             float scale, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dequantized[i] = static_cast<float>(quantized[i]) * scale;
    }
}

// PowerManager Implementation
int PowerManager::current_performance_level_ = 1;
bool PowerManager::adaptive_scaling_enabled_ = true;

void PowerManager::set_performance_level(int level) {
    current_performance_level_ = std::max(0, std::min(3, level));
    // In a real implementation, this would interface with the OS power management
}

int PowerManager::get_current_performance_level() {
    return current_performance_level_;
}

bool PowerManager::is_thermal_throttling_active() {
    // In a real implementation, this would check thermal sensors
    return false;
}

void PowerManager::enable_adaptive_scaling(bool enable) {
    adaptive_scaling_enabled_ = enable;
}

float PowerManager::get_current_power_mw() {
    // Placeholder implementation
    return 100.0f * (current_performance_level_ + 1);
}

float PowerManager::get_average_power_mw() {
    // Placeholder implementation
    return get_current_power_mw();
}

} // namespace NEON
} // namespace ML
