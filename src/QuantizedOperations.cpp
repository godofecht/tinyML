//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 5: Quantized Operations Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "QuantizedOperations.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <map>

namespace ML {
namespace Quantized {

// QuantizedVectorOps Implementation
QuantParams QuantizedVectorOps::calibrate_quantization(const std::vector<float>& weights) {
    if (weights.empty()) return QuantParams();
    
    float min_val = *std::min_element(weights.begin(), weights.end());
    float max_val = *std::max_element(weights.begin(), weights.end());
    
    // Avoid division by zero
    if (max_val == min_val) {
        return QuantParams(1.0f, 0, -128, 127);
    }
    
    float scale = (max_val - min_val) / 255.0f;
    int32_t zero_point = static_cast<int32_t>(-min_val / scale);
    
    // Clamp zero point to int8 range
    zero_point = std::max(-128, std::min(127, zero_point));
    
    return QuantParams(scale, zero_point, -128, 127);
}

std::vector<int8_t> QuantizedVectorOps::quantize(const std::vector<float>& input, const QuantParams& params) {
    std::vector<int8_t> quantized(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        float scaled = input[i] / params.scale + params.zero_point;
        int32_t clamped = std::max(-128, std::min(127, static_cast<int32_t>(scaled)));
        quantized[i] = static_cast<int8_t>(clamped);
    }
    
    return quantized;
}

std::vector<float> QuantizedVectorOps::dequantize(const std::vector<int8_t>& input, const QuantParams& params) {
    std::vector<float> dequantized(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        dequantized[i] = (static_cast<float>(input[i]) - params.zero_point) * params.scale;
    }
    
    return dequantized;
}

std::vector<int8_t> QuantizedVectorOps::vector_add_quantized(
    const std::vector<int8_t>& a, const std::vector<int8_t>& b, const QuantParams& params) {
    
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    
    std::vector<int8_t> result(a.size());
    
    for (size_t i = 0; i < a.size(); ++i) {
        int32_t sum = static_cast<int32_t>(a[i]) + static_cast<int32_t>(b[i]);
        sum = std::max(-128, std::min(127, sum));
        result[i] = static_cast<int8_t>(sum);
    }
    
    return result;
}

std::vector<int8_t> QuantizedVectorOps::vector_mul_quantized(
    const std::vector<int8_t>& a, const std::vector<int8_t>& b, const QuantParams& params) {
    
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    
    std::vector<int8_t> result(a.size());
    
    for (size_t i = 0; i < a.size(); ++i) {
        int32_t product = (static_cast<int32_t>(a[i]) - params.zero_point) * 
                         (static_cast<int32_t>(b[i]) - params.zero_point);
        product = product / 256 + params.zero_point; // Scale down to prevent overflow
        product = std::max(-128, std::min(127, product));
        result[i] = static_cast<int8_t>(product);
    }
    
    return result;
}

std::vector<int8_t> QuantizedVectorOps::matrix_vector_multiply_quantized(
    const std::vector<int8_t>& weights, const std::vector<int8_t>& input,
    size_t output_size, size_t input_size, const QuantParams& params) {
    
    if (weights.size() != output_size * input_size || input.size() != input_size) {
        throw std::invalid_argument("Matrix-vector dimensions don't match");
    }
    
    std::vector<int8_t> output(output_size, 0);
    
    for (size_t i = 0; i < output_size; ++i) {
        int32_t sum = 0;
        for (size_t j = 0; j < input_size; ++j) {
            int32_t w = static_cast<int32_t>(weights[i * input_size + j]) - params.zero_point;
            int32_t x = static_cast<int32_t>(input[j]) - params.zero_point;
            sum += w * x;
        }
        
        // Scale and quantize result
        sum = sum / 256 + params.zero_point; // Scale down
        sum = std::max(-128, std::min(127, sum));
        output[i] = static_cast<int8_t>(sum);
    }
    
    return output;
}

std::vector<int8_t> QuantizedVectorOps::relu_quantized(const std::vector<int8_t>& input, const QuantParams& params) {
    std::vector<int8_t> output(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        // Apply ReLU in quantized space: max(0, value)
        // Since we're working with int8_t, 0 is the zero point
        output[i] = std::max(static_cast<int8_t>(0), input[i]);
    }
    
    return output;
}

std::vector<int8_t> QuantizedVectorOps::tanh_quantized(const std::vector<int8_t>& input, const QuantParams& params) {
    std::vector<int8_t> output(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        // Dequantize, apply tanh, then re-quantize
        float dequant = (static_cast<float>(input[i]) - params.zero_point) * params.scale;
        float tanh_val = std::tanh(dequant);
        
        // Re-quantize
        float scaled = tanh_val / params.scale + params.zero_point;
        int32_t clamped = std::max(-128, std::min(127, static_cast<int32_t>(scaled)));
        output[i] = static_cast<int8_t>(clamped);
    }
    
    return output;
}

void QuantizedVectorOps::benchmark_quantized_vs_float(size_t vector_size, size_t iterations) {
    std::cout << "\n=== Quantized vs Float Benchmark ===\n";
    
    // Generate test data
    std::vector<float> float_a(vector_size), float_b(vector_size);
    std::vector<int8_t> quant_a(vector_size), quant_b(vector_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < vector_size; ++i) {
        float_a[i] = dis(gen);
        float_b[i] = dis(gen);
    }
    
    // Quantize
    QuantParams params = calibrate_quantization(float_a);
    quant_a = quantize(float_a, params);
    quant_b = quantize(float_b, params);
    
    // Benchmark float operations
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        std::vector<float> result(vector_size);
        for (size_t j = 0; j < vector_size; ++j) {
            result[j] = float_a[j] + float_b[j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto float_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark quantized operations
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        auto result = vector_add_quantized(quant_a, quant_b, params);
    }
    end = std::chrono::high_resolution_clock::now();
    auto quant_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Vector size: " << vector_size << ", Iterations: " << iterations << std::endl;
    std::cout << "Float time: " << float_time.count() << "μs" << std::endl;
    std::cout << "Quantized time: " << quant_time.count() << "μs" << std::endl;
    std::cout << "Speedup: " << static_cast<double>(float_time.count()) / quant_time.count() << "x" << std::endl;
}

// QuantizedAttention Implementation
QuantizedAttention::QuantizedAttention(const Config& config)
    : config_(config), sparse_enabled_(false) {
    
    initialize_quantized_weights();
    
    if (config_.sparse_attention) {
        enable_sparse_attention(config_.sparsity_ratio);
    }
}

void QuantizedAttention::initialize_quantized_weights() {
    // Initialize quantization parameters for different layers
    quant_params_.resize(4); // Q, K, V, O projections
    
    for (auto& params : quant_params_) {
        params = QuantParams(0.01f, 0, -128, 127); // Default parameters
    }
    
    // Initialize quantized weights (simplified)
    size_t qkv_size = config_.embed_dim * config_.embed_dim;
    q_weights_.resize(qkv_size, 0);
    k_weights_.resize(qkv_size, 0);
    v_weights_.resize(qkv_size, 0);
    o_weights_.resize(qkv_size, 0);
    
    attention_buffer_.resize(config_.embed_dim, 0);
}

std::vector<int8_t> QuantizedAttention::forward(const std::vector<int8_t>& input) {
    if (input.size() != config_.embed_dim) {
        throw std::invalid_argument("Input size mismatch");
    }
    
    // Compute QKV projections
    auto qkv = compute_qkv_quantized(input);
    
    // Compute attention
    std::vector<int8_t> q_part(qkv.begin(), qkv.begin() + config_.embed_dim);
    std::vector<int8_t> k_part(qkv.begin() + config_.embed_dim, qkv.begin() + 2 * config_.embed_dim);
    std::vector<int8_t> v_part(qkv.begin() + 2 * config_.embed_dim, qkv.end());
    
    auto attention_output = scaled_dot_product_quantized(q_part, k_part, v_part);
    
    return attention_output;
}

size_t QuantizedAttention::get_memory_usage() const {
    size_t total = 0;
    total += q_weights_.size() * sizeof(int8_t);
    total += k_weights_.size() * sizeof(int8_t);
    total += v_weights_.size() * sizeof(int8_t);
    total += o_weights_.size() * sizeof(int8_t);
    total += attention_buffer_.size() * sizeof(int8_t);
    total += quant_params_.size() * sizeof(QuantParams);
    
    if (sparse_enabled_) {
        total += attention_mask_.size() * sizeof(bool);
    }
    
    return total;
}

void QuantizedAttention::enable_sparse_attention(float sparsity_ratio) {
    sparse_enabled_ = true;
    size_t mask_size = config_.embed_dim * config_.embed_dim;
    attention_mask_.resize(mask_size);
    
    // Create random sparse mask
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (size_t i = 0; i < mask_size; ++i) {
        attention_mask_[i] = dis(gen) < sparsity_ratio;
    }
}

void QuantizedAttention::disable_sparse_attention() {
    sparse_enabled_ = false;
    attention_mask_.clear();
}

std::vector<int8_t> QuantizedAttention::compute_qkv_quantized(const std::vector<int8_t>& input) {
    std::vector<int8_t> qkv(3 * config_.embed_dim);
    
    // Simplified QKV computation
    auto q = QuantizedVectorOps::matrix_vector_multiply_quantized(
        q_weights_, input, config_.embed_dim, config_.embed_dim, quant_params_[0]);
    
    auto k = QuantizedVectorOps::matrix_vector_multiply_quantized(
        k_weights_, input, config_.embed_dim, config_.embed_dim, quant_params_[1]);
    
    auto v = QuantizedVectorOps::matrix_vector_multiply_quantized(
        v_weights_, input, config_.embed_dim, config_.embed_dim, quant_params_[2]);
    
    // Copy results
    std::copy(q.begin(), q.end(), qkv.begin());
    std::copy(k.begin(), k.end(), qkv.begin() + config_.embed_dim);
    std::copy(v.begin(), v.end(), qkv.begin() + 2 * config_.embed_dim);
    
    return qkv;
}

std::vector<int8_t> QuantizedAttention::scaled_dot_product_quantized(
    const std::vector<int8_t>& q, const std::vector<int8_t>& k, const std::vector<int8_t>& v) {
    
    // Simplified attention computation
    std::vector<int8_t> output(config_.embed_dim, 0);
    
    // In a full implementation, this would compute proper attention weights
    // For now, just return a weighted combination
    for (size_t i = 0; i < config_.embed_dim; ++i) {
        int32_t sum = 0;
        for (size_t j = 0; j < config_.embed_dim; ++j) {
            if (!sparse_enabled_ || attention_mask_[i * config_.embed_dim + j]) {
                sum += static_cast<int32_t>(q[j]) * static_cast<int32_t>(v[j]);
            }
        }
        sum = sum / config_.embed_dim; // Average
        sum = std::max(-128, std::min(127, sum));
        output[i] = static_cast<int8_t>(sum);
    }
    
    return output;
}

// MemoryOptimizer Implementation
MemoryOptimizer::QuantizedMemoryPool::QuantizedMemoryPool(size_t pool_size_mb)
    : pool_size_bytes_(pool_size_mb * 1024 * 1024), allocated_bytes_(0), peak_usage_(0) {
    
    memory_pool_.resize(pool_size_bytes_);
    allocation_map_.resize(pool_size_bytes_ / 1024, false); // 1KB granularity
}

MemoryOptimizer::QuantizedMemoryPool::~QuantizedMemoryPool() = default;

int8_t* MemoryOptimizer::QuantizedMemoryPool::allocate(size_t num_elements) {
    size_t bytes_needed = num_elements * sizeof(int8_t);
    size_t blocks_needed = (bytes_needed + 1023) / 1024; // Round up to 1KB blocks
    
    // Find contiguous free blocks
    for (size_t i = 0; i <= allocation_map_.size() - blocks_needed; ++i) {
        bool found = true;
        for (size_t j = 0; j < blocks_needed; ++j) {
            if (allocation_map_[i + j]) {
                found = false;
                break;
            }
        }
        
        if (found) {
            // Mark blocks as allocated
            for (size_t j = 0; j < blocks_needed; ++j) {
                allocation_map_[i + j] = true;
            }
            
            allocated_bytes_ += bytes_needed;
            peak_usage_ = std::max(peak_usage_, allocated_bytes_);
            
            return reinterpret_cast<int8_t*>(&memory_pool_[i * 1024]);
        }
    }
    
    return nullptr; // Out of memory
}

void MemoryOptimizer::QuantizedMemoryPool::deallocate(int8_t* ptr) {
    // Find the allocation and free it
    size_t offset = ptr - reinterpret_cast<int8_t*>(memory_pool_.data());
    size_t block_index = offset / 1024;
    
    // Find the end of this allocation (simplified - assumes single block)
    if (block_index < allocation_map_.size()) {
        allocation_map_[block_index] = false;
        allocated_bytes_ -= 1024;
    }
}

void MemoryOptimizer::QuantizedMemoryPool::clear() {
    std::fill(allocation_map_.begin(), allocation_map_.end(), false);
    allocated_bytes_ = 0;
}

size_t MemoryOptimizer::estimate_memory_usage(size_t model_size, bool quantized_8bit, bool quantized_4bit) {
    if (quantized_4bit) {
        return model_size / 8; // 4-bit is 8x smaller than 32-bit
    } else if (quantized_8bit) {
        return model_size / 4; // 8-bit is 4x smaller than 32-bit
    } else {
        return model_size * 4; // 32-bit float
    }
}

// UltraQuantizedOps Implementation
std::vector<uint8_t> UltraQuantizedOps::pack_4bit(const std::vector<int8_t>& input) {
    std::vector<uint8_t> packed((input.size() + 1) / 2);
    
    for (size_t i = 0; i < input.size(); i += 2) {
        uint8_t packed_val = 0;
        
        // Pack first 4-bit value (lower nibble) - clamp to 0-15 range
        int8_t first_val = input[i];
        if (first_val < -8) first_val = -8;
        if (first_val > 7) first_val = 7;
        packed_val = static_cast<uint8_t>(first_val + 8); // Shift to 0-15 range
        
        // Pack second 4-bit value (upper nibble) if it exists
        if (i + 1 < input.size()) {
            int8_t second_val = input[i + 1];
            if (second_val < -8) second_val = -8;
            if (second_val > 7) second_val = 7;
            packed_val |= static_cast<uint8_t>((second_val + 8) << 4);
        }
        
        packed[i / 2] = packed_val;
    }
    
    return packed;
}

std::vector<int8_t> UltraQuantizedOps::unpack_4bit(const std::vector<uint8_t>& packed) {
    std::vector<int8_t> unpacked(packed.size() * 2);
    
    for (size_t i = 0; i < packed.size(); ++i) {
        // Extract lower nibble and shift back to -8 to 7 range
        unpacked[i * 2] = static_cast<int8_t>((packed[i] & 0x0F) - 8);
        
        // Extract upper nibble and shift back to -8 to 7 range
        unpacked[i * 2 + 1] = static_cast<int8_t>(((packed[i] >> 4) & 0x0F) - 8);
    }
    
    return unpacked;
}

std::vector<uint8_t> UltraQuantizedOps::vector_add_4bit(
    const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
    const QuantParams& params) {
    
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    
    std::vector<uint8_t> result(a.size());
    
    for (size_t i = 0; i < a.size(); ++i) {
        // Simple 4-bit addition with saturation
        int16_t sum = (a[i] & 0x0F) + (b[i] & 0x0F);
        result[i] = static_cast<uint8_t>(std::min(static_cast<int>(sum), 15));
    }
    
    return result;
}

std::vector<uint8_t> UltraQuantizedOps::matrix_vector_multiply_4bit(
    const std::vector<uint8_t>& weights, const std::vector<uint8_t>& input,
    size_t output_size, size_t input_size, const QuantParams& params) {
    
    if (weights.size() != output_size * input_size / 2 || input.size() != input_size / 2) {
        throw std::invalid_argument("Matrix-vector dimensions don't match");
    }
    
    std::vector<uint8_t> output(output_size, 0);
    
    for (size_t i = 0; i < output_size; ++i) {
        int16_t sum = 0;
        for (size_t j = 0; j < input_size / 2; ++j) {
            // Extract 4-bit values
            uint8_t w_val = weights[i * input_size / 2 + j];
            uint8_t x_val = input[j];
            
            // Unpack and multiply
            int8_t w_low = static_cast<int8_t>(w_val & 0x0F);
            int8_t x_low = static_cast<int8_t>(x_val & 0x0F);
            
            sum += w_low * x_low;
        }
        
        // Quantize result to 4-bit
        output[i] = static_cast<uint8_t>(std::max(0, std::min(15, sum / 16)));
    }
    
    return output;
}

// Add missing QuantizedAttention method
void QuantizedAttention::optimize_memory_layout() {
    // Simple memory layout optimization
    // In a full implementation, this would reorder data for better cache performance
    if (sparse_enabled_) {
        // Compact the attention mask
        std::vector<bool> compact_mask;
        for (bool val : attention_mask_) {
            if (val) {
                compact_mask.push_back(true);
            }
        }
        attention_mask_ = compact_mask;
    }
}

} // namespace Quantized
} // namespace ML
