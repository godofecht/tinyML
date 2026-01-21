//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "LightweightAttention.h"
#include <algorithm>
#include <random>
#include <cstring>
#include <iostream>

namespace ML {
namespace RealTime {

LightweightAttention::LightweightAttention(const Config& config)
    : config_(config), 
      use_simd_(XSIMD::XSIMDVector::has_simd_support()),
      streaming_mode_(false),
      streaming_position_(0),
      packed_layout_(false) {
    
    // Validate configuration
    if (config_.embed_dim % config_.num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    config_.head_dim = config_.embed_dim / config_.num_heads;
    
    initialize_weights();
    allocate_buffers();
    optimize_memory_layout();
}

void LightweightAttention::initialize_weights() {
    // Initialize weight matrices with Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / config_.embed_dim);
    std::normal_distribution<float> dis(0.0f, scale);
    
    // Allocate weights
    size_t weight_size = config_.embed_dim * config_.embed_dim;
    q_weights_.resize(weight_size);
    k_weights_.resize(weight_size);
    v_weights_.resize(weight_size);
    out_weights_.resize(weight_size);
    
    // Initialize biases to zero
    q_bias_.resize(config_.embed_dim, 0.0f);
    k_bias_.resize(config_.embed_dim, 0.0f);
    v_bias_.resize(config_.embed_dim, 0.0f);
    out_bias_.resize(config_.embed_dim, 0.0f);
    
    // Initialize weights
    for (size_t i = 0; i < weight_size; ++i) {
        q_weights_[i] = dis(gen);
        k_weights_[i] = dis(gen);
        v_weights_[i] = dis(gen);
        out_weights_[i] = dis(gen);
    }
}

void LightweightAttention::allocate_buffers() {
    size_t seq_len = config_.sequence_length;
    size_t embed_dim = config_.embed_dim;
    size_t head_dim = config_.head_dim;
    size_t num_heads = config_.num_heads;
    
    // Allocate temporary buffers
    q_buffer_.resize(seq_len * embed_dim);
    k_buffer_.resize(seq_len * embed_dim);
    v_buffer_.resize(seq_len * embed_dim);
    attention_scores_.resize(num_heads * seq_len * seq_len);
    attention_probs_.resize(num_heads * seq_len * seq_len);
    context_buffer_.resize(num_heads * seq_len * head_dim);
    output_buffer_.resize(seq_len * embed_dim);
    
    // Allocate streaming cache
    streaming_cache_k_.resize(seq_len * embed_dim);
    streaming_cache_v_.resize(seq_len * embed_dim);
}

void LightweightAttention::deallocate_buffers() {
    q_buffer_.clear();
    k_buffer_.clear();
    v_buffer_.clear();
    attention_scores_.clear();
    attention_probs_.clear();
    context_buffer_.clear();
    output_buffer_.clear();
    streaming_cache_k_.clear();
    streaming_cache_v_.clear();
}

void LightweightAttention::optimize_memory_layout() {
    // Pack weights for better cache locality
    if (use_simd_ && config_.embed_dim % 8 == 0) {
        packed_layout_ = true;
        // In a real implementation, we would transpose and align matrices
        // for optimal XSIMD processing
    }
}

std::vector<float> LightweightAttention::forward(const std::vector<float>& input) {
    if (input.size() != config_.embed_dim) {
        throw std::invalid_argument("Input size must match embed_dim");
    }
    
    // Clear buffers
    std::fill(output_buffer_.begin(), output_buffer_.end(), 0.0f);
    
    // Step 1: Compute Q, K, V projections
    compute_qkv(input);
    
    // Step 2: Compute attention scores
    compute_attention_scores();
    
    // Step 3: Apply softmax and masking
    apply_softmax_and_mask();
    
    // Step 4: Compute context vectors
    compute_context();
    
    // Step 5: Compute final output projection
    compute_output_projection();
    
    return std::vector<float>(output_buffer_.begin(),
                              output_buffer_.begin() + config_.embed_dim);
}

void LightweightAttention::compute_qkv(const std::vector<float>& input) {
    size_t embed_dim = config_.embed_dim;
    
    // Compute Q = input * q_weights + q_bias
    if (use_simd_) {
        XSIMD::VectorOps::matrix_vector_multiply(
            q_weights_.data(), input.data(), q_buffer_.data(), 1, embed_dim);
        XSIMD::VectorOps::vector_add_vector(
            q_buffer_.data(), q_bias_.data(), q_buffer_.data(), embed_dim);
        
        // Compute K = input * k_weights + k_bias
        XSIMD::VectorOps::matrix_vector_multiply(
            k_weights_.data(), input.data(), k_buffer_.data(), 1, embed_dim);
        XSIMD::VectorOps::vector_add_vector(
            k_buffer_.data(), k_bias_.data(), k_buffer_.data(), embed_dim);
        
        // Compute V = input * v_weights + v_bias
        XSIMD::VectorOps::matrix_vector_multiply(
            v_weights_.data(), input.data(), v_buffer_.data(), 1, embed_dim);
        XSIMD::VectorOps::vector_add_vector(
            v_buffer_.data(), v_bias_.data(), v_buffer_.data(), embed_dim);
    } else {
        // Fallback to scalar computation
        for (size_t i = 0; i < embed_dim; ++i) {
            q_buffer_[i] = q_bias_[i];
            k_buffer_[i] = k_bias_[i];
            v_buffer_[i] = v_bias_[i];
            
            for (size_t j = 0; j < embed_dim; ++j) {
                q_buffer_[i] += input[j] * q_weights_[i * embed_dim + j];
                k_buffer_[i] += input[j] * k_weights_[i * embed_dim + j];
                v_buffer_[i] += input[j] * v_weights_[i * embed_dim + j];
            }
        }
    }
}

void LightweightAttention::compute_attention_scores() {
    size_t num_heads = config_.num_heads;
    size_t head_dim = config_.head_dim;
    size_t seq_len = 1; // Single sequence for now
    
    float scale = 1.0f / sqrt_head_dim();
    
    for (size_t head = 0; head < num_heads; ++head) {
        size_t head_offset = head * head_dim;
        size_t score_offset = head * seq_len * seq_len;
        
        // Compute attention score = Q * K^T / sqrt(head_dim)
        const float* q_head = &q_buffer_[head_offset];
        const float* k_head = &k_buffer_[head_offset];
        float* scores = &attention_scores_[score_offset];
        
        if (use_simd_) {
            float score = XSIMD::VectorOps::dot_product(q_head, k_head, head_dim);
            scores[0] = score * scale;
        } else {
            float score = 0.0f;
            for (size_t i = 0; i < head_dim; ++i) {
                score += q_head[i] * k_head[i];
            }
            scores[0] = score * scale;
        }
    }
}

void LightweightAttention::apply_softmax_and_mask() {
    size_t num_heads = config_.num_heads;
    size_t seq_len = 1;
    
    for (size_t head = 0; head < num_heads; ++head) {
        size_t score_offset = head * seq_len * seq_len;
        size_t prob_offset = head * seq_len * seq_len;
        
        float score = attention_scores_[score_offset];
        
        // Apply softmax (for single element, it's just 1.0)
        attention_probs_[prob_offset] = 1.0f;
        
        // Apply causal mask if needed
        if (config_.use_causal_mask) {
            apply_causal_mask();
        }
    }
}

void LightweightAttention::apply_causal_mask() {
    // For single sequence, no masking needed
    // This would be implemented for longer sequences
}

void LightweightAttention::compute_context() {
    size_t num_heads = config_.num_heads;
    size_t head_dim = config_.head_dim;
    size_t seq_len = 1;
    
    for (size_t head = 0; head < num_heads; ++head) {
        size_t head_offset = head * head_dim;
        size_t prob_offset = head * seq_len * seq_len;
        size_t context_offset = head * seq_len * head_dim;
        
        const float* v_head = &v_buffer_[head_offset];
        float prob = attention_probs_[prob_offset];
        float* context = &context_buffer_[context_offset];
        
        // Context = attention_prob * V
        if (use_simd_) {
            XSIMD::VectorOps::vector_mul_scalar(v_head, prob, context, head_dim);
        } else {
            for (size_t i = 0; i < head_dim; ++i) {
                context[i] = v_head[i] * prob;
            }
        }
    }
}

void LightweightAttention::compute_output_projection() {
    size_t embed_dim = config_.embed_dim;
    size_t num_heads = config_.num_heads;
    size_t head_dim = config_.head_dim;
    
    // Concatenate all heads and apply output projection
    std::vector<float> concatenated(embed_dim);
    
    // Concatenate heads
    for (size_t head = 0; head < num_heads; ++head) {
        size_t head_offset = head * head_dim;
        size_t context_offset = head * head_dim;
        const float* context = &context_buffer_[context_offset];
        
        std::copy(context, context + head_dim, concatenated.begin() + head_offset);
    }
    
    // Apply output projection: output = concatenated * out_weights + out_bias
    if (use_simd_) {
        XSIMD::VectorOps::matrix_vector_multiply(
            out_weights_.data(), concatenated.data(), output_buffer_.data(), 1, embed_dim);
        XSIMD::VectorOps::vector_add_vector(
            output_buffer_.data(), out_bias_.data(), output_buffer_.data(), embed_dim);
    } else {
        for (size_t i = 0; i < embed_dim; ++i) {
            output_buffer_[i] = out_bias_[i];
            for (size_t j = 0; j < embed_dim; ++j) {
                output_buffer_[i] += concatenated[j] * out_weights_[i * embed_dim + j];
            }
        }
    }
}

std::vector<std::vector<float>> LightweightAttention::forward_batch(
    const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> outputs;
    outputs.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        outputs.push_back(forward(input));
    }
    
    return outputs;
}

void LightweightAttention::start_stream() {
    streaming_mode_ = true;
    streaming_position_ = 0;
    std::fill(streaming_cache_k_.begin(), streaming_cache_k_.end(), 0.0f);
    std::fill(streaming_cache_v_.begin(), streaming_cache_v_.end(), 0.0f);
}

std::vector<float> LightweightAttention::process_chunk(const std::vector<float>& chunk) {
    if (!streaming_mode_) {
        return forward(chunk);
    }
    
    // For streaming, we would implement incremental attention computation
    // This is a simplified version that just processes the chunk normally
    return forward(chunk);
}

void LightweightAttention::end_stream() {
    streaming_mode_ = false;
    streaming_position_ = 0;
}

size_t LightweightAttention::get_memory_usage() const {
    size_t total = 0;
    
    // Weights
    total += q_weights_.size() * sizeof(float);
    total += k_weights_.size() * sizeof(float);
    total += v_weights_.size() * sizeof(float);
    total += out_weights_.size() * sizeof(float);
    
    // Biases
    total += q_bias_.size() * sizeof(float);
    total += k_bias_.size() * sizeof(float);
    total += v_bias_.size() * sizeof(float);
    total += out_bias_.size() * sizeof(float);
    
    // Buffers
    total += q_buffer_.size() * sizeof(float);
    total += k_buffer_.size() * sizeof(float);
    total += v_buffer_.size() * sizeof(float);
    total += attention_scores_.size() * sizeof(float);
    total += attention_probs_.size() * sizeof(float);
    total += context_buffer_.size() * sizeof(float);
    total += output_buffer_.size() * sizeof(float);
    
    // Streaming cache
    total += streaming_cache_k_.size() * sizeof(float);
    total += streaming_cache_v_.size() * sizeof(float);
    
    return total;
}

void LightweightAttention::optimize_for_latency() {
    // Implement latency optimizations
    // - Pre-allocate all buffers
    // - Use XSIMD when available
    // - Minimize memory allocations
    packed_layout_ = use_simd_;
}

void LightweightAttention::optimize_for_memory() {
    // Implement memory optimizations
    // - Reduce buffer sizes
    // - Use in-place operations where possible
    // - Deallocate unused buffers
    
    // Reduce sequence length for memory-constrained scenarios
    if (config_.sequence_length > 128) {
        config_.sequence_length = 128;
        deallocate_buffers();
        allocate_buffers();
    }
}

// Factory function
std::unique_ptr<LightweightAttention> create_lightweight_attention(
    const LightweightAttention::Config& config) {
    return std::make_unique<LightweightAttention>(config);
}

} // namespace RealTime
} // namespace ML
