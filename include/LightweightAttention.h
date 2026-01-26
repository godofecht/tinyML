//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef LIGHTWEIGHT_ATTENTION_H
#define LIGHTWEIGHT_ATTENTION_H

#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include "XSIMDOperations.h"

namespace ML {
namespace RealTime {

// Efficient multi-head attention implementation optimized for real-time inference
class LightweightAttention {
public:
    struct Config {
        size_t embed_dim = 256;        // Embedding dimension
        size_t num_heads = 8;           // Number of attention heads
        size_t head_dim = 32;           // Dimension per head (embed_dim / num_heads)
        size_t sequence_length = 512;   // Maximum sequence length
        float dropout_rate = 0.1f;      // Dropout rate (disabled during inference)
        bool use_causal_mask = false;    // Whether to use causal masking
    };

    LightweightAttention(const Config& config);
    ~LightweightAttention() = default;

    // Forward pass - main attention computation
    std::vector<float> forward(const std::vector<float>& input);
    
    // Batch processing for multiple sequences
    std::vector<std::vector<float>> forward_batch(
        const std::vector<std::vector<float>>& inputs);
    
    // Real-time streaming interface
    void start_stream();
    std::vector<float> process_chunk(const std::vector<float>& chunk);
    void end_stream();
    
    // Memory management
    void allocate_buffers();
    void deallocate_buffers();
    size_t get_memory_usage() const;
    
    // Performance optimization
    void optimize_for_latency();
    void optimize_for_memory();
    
    // Configuration access
    const Config& get_config() const { return config_; }
    size_t get_output_dim() const { return config_.embed_dim; }

    void perturb_weights(float sigma, unsigned int seed, float direction = 1.0f) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, sigma);
        
        for (auto& w : q_weights_) w += direction * dist(gen);
        for (auto& w : k_weights_) w += direction * dist(gen);
        for (auto& w : v_weights_) w += direction * dist(gen);
        for (auto& w : out_weights_) w += direction * dist(gen);
        
        for (auto& b : q_bias_) b += direction * dist(gen);
        for (auto& b : k_bias_) b += direction * dist(gen);
        for (auto& b : v_bias_) b += direction * dist(gen);
        for (auto& b : out_bias_) b += direction * dist(gen);
    }

    std::vector<float> get_weights() const {
        std::vector<float> all_weights;
        all_weights.insert(all_weights.end(), q_weights_.begin(), q_weights_.end());
        all_weights.insert(all_weights.end(), k_weights_.begin(), k_weights_.end());
        all_weights.insert(all_weights.end(), v_weights_.begin(), v_weights_.end());
        all_weights.insert(all_weights.end(), out_weights_.begin(), out_weights_.end());
        return all_weights;
    }

private:
    Config config_;
    bool use_simd_;
    
    // Weight matrices (Q, K, V projections and output projection)
    std::vector<float> q_weights_;      // [embed_dim, embed_dim]
    std::vector<float> k_weights_;      // [embed_dim, embed_dim]
    std::vector<float> v_weights_;      // [embed_dim, embed_dim]
    std::vector<float> out_weights_;    // [embed_dim, embed_dim]
    
    // Bias vectors
    std::vector<float> q_bias_;         // [embed_dim]
    std::vector<float> k_bias_;         // [embed_dim]
    std::vector<float> v_bias_;         // [embed_dim]
    std::vector<float> out_bias_;       // [embed_dim]
    
    // Temporary buffers (allocated once for efficiency)
    std::vector<float> q_buffer_;        // [seq_len, embed_dim]
    std::vector<float> k_buffer_;        // [seq_len, embed_dim]
    std::vector<float> v_buffer_;        // [seq_len, embed_dim]
    std::vector<float> attention_scores_; // [num_heads, seq_len, seq_len]
    std::vector<float> attention_probs_;  // [num_heads, seq_len, seq_len]
    std::vector<float> context_buffer_;   // [num_heads, seq_len, head_dim]
    std::vector<float> output_buffer_;    // [seq_len, embed_dim]
    
    // Streaming state
    bool streaming_mode_;
    std::vector<float> streaming_cache_k_;
    std::vector<float> streaming_cache_v_;
    size_t streaming_position_;
    
    // Core attention computations
    void compute_qkv(const std::vector<float>& input);
    void compute_attention_scores();
    void apply_softmax_and_mask();
    void compute_context();
    void compute_output_projection();
    
    // SIMD-optimized operations
    void scaled_dot_product_attention_simd(
        const float* q, const float* k, const float* v,
        float* output, size_t seq_len, size_t head_dim);
    
    void multi_head_concatenation_simd(
        const float* heads_output, float* output,
        size_t num_heads, size_t seq_len, size_t head_dim);
    
    // Utility functions
    void initialize_weights();
    void apply_causal_mask();
    float sqrt_head_dim() const { return std::sqrt(static_cast<float>(config_.head_dim)); }
    
    // Memory layout optimization
    void optimize_memory_layout();
    bool is_packed_layout() const { return packed_layout_; }
    bool packed_layout_;
};

// Factory function for creating optimized attention instances
std::unique_ptr<LightweightAttention> create_lightweight_attention(
    const LightweightAttention::Config& config);

} // namespace RealTime
} // namespace ML

#endif // LIGHTWEIGHT_ATTENTION_H
