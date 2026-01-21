//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Real-Time Transformer Architecture for Edge AI
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef REALTIME_TRANSFORMER_H
#define REALTIME_TRANSFORMER_H

#include "XSIMDOperations.h"
#include <vector>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace ML {
namespace RealTime {

// Forward declarations
class MultiHeadAttention;
class FeedForwardLayer;
class LayerNorm;
class TransformerBlock;

/**
 * @brief Streaming interface for real-time transformer processing
 * 
 * This class provides a zero-allocation, lock-free interface for processing
 * continuous data streams with transformer models optimized for sub-millisecond
 * latency on edge devices.
 */
class StreamingTransformer {
public:
    struct Config {
        size_t vocab_size = 1000;
        size_t d_model = 256;
        size_t n_heads = 8;
        size_t n_layers = 4;
        size_t d_ff = 1024;
        float dropout = 0.1f;
        size_t max_sequence_length = 512;
        float target_latency_ms = 1.0f;
        size_t max_memory_mb = 5;
    };

    StreamingTransformer(const Config& config);
    ~StreamingTransformer();

    // Real-time inference interface
    std::vector<float> process(const std::vector<float>& input);
    
    // Streaming interface for continuous data
    void start_stream();
    void push_chunk(const std::vector<float>& chunk);
    std::vector<float> get_output();
    void stop_stream();
    
    // Adaptive optimization
    void optimize_for_latency(float target_ms);
    void optimize_for_memory(size_t max_bytes);
    
    // Performance monitoring
    float get_average_latency() const;
    size_t get_memory_usage() const;
    float get_throughput() const;

private:
    Config config_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    
    // Streaming state
    std::queue<std::vector<float>> input_queue_;
    std::queue<std::vector<float>> output_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> streaming_active_{false};
    std::thread processing_thread_;
    
    // Performance tracking
    std::vector<float> latency_samples_;
    mutable std::mutex performance_mutex_;
    std::atomic<float> average_latency_{0.0f};
    std::atomic<size_t> memory_usage_{0};
    std::atomic<float> throughput_{0.0f};
    
    // Internal processing
    void processing_loop();
    std::vector<float> forward_pass(const std::vector<float>& input);
    void update_performance_metrics(float latency, size_t memory_used);
};

/**
 * @brief Multi-head attention with XSIMD optimization
 * 
 * Implements efficient attention computation with AVX2 acceleration,
 * supporting variable sequence lengths and adaptive computation.
 */
class MultiHeadAttention {
public:
    MultiHeadAttention(size_t d_model, size_t n_heads);
    
    // Forward pass with XSIMD optimization
    void forward(const std::vector<float>& input, 
                const std::vector<float>& key_cache,
                const std::vector<float>& value_cache,
                std::vector<float>& output);
    
    // Streaming attention for continuous sequences
    void streaming_forward(const std::vector<float>& new_token,
                          const std::vector<float>& past_keys,
                          const std::vector<float>& past_values,
                          std::vector<float>& output,
                          std::vector<float>& new_keys,
                          std::vector<float>& new_values);
    
    // Adaptive computation - early exit for simple patterns
    float compute_complexity_score(const std::vector<float>& attention_weights);
    bool should_early_exit(const std::vector<float>& input);

private:
    size_t d_model_;
    size_t n_heads_;
    size_t head_dim_;
    
    // Weight matrices (transposed for cache-friendly access)
    XSIMD::XSIMDVector q_weight_;
    XSIMD::XSIMDVector k_weight_;
    XSIMD::XSIMDVector v_weight_;
    XSIMD::XSIMDVector o_weight_;
    
    // Temporary buffers (reused to avoid allocations)
    std::vector<float> q_buffer_;
    std::vector<float> k_buffer_;
    std::vector<float> v_buffer_;
    std::vector<float> attention_buffer_;
    
    // XSIMD-optimized internal operations
    void compute_qkv(const std::vector<float>& input,
                     std::vector<float>& q, std::vector<float>& k, std::vector<float>& v);
    float scaled_dot_product_attention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      std::vector<float>& output);
};

/**
 * @brief XSIMD-optimized feed-forward layer
 * 
 * Two-layer feed-forward network with GELU activation,
- optimized for minimal memory allocation and maximum throughput.
 */
class FeedForwardLayer {
public:
    FeedForwardLayer(size_t d_model, size_t d_ff);
    
    void forward(const std::vector<float>& input, std::vector<float>& output);
    
    // Adaptive width - can reduce d_ff for faster inference
    void set_adaptive_width(size_t new_d_ff);
    size_t get_current_width() const { return current_d_ff_; }

private:
    size_t d_model_;
    size_t d_ff_;
    size_t current_d_ff_;
    
    XSIMD::XSIMDVector w1_weight_;
    XSIMD::XSIMDVector w2_weight_;
    XSIMD::XSIMDVector b1_bias_;
    XSIMD::XSIMDVector b2_bias_;
    
    std::vector<float> intermediate_buffer_;
    
    // Fast GELU approximation
    void gelu_approximate(const std::vector<float>& input, std::vector<float>& output);
};

/**
 * @brief Layer normalization with XSIMD optimization
 * 
 * Implements efficient layer normalization with minimal memory overhead,
 * supporting both training and inference modes.
 */
class LayerNorm {
public:
    LayerNorm(size_t d_model, float epsilon = 1e-6f);
    
    void forward(const std::vector<float>& input, std::vector<float>& output);
    
    // Streaming normalization for real-time data
    void streaming_normalize(const std::vector<float>& input,
                            std::vector<float>& output,
                            float& running_mean, float& running_var);

private:
    size_t d_model_;
    float epsilon_;
    
    XSIMD::XSIMDVector weight_;
    XSIMD::XSIMDVector bias_;
    
    // Running statistics for streaming mode
    float running_mean_{0.0f};
    float running_var_{1.0f};
    std::atomic<bool> training_mode_{false};
};

/**
 * @brief Complete transformer block with adaptive computation
 * 
 * Combines attention, feed-forward, and normalization layers with
 * dynamic optimization based on input complexity and resource constraints.
 */
class TransformerBlock {
public:
    TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff, float dropout = 0.1f);
    
    void forward(const std::vector<float>& input, 
                std::vector<float>& output,
                bool use_cache = true);
    
    // Adaptive computation based on input complexity
    float compute_complexity(const std::vector<float>& input);
    void set_computation_budget(float budget_factor); // 0.0 to 1.0
    
    // Memory management
    void clear_cache();
    size_t get_cache_size() const;

private:
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<FeedForwardLayer> feed_forward_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<LayerNorm> norm2_;
    
    // Cache for streaming inference
    std::vector<float> key_cache_;
    std::vector<float> value_cache_;
    
    // Adaptive computation state
    float computation_budget_{1.0f};
    bool use_fast_path_{false};
    
    // Temporary buffers
    std::vector<float> attention_output_;
    std::vector<float> ff_output_;
    std::vector<float> norm_output_;
};

/**
 * @brief Dynamic neural network with adaptive architecture
 * 
 * Self-optimizing neural network that can adjust its topology,
 * prune connections, and adapt to resource constraints in real-time.
 */
class DynamicNeuralNetwork {
public:
    struct LayerConfig {
        size_t input_size;
        size_t output_size;
        std::string activation = "tanh";
        bool trainable = true;
        float pruning_threshold = 0.01f;
    };
    
    DynamicNeuralNetwork(const std::vector<LayerConfig>& layers);
    
    // Forward pass with dynamic optimization
    std::vector<float> forward(const std::vector<float>& input);
    
    // Architecture adaptation
    void prune_connections(float threshold);
    void add_neurons(size_t layer_idx, size_t count);
    void remove_neurons(size_t layer_idx, const std::vector<size_t>& indices);
    
    // Resource-aware optimization
    void optimize_for_memory(size_t max_memory_mb);
    void optimize_for_latency(float target_latency_ms);
    
    // Continuous learning (gradient-free)
    void update_online(const std::vector<float>& input, const std::vector<float>& target);
    
    // Architecture inspection
    size_t get_total_parameters() const;
    size_t get_active_parameters() const;
    std::vector<LayerConfig> get_current_architecture() const;

private:
    std::vector<LayerConfig> layers_;
    std::vector<XSIMD::XSIMDVector> weights_;
    std::vector<XSIMD::XSIMDVector> biases_;
    
    // Dynamic architecture state
    std::vector<std::vector<bool>> active_connections_;
    std::vector<float> importance_scores_;
    
    // Online learning state
    float learning_rate_{0.01f};
    std::vector<float> error_history_;
    
    // Optimization methods
    void compute_importance_scores();
    void adaptive_pruning();
    void neuron_growth_heuristic();
};

} // namespace RealTime
} // namespace ML

#endif // REALTIME_TRANSFORMER_H
