//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Real-Time Transformer Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "RealTimeTransformer.h"
#include "LightweightAttention.h"
#include "DynamicNeuralNetwork.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

namespace ML {
namespace RealTime {

// TransformerBlock Implementation
class TransformerBlock {
public:
    TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff, float dropout = 0.1f)
        : d_model_(d_model), n_heads_(n_heads), d_ff_(d_ff), dropout_(dropout) {
        
        // Initialize attention layer
        ML::RealTime::LightweightAttention::Config attn_config{d_model, n_heads, d_model/n_heads, 512};
        attention_ = std::make_unique<ML::RealTime::LightweightAttention>(attn_config);
        
        // Initialize feed-forward layers
        ML::Dynamic::DynamicLayer::Config ff1_config{d_model, d_ff, "relu", false, dropout};
        ML::Dynamic::DynamicLayer::Config ff2_config{d_ff, d_model, "tanh", false, dropout};
        
        ff1_ = std::make_unique<ML::Dynamic::DynamicDenseLayer>(ff1_config);
        ff2_ = std::make_unique<ML::Dynamic::DynamicDenseLayer>(ff2_config);
        
        // Layer normalization weights
        norm1_weight_.resize(d_model, 1.0f);
        norm1_bias_.resize(d_model, 0.0f);
        norm2_weight_.resize(d_model, 1.0f);
        norm2_bias_.resize(d_model, 0.0f);
        
        use_simd_ = XSIMD::XSIMDVector::has_simd_support();
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        if (input.size() != d_model_) {
            throw std::invalid_argument("Input size mismatch");
        }
        
        // Pre-norm: LayerNorm -> Attention
        std::vector<float> normed_input = layer_norm(input, norm1_weight_, norm1_bias_);
        std::vector<float> attn_output = attention_->forward(normed_input);
        
        // Residual connection
        std::vector<float> hidden1 = input;
        if (use_simd_) {
            XSIMD::VectorOps::vector_add_vector(hidden1.data(), attn_output.data(), 
                                               hidden1.data(), d_model_);
        } else {
            for (size_t i = 0; i < d_model_; ++i) {
                hidden1[i] += attn_output[i];
            }
        }
        
        // Pre-norm: LayerNorm -> FeedForward
        std::vector<float> normed_hidden = layer_norm(hidden1, norm2_weight_, norm2_bias_);
        std::vector<float> ff_output = ff1_->forward(normed_hidden);
        ff_output = ff2_->forward(ff_output);
        
        // Residual connection
        std::vector<float> output = hidden1;
        if (use_simd_) {
            XSIMD::VectorOps::vector_add_vector(output.data(), ff_output.data(), 
                                               output.data(), d_model_);
        } else {
            for (size_t i = 0; i < d_model_; ++i) {
                output[i] += ff_output[i];
            }
        }
        
        return output;
    }
    
    size_t get_memory_usage() const {
        size_t total = 0;
        total += attention_->get_memory_usage();
        total += ff1_->get_memory_usage();
        total += ff2_->get_memory_usage();
        total += (norm1_weight_.size() + norm1_bias_.size() + 
                 norm2_weight_.size() + norm2_bias_.size()) * sizeof(float);
        return total;
    }
    
private:
    size_t d_model_, n_heads_, d_ff_;
    float dropout_;
    bool use_simd_;
    
    std::unique_ptr<ML::RealTime::LightweightAttention> attention_;
    std::unique_ptr<ML::Dynamic::DynamicDenseLayer> ff1_, ff2_;
    
    std::vector<float> norm1_weight_, norm1_bias_;
    std::vector<float> norm2_weight_, norm2_bias_;
    
    std::vector<float> layer_norm(const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 const std::vector<float>& bias) {
        std::vector<float> output = input;
        
        // Compute mean and variance
        float mean = 0.0f;
        if (use_simd_) {
            mean = XSIMD::VectorOps::reduce_sum(input.data(), input.size()) / input.size();
        } else {
            for (float val : input) mean += val;
            mean /= input.size();
        }
        
        // Normalize
        float variance = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= input.size();
        float std_dev = std::sqrt(variance + 1e-6f);
        
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = ((input[i] - mean) / std_dev) * weight[i] + bias[i];
        }
        
        return output;
    }
};

// StreamingTransformer Implementation
StreamingTransformer::StreamingTransformer(const Config& config)
    : config_(config), 
      streaming_active_(false),
      total_tokens_processed_(0),
      current_latency_ms_(0.0f) {
    
    // Initialize transformer blocks
    for (size_t i = 0; i < config.n_layers; ++i) {
        blocks_.push_back(std::make_unique<TransformerBlock>(
            config.d_model, config.n_heads, config.d_ff, config.dropout));
    }
    
    // Initialize embeddings
    embedding_weights_.resize(config.vocab_size * config.d_model);
    position_embeddings_.resize(config.max_sequence_length * config.d_model);
    
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 0.02f);
    
    for (auto& weight : embedding_weights_) {
        weight = dis(gen);
    }
    
    for (auto& pos_emb : position_embeddings_) {
        pos_emb = dis(gen);
    }
    
    use_simd_ = XSIMD::XSIMDVector::has_simd_support();
}

StreamingTransformer::~StreamingTransformer() {
    stop_streaming();
}

void StreamingTransformer::start_stream() {
    streaming_active_ = true;
    processing_thread_ = std::thread(&StreamingTransformer::process_stream, this);
}

void StreamingTransformer::stop_stream() {
    streaming_active_ = false;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

void StreamingTransformer::process_token(const std::vector<float>& token_embedding) {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    input_queue_.push(token_embedding);
    stream_cv_.notify_one();
}

std::vector<float> StreamingTransformer::get_next_output() {
    std::unique_lock<std::mutex> lock(stream_mutex_);
    stream_cv_.wait(lock, [this] { return !output_queue_.empty() || !streaming_active_; });
    
    if (!output_queue_.empty()) {
        auto output = output_queue_.front();
        output_queue_.pop();
        return output;
    }
    
    return {};
}

void StreamingTransformer::process_stream() {
    while (streaming_active_) {
        std::unique_lock<std::mutex> lock(stream_mutex_);
        stream_cv_.wait(lock, [this] { return !input_queue_.empty() || !streaming_active_; });
        
        if (!streaming_active_) break;
        
        auto input = input_queue_.front();
        input_queue_.pop();
        lock.unlock();
        
        // Process token through transformer
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto output = forward_single(input);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        current_latency_ms_ = static_cast<float>(duration.count()) / 1000.0f;
        
        // Add to output queue
        lock.lock();
        output_queue_.push(output);
        lock.unlock();
        
        total_tokens_processed_++;
    }
}

std::vector<float> StreamingTransformer::forward_single(const std::vector<float>& input) {
    // Add positional encoding
    std::vector<float> hidden = input;
    if (use_simd_) {
        XSIMD::VectorOps::vector_add_vector(hidden.data(), position_embeddings_.data(), 
                                           hidden.data(), config_.d_model);
    } else {
        for (size_t i = 0; i < config_.d_model; ++i) {
            hidden[i] += position_embeddings_[i];
        }
    }
    
    // Process through transformer blocks
    for (const auto& block : blocks_) {
        hidden = block->forward(hidden);
    }
    
    return hidden;
}

std::vector<float> StreamingTransformer::forward(const std::vector<std::vector<float>>& sequence) {
    std::vector<float> output;
    
    for (const auto& token : sequence) {
        auto token_output = forward_single(token);
        output.insert(output.end(), token_output.begin(), token_output.end());
    }
    
    return output;
}

void StreamingTransformer::optimize_for_latency() {
    // Enable SIMD optimizations
    use_simd_ = XSIMD::XSIMDVector::has_simd_support();
    
    // Optimize memory layout
    for (auto& block : blocks_) {
        // Blocks are already optimized
    }
}

void StreamingTransformer::optimize_for_memory() {
    // Reduce memory usage
    // Clear caches and temporary buffers
    std::queue<std::vector<float>> empty;
    std::swap(input_queue_, empty);
    std::swap(output_queue_, empty);
}

size_t StreamingTransformer::get_memory_usage() const {
    size_t total = 0;
    
    // Embeddings
    total += embedding_weights_.size() * sizeof(float);
    total += position_embeddings_.size() * sizeof(float);
    
    // Transformer blocks
    for (const auto& block : blocks_) {
        total += block->get_memory_usage();
    }
    
    // Queues (approximate)
    total += (input_queue_.size() + output_queue_.size()) * config_.d_model * sizeof(float);
    
    return total;
}

float StreamingTransformer::get_throughput_tokens_per_second() const {
    if (total_tokens_processed_ == 0) return 0.0f;
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_);
    
    if (duration.count() == 0) return 0.0f;
    
    return static_cast<float>(total_tokens_processed_) / duration.count();
}

bool StreamingTransformer::meets_latency_target() const {
    return current_latency_ms_ <= config_.target_latency_ms;
}

bool StreamingTransformer::meets_memory_target() const {
    size_t memory_mb = get_memory_usage() / (1024 * 1024);
    return memory_mb <= config_.max_memory_mb;
}

// RealTimeTransformerFactory Implementation
std::unique_ptr<StreamingTransformer> RealTimeTransformerFactory::create_for_edge(
    size_t vocab_size, float target_latency_ms) {
    
    StreamingTransformer::Config config;
    config.vocab_size = vocab_size;
    config.d_model = 128;  // Smaller for edge
    config.n_heads = 4;
    config.n_layers = 2;
    config.d_ff = 512;
    config.dropout = 0.1f;
    config.max_sequence_length = 256;
    config.target_latency_ms = target_latency_ms;
    config.max_memory_mb = 2;
    
    return std::make_unique<StreamingTransformer>(config);
}

std::unique_ptr<StreamingTransformer> RealTimeTransformerFactory::create_for_mobile(
    size_t vocab_size, float target_latency_ms) {
    
    StreamingTransformer::Config config;
    config.vocab_size = vocab_size;
    config.d_model = 64;   // Even smaller for mobile
    config.n_heads = 2;
    config.n_layers = 2;
    config.d_ff = 256;
    config.dropout = 0.1f;
    config.max_sequence_length = 128;
    config.target_latency_ms = target_latency_ms;
    config.max_memory_mb = 1;
    
    return std::make_unique<StreamingTransformer>(config);
}

std::unique_ptr<StreamingTransformer> RealTimeTransformerFactory::create_for_server(
    size_t vocab_size, float target_latency_ms) {
    
    StreamingTransformer::Config config;
    config.vocab_size = vocab_size;
    config.d_model = 512;  // Larger for server
    config.n_heads = 8;
    config.n_layers = 6;
    config.d_ff = 2048;
    config.dropout = 0.1f;
    config.max_sequence_length = 1024;
    config.target_latency_ms = target_latency_ms;
    config.max_memory_mb = 10;
    
    return std::make_unique<StreamingTransformer>(config);
}

} // namespace RealTime
} // namespace ML
