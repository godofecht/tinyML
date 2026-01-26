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
    
    std::vector<std::vector<float>> get_activations(const std::vector<float>& input) {
        std::vector<std::vector<float>> activations;
        
        // Pre-norm: LayerNorm -> Attention
        std::vector<float> normed_input = layer_norm(input, norm1_weight_, norm1_bias_);
        std::vector<float> attn_output = attention_->forward(normed_input);
        activations.push_back(attn_output);
        
        // Residual connection
        std::vector<float> hidden1 = input;
        for (size_t i = 0; i < d_model_; ++i) {
            hidden1[i] += attn_output[i];
        }
        
        // Pre-norm: LayerNorm -> FeedForward
        std::vector<float> normed_hidden = layer_norm(hidden1, norm2_weight_, norm2_bias_);
        std::vector<float> ff1_out = ff1_->forward(normed_hidden);
        activations.push_back(ff1_out);
        
        std::vector<float> ff2_out = ff2_->forward(ff1_out);
        activations.push_back(ff2_out);
        
        return activations;
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

    void perturb_weights(float sigma, unsigned int seed, float direction = 1.0f) {
        ff1_->perturb_weights(sigma, seed, direction);
        ff2_->perturb_weights(sigma, seed, direction);
        
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, sigma);
        
        for (auto& w : norm1_weight_) w += direction * dist(gen);
        for (auto& b : norm1_bias_) b += direction * dist(gen);
        for (auto& w : norm2_weight_) w += direction * dist(gen);
        for (auto& b : norm2_bias_) b += direction * dist(gen);
        
        attention_->perturb_weights(sigma, seed, direction);
    }

    std::vector<float> get_weights() const {
        std::vector<float> all_weights;
        
        // Attention weights
        auto attn_weights = attention_->get_weights();
        all_weights.insert(all_weights.end(), attn_weights.begin(), attn_weights.end());
        
        // FF weights
        auto ff1_weights = ff1_->get_weights();
        all_weights.insert(all_weights.end(), ff1_weights.begin(), ff1_weights.end());
        
        auto ff2_weights = ff2_->get_weights();
        all_weights.insert(all_weights.end(), ff2_weights.begin(), ff2_weights.end());
        
        // Norm weights
        all_weights.insert(all_weights.end(), norm1_weight_.begin(), norm1_weight_.end());
        all_weights.insert(all_weights.end(), norm1_bias_.begin(), norm1_bias_.end());
        all_weights.insert(all_weights.end(), norm2_weight_.begin(), norm2_weight_.end());
        all_weights.insert(all_weights.end(), norm2_bias_.begin(), norm2_bias_.end());
        
        return all_weights;
    }
    
    std::vector<std::vector<float>> get_weights_structured() const {
        std::vector<std::vector<float>> structured_weights;
        
        // Attention weights
        structured_weights.push_back(attention_->get_weights());
        
        // FF1 weights
        structured_weights.push_back(ff1_->get_weights());
        
        // FF2 weights
        structured_weights.push_back(ff2_->get_weights());
        
        return structured_weights;
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
    stop_stream();
}

void StreamingTransformer::start_stream() {
    streaming_active_ = true;
    start_time_ = std::chrono::steady_clock::now();
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

std::vector<std::vector<float>> StreamingTransformer::get_activations(const std::vector<float>& input) {
    std::vector<std::vector<float>> all_activations;
    
    // Initial embedding (input + pos)
    std::vector<float> hidden = input;
    // Add pos embedding
    for (size_t i = 0; i < config_.d_model; ++i) {
        if (i < hidden.size()) hidden[i] += position_embeddings_[i];
    }
    all_activations.push_back(hidden);
    
    // Process through transformer blocks
    for (const auto& block : blocks_) {
        // Get activations from block
        auto block_acts = block->get_activations(hidden);
        all_activations.insert(all_activations.end(), block_acts.begin(), block_acts.end());
        
        // Update hidden for next block
        hidden = block->forward(hidden);
    }
    
    all_activations.push_back(hidden); // Final output
    return all_activations;
}

void StreamingTransformer::perturb_weights(float noise_std, unsigned int seed, float direction) {
    // Simple random perturbation to simulate training updates
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, noise_std);

    for (auto& w : embedding_weights_) {
        w += direction * dist(gen);
    }
    
    // Perturb block weights
    for (auto& block : blocks_) {
        block->perturb_weights(noise_std, seed, direction);
    }
}

std::vector<float> StreamingTransformer::get_weights() const {
    std::vector<float> all_weights;
    
    // Embeddings
    all_weights.insert(all_weights.end(), embedding_weights_.begin(), embedding_weights_.end());
    
    // Blocks
    for (const auto& block : blocks_) {
        auto block_weights = block->get_weights();
        all_weights.insert(all_weights.end(), block_weights.begin(), block_weights.end());
    }
    
    return all_weights;
}

std::vector<std::vector<float>> StreamingTransformer::get_weights_structured() const {
        std::vector<std::vector<float>> structured_weights;
        
        for (size_t i = 0; i < blocks_.size(); ++i) {
            auto block_weights = blocks_[i]->get_weights_structured();
            structured_weights.insert(structured_weights.end(), block_weights.begin(), block_weights.end());
            
            // Add empty weights for the connection between blocks (residual/identity)
            // This ensures alignment with get_activations which has an intermediate state between blocks
            if (i < blocks_.size() - 1) {
                structured_weights.push_back({});
            }
        }
        
        // Note: The final connection from Last Block FF2 -> Output is also implicit/residual
        // get_activations returns [..., FF2_out, Final_Output]
        // The loop above handles layers within blocks.
        // We need one last empty weight set for the final connection if we want to match connection count.
        // Connections = 3*N + 1.
        // Current size = 3*N + (N-1) = 4N - 1.
        // Wait.
        // N=1: Activations=5 (In, Attn, FF1, FF2, Out). Connections=4.
        // We added: Attn, FF1, FF2. Size=3.
        // Conn 0: In->Attn (W0)
        // Conn 1: Attn->FF1 (W1)
        // Conn 2: FF1->FF2 (W2)
        // Conn 3: FF2->Out (No W).
        // So we need one more empty vector at the end?
        
        // N=2: Activations=8 (In, A1, F1, F2, In2, A2, F1, F2, Out). Connections=7.
        // We added: A1, F1, F2, {}, A2, F1, F2. Size=7.
        // Conn 0: In->A1 (W0)
        // ...
        // Conn 2: F1->F2 (W2)
        // Conn 3: F2->In2 (W3 - empty) -> Grey. Correct.
        // Conn 4: In2->A2 (W4)
        // ...
        // Conn 6: F2->Out (No W).
        // But our vector has 7 elements.
        // Indices 0..6.
        // Conn 6 uses weights[6] which is F2 weights?
        // Wait.
        // blocks_[i]->get_weights_structured() returns [Attn, FF1, FF2].
        // So weights[6] is FF2 weights of block 2.
        // Conn 6 is F2 -> Out.
        // Is Conn 6 represented by FF2 weights?
        // No, FF2 weights are used to compute F2 output from F1 output.
        // That is Conn 5 (F1 -> F2).
        
        // Let's re-trace N=1.
        // Acts: In, A, F1, F2, Out.
        // Conn 0: In -> A. Uses Attn weights. Correct.
        // Conn 1: A -> F1. Uses FF1 weights. Correct.
        // Conn 2: F1 -> F2. Uses FF2 weights. Correct.
        // Conn 3: F2 -> Out. Residual. Should be empty.
        
        // So for N=1, we need [A, F1, F2, {}]. Size 4.
        
        // N=2.
        // Acts: In, A1, F1, F2, In2, A2, F1, F2, Out.
        // Conn 0: In->A1 (A1)
        // Conn 1: A1->F1 (F1)
        // Conn 2: F1->F2 (F2)
        // Conn 3: F2->In2 ({})
        // Conn 4: In2->A2 (A2)
        // Conn 5: A2->F1 (F1)
        // Conn 6: F1->F2 (F2)
        // Conn 7: F2->Out ({}).
        
        // Total connections = 8?
        // Acts size = 9. 9 nodes -> 8 intervals.
        // My previous count: 3N + 2 acts. N=2 -> 8 acts. 7 intervals.
        // Acts: In, A1, F1, F2, A2, F1, F2, Out.
        // Wait, get_activations logic:
        // push(In)
        // Loop N:
        //   push(A)
        //   push(F1)
        //   push(F2)
        //   hidden = forward(hidden) // Updates hidden for next iter
        // push(hidden) // Final Out
        
        // Loop 1 (Block 1):
        //   push(A1)
        //   push(F1_1)
        //   push(F2_1)
        //   hidden becomes Block1_Out (which is In2)
        // Loop 2 (Block 2):
        //   push(A2)
        //   push(F1_2)
        //   push(F2_2)
        //   hidden becomes Block2_Out
        // push(Block2_Out)
        
        // Total: 1 + 3 + 3 + 1 = 8.
        // Connections: 7.
        // 0: In->A1 (A1 weights)
        // 1: A1->F1 (F1 weights)
        // 2: F1->F2 (F2 weights)
        // 3: F2->A2. (Transition Block1->Block2).
        //    Wait. F2 is the last activation pushed in Block1 loop.
        //    A2 is the first activation pushed in Block2 loop.
        //    Is there an intermediate "In2" activation?
        //    No. `hidden` is updated but not pushed as a separate "Input to Block 2" node.
        //    The next pushed node is `block->get_activations` -> `attn_output`.
        //    So Conn 3 connects F2_1 -> A2.
        
        //    Is F2_1 -> A2 a direct connection with weights?
        //    F2_1 is output of FF2 in Block 1.
        //    Block 1 Output = Input + F2_1 (Residual).
        //    Block 2 Input = Block 1 Output.
        //    Block 2 Attn = Attention(Block 2 Input).
        //    So F2_1 contributes to Block 2 Input, which goes into Attn.
        //    There is no single weight matrix between F2_1 and A2.
        //    Also, the "Input" to A2 is (In + A1 + F1 + F2) essentially (simplifying residual).
        
        //    If we visualize F2_1 -> A2, we are skipping the residual summation step.
        //    However, `script.js` just draws lines between layers.
        //    If we provide weights[3] as A2 weights?
        //    A2 weights operate on Block 2 Input.
        //    Block 2 Input is roughly F2_1 (plus residuals).
        //    So using A2 weights for the connection F2_1 -> A2 is *plausible* for visualization.
        //    It shows that A2 depends on the previous output via A2 weights.
        
        //    So:
        //    Conn 0: In->A1 (A1 W)
        //    Conn 1: A1->F1 (F1 W)
        //    Conn 2: F1->F2 (F2 W)
        //    Conn 3: F2->A2 (A2 W)
        //    Conn 4: A2->F1 (F1 W)
        //    Conn 5: F1->F2 (F2 W)
        //    Conn 6: F2->Out (No W / Identity).
        
        //    So we need: [A1, F1, F2, A2, F1, F2, {}].
        //    This means we simply concatenate all block weights, and append one empty at the end.
        
        for (const auto& block : blocks_) {
            auto block_weights = block->get_weights_structured();
            structured_weights.insert(structured_weights.end(), block_weights.begin(), block_weights.end());
        }
        
        // Add one empty for the final connection (F2 -> Final Output)
        structured_weights.push_back({});
        
        return structured_weights;
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

float StreamingTransformer::train_step(const std::vector<float>& input, const std::vector<float>& target) {
    // Compute initial loss
    std::vector<float> output = forward_single(input);
    float initial_loss = 0.0f;
    for (size_t i = 0; i < output.size() && i < target.size(); ++i) {
        float diff = output[i] - target[i];
        initial_loss += diff * diff;
    }
    
    // Generate random seed
    std::random_device rd;
    unsigned int seed = rd();
    float noise_std = 0.02f;
    
    // Perturb weights (positive direction)
    perturb_weights(noise_std, seed, 1.0f);
    
    // Compute new loss
    std::vector<float> new_output = forward_single(input);
    float new_loss = 0.0f;
    for (size_t i = 0; i < new_output.size() && i < target.size(); ++i) {
        float diff = new_output[i] - target[i];
        new_loss += diff * diff;
    }
    
    // Hill Climbing (1+1 ES)
    if (new_loss < initial_loss) {
        std::cout << "Train step: Improved loss " << initial_loss << " -> " << new_loss << std::endl;
        return new_loss;
    } else {
        // Revert changes (negative direction with same seed)
        perturb_weights(noise_std, seed, -1.0f);
        
        // Verify revert
        std::vector<float> reverted_output = forward_single(input);
        float reverted_loss = 0.0f;
        for (size_t i = 0; i < reverted_output.size() && i < target.size(); ++i) {
           float diff = reverted_output[i] - target[i];
           reverted_loss += diff * diff;
        }
        std::cout << "Train step: Reverted. Initial: " << initial_loss << ", New: " << new_loss << ", Reverted: " << reverted_loss << std::endl;
        
        return initial_loss;
    }
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

const StreamingTransformer::Config& StreamingTransformer::get_config() const {
    return config_;
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
    config.dropout = 0.0f;
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
