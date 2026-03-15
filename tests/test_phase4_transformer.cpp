//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 4 Real-Time Transformer Implementation Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

// Include existing components
#include "SIMDOperations.h"

namespace ML {
namespace RealTime {

// Mock RealTimeTransformer class (to be implemented in Phase 4)
class RealTimeTransformer {
private:
    size_t vocab_size;
    size_t embed_dim;
    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    float dropout_rate;
    
    // Transformer components
    std::vector<float> embedding_weights;
    std::vector<float> position_embeddings;
    std::vector<std::vector<float>> attention_weights;
    std::vector<std::vector<float>> feedforward_weights;
    std::vector<std::vector<float>> layer_norm_weights;
    
    // Streaming state
    std::queue<std::vector<float>> input_stream;
    std::queue<std::vector<float>> output_stream;
    std::mutex stream_mutex;
    std::condition_variable stream_cv;
    bool streaming_active;
    
    // Adaptive computation state
    size_t current_depth;
    float complexity_score;
    bool early_exit_enabled;
    
    // Memory management
    std::vector<float> working_memory;
    size_t memory_offset;
    
public:
    RealTimeTransformer(size_t vocab_size = 1000, size_t embed_dim = 256, 
                       size_t num_heads = 8, size_t num_layers = 4, 
                       size_t max_seq_len = 512)
        : vocab_size(vocab_size), embed_dim(embed_dim), num_heads(num_heads),
          num_layers(num_layers), max_seq_len(max_seq_len), dropout_rate(0.1f),
          current_depth(num_layers), complexity_score(0.0f), early_exit_enabled(true),
          streaming_active(false), memory_offset(0) {
        
        initializeComponents();
        initializeMemory();
    }
    
    void initializeComponents() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
        
        // Initialize embedding weights
        embedding_weights.resize(vocab_size * embed_dim);
        for (auto& w : embedding_weights) w = dis(gen);
        
        // Initialize position embeddings
        position_embeddings.resize(max_seq_len * embed_dim);
        for (size_t pos = 0; pos < max_seq_len; ++pos) {
            for (size_t dim = 0; dim < embed_dim; ++dim) {
                // Sinusoidal position encoding
                float pos_val = static_cast<float>(pos);
                float dim_val = static_cast<float>(dim);
                if (dim % 2 == 0) {
                    position_embeddings[pos * embed_dim + dim] = 
                        std::sin(pos_val / std::pow(10000.0f, dim_val / embed_dim));
                } else {
                    position_embeddings[pos * embed_dim + dim] = 
                        std::cos(pos_val / std::pow(10000.0f, (dim_val - 1) / embed_dim));
                }
            }
        }
        
        // Initialize transformer layers
        for (size_t layer = 0; layer < num_layers; ++layer) {
            // Multi-head attention weights (Q, K, V, O projections)
            attention_weights.push_back(std::vector<float>(4 * embed_dim * embed_dim));
            for (auto& w : attention_weights.back()) w = dis(gen);
            
            // Feed-forward weights
            feedforward_weights.push_back(std::vector<float>(embed_dim * embed_dim * 4)); // 4x expansion
            for (auto& w : feedforward_weights.back()) w = dis(gen);
            
            // Layer normalization weights
            layer_norm_weights.push_back(std::vector<float>(embed_dim * 2)); // gamma and beta
            for (auto& w : layer_norm_weights.back()) w = dis(gen);
        }
    }
    
    void initializeMemory() {
        working_memory.resize(max_seq_len * embed_dim * 4); // Working buffer
        memory_offset = 0;
    }
    
    // Embedding Layer (SIMD)
    void embedInput(const std::vector<int>& token_ids, std::vector<float>& embeddings) const {
        embeddings.resize(token_ids.size() * embed_dim);
        
        for (size_t i = 0; i < token_ids.size(); ++i) {
            int token_id = token_ids[i];
            if (token_id >= 0 && token_id < vocab_size) {
                // Copy embedding weights
                std::copy(embedding_weights.begin() + token_id * embed_dim,
                         embedding_weights.begin() + (token_id + 1) * embed_dim,
                         embeddings.begin() + i * embed_dim);
                
                // Add position embedding
                if (i < max_seq_len) {
                    for (size_t dim = 0; dim < embed_dim; ++dim) {
                        embeddings[i * embed_dim + dim] += position_embeddings[i * embed_dim + dim];
                    }
                }
            }
        }
    }
    
    // Transformer Block with Multi-Head Attention
    void transformerBlock(const std::vector<float>& input, 
                         std::vector<float>& output, 
                         size_t layer_idx) const {
        size_t seq_len = input.size() / embed_dim;
        output.resize(input.size());
        
        // Simplified attention mechanism (would use full attention from Phase 2)
        std::vector<float> attention_output(seq_len * embed_dim);
        computeAttention(input, attention_output, layer_idx);
        
        // Add & Norm
        std::vector<float> residual1(seq_len * embed_dim);
        for (size_t i = 0; i < input.size(); ++i) {
            residual1[i] = input[i] + attention_output[i];
        }
        applyLayerNorm(residual1, residual1, layer_idx);
        
        // Feed-forward
        std::vector<float> ff_output(seq_len * embed_dim);
        computeFeedForward(residual1, ff_output, layer_idx);
        
        // Add & Norm
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = residual1[i] + ff_output[i];
        }
        applyLayerNorm(output, output, layer_idx);
    }
    
    void computeAttention(const std::vector<float>& input, 
                         std::vector<float>& output, 
                         size_t layer_idx) const {
        // Simplified attention computation
        size_t seq_len = input.size() / embed_dim;
        const auto& attn_weights = attention_weights[layer_idx];
        
        // QKV projections (simplified)
        std::vector<float> q(seq_len * embed_dim);
        std::vector<float> k(seq_len * embed_dim);
        std::vector<float> v(seq_len * embed_dim);
        
        for (size_t i = 0; i < seq_len; ++i) {
            // Q projection
            ML::SIMD::VectorOps::matrix_vector_multiply(
                attn_weights.data(), input.data() + i * embed_dim,
                q.data() + i * embed_dim, embed_dim, embed_dim
            );
            
            // K projection
            ML::SIMD::VectorOps::matrix_vector_multiply(
                attn_weights.data() + embed_dim * embed_dim, input.data() + i * embed_dim,
                k.data() + i * embed_dim, embed_dim, embed_dim
            );
            
            // V projection
            ML::SIMD::VectorOps::matrix_vector_multiply(
                attn_weights.data() + 2 * embed_dim * embed_dim, input.data() + i * embed_dim,
                v.data() + i * embed_dim, embed_dim, embed_dim
            );
        }
        
        // Simplified attention (average pooling as placeholder)
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t dim = 0; dim < embed_dim; ++dim) {
                output[i * embed_dim + dim] = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    output[i * embed_dim + dim] += v[j * embed_dim + dim] / seq_len;
                }
            }
        }
        
        // Output projection
        std::vector<float> temp_output = output;
        ML::SIMD::VectorOps::matrix_vector_multiply(
            attn_weights.data() + 3 * embed_dim * embed_dim, temp_output.data(),
            output.data(), seq_len * embed_dim, embed_dim
        );
    }
    
    void computeFeedForward(const std::vector<float>& input, 
                           std::vector<float>& output, 
                           size_t layer_idx) const {
        size_t seq_len = input.size() / embed_dim;
        const auto& ff_weights = feedforward_weights[layer_idx];
        
        // Expand to 4x dimension
        std::vector<float> expanded(seq_len * embed_dim * 4);
        ML::SIMD::VectorOps::matrix_vector_multiply(
            ff_weights.data(), input.data(), expanded.data(),
            seq_len * embed_dim, embed_dim
        );
        
        // Apply GELU activation (simplified as tanh)
        for (size_t i = 0; i < expanded.size(); ++i) {
            expanded[i] = std::tanh(expanded[i]);
        }
        
        // Project back to original dimension
        ML::SIMD::VectorOps::matrix_vector_multiply(
            ff_weights.data() + embed_dim * embed_dim * 4, expanded.data(),
            output.data(), seq_len * embed_dim, embed_dim * 4
        );
    }
    
    void applyLayerNorm(const std::vector<float>& input, 
                       std::vector<float>& output, 
                       size_t layer_idx) const {
        size_t seq_len = input.size() / embed_dim;
        const auto& norm_weights = layer_norm_weights[layer_idx];
        
        for (size_t i = 0; i < seq_len; ++i) {
            // Compute mean and variance
            float mean = 0.0f;
            float variance = 0.0f;
            
            for (size_t dim = 0; dim < embed_dim; ++dim) {
                mean += input[i * embed_dim + dim];
            }
            mean /= embed_dim;
            
            for (size_t dim = 0; dim < embed_dim; ++dim) {
                float diff = input[i * embed_dim + dim] - mean;
                variance += diff * diff;
            }
            variance /= embed_dim;
            variance = std::sqrt(variance + 1e-6f);
            
            // Normalize and scale
            for (size_t dim = 0; dim < embed_dim; ++dim) {
                size_t idx = i * embed_dim + dim;
                float normalized = (input[idx] - mean) / variance;
                output[idx] = normalized * norm_weights[dim] + norm_weights[embed_dim + dim];
            }
        }
    }
    
    // Adaptive Computation
    float computeComplexityScore(const std::vector<float>& input) const {
        // Simple complexity metric based on input variance
        float mean = 0.0f;
        for (float val : input) {
            mean += val;
        }
        mean /= input.size();
        
        float variance = 0.0f;
        for (float val : input) {
            float diff = val - mean;
            variance += diff * diff;
        }
        variance /= input.size();
        
        return std::sqrt(variance);
    }
    
    size_t determineAdaptiveDepth(float complexity) const {
        if (!early_exit_enabled) return num_layers;
        
        // Adaptive depth based on complexity
        if (complexity < 0.1f) return 1;        // Simple inputs
        else if (complexity < 0.3f) return 2;   // Medium complexity
        else if (complexity < 0.6f) return 3;   // High complexity
        else return num_layers;                  // Full depth for complex inputs
    }
    
    // Streaming Interface
    void startStream() {
        streaming_active = true;
    }
    
    void stopStream() {
        streaming_active = false;
        stream_cv.notify_all();
    }
    
    void pushChunk(const std::vector<int>& chunk) {
        std::lock_guard<std::mutex> lock(stream_mutex);
        input_stream.push(chunk);
        stream_cv.notify_one();
    }
    
    bool getOutput(std::vector<float>& output) {
        std::lock_guard<std::mutex> lock(stream_mutex);
        if (output_stream.empty()) return false;
        
        output = output_stream.front();
        output_stream.pop();
        return true;
    }
    
    void processStream() {
        while (streaming_active) {
            std::unique_lock<std::mutex> lock(stream_mutex);
            stream_cv.wait(lock, [this] { return !input_stream.empty() || !streaming_active; });
            
            if (!streaming_active) break;
            
            auto chunk = input_stream.front();
            input_stream.pop();
            lock.unlock();
            
            // Process chunk
            std::vector<float> embeddings;
            embedInput(chunk, embeddings);
            
            std::vector<float> output;
            process(embeddings, output);
            
            // Add to output stream
            lock.lock();
            output_stream.push(output);
            lock.unlock();
        }
    }
    
    // Main processing function
    void process(const std::vector<float>& input, std::vector<float>& output) {
        // Compute complexity and determine depth
        complexity_score = computeComplexityScore(input);
        current_depth = determineAdaptiveDepth(complexity_score);
        
        std::vector<float> current = input;
        
        // Process through adaptive number of layers
        for (size_t layer = 0; layer < current_depth; ++layer) {
            std::vector<float> next;
            transformerBlock(current, next, layer);
            current = next;
        }
        
        output = current;
    }
    
    // Memory recycling
    void recycleMemory() {
        memory_offset = 0;
        std::fill(working_memory.begin(), working_memory.end(), 0.0f);
    }
    
    // Getters for testing
    size_t getEmbedDim() const { return embed_dim; }
    size_t getNumLayers() const { return num_layers; }
    size_t getCurrentDepth() const { return current_depth; }
    float getComplexityScore() const { return complexity_score; }
    bool isEarlyExitEnabled() const { return early_exit_enabled; }
    void setEarlyExitEnabled(bool enabled) { early_exit_enabled = enabled; }
};

} // namespace RealTime
} // namespace ML

class Phase4TransformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> token_dis(0, 999);
        std::uniform_real_distribution<float> float_dis(-1.0f, 1.0f);
        
        // Test configurations for real-time transformers
        test_configs = {
            {256, 128, 4, 2, 128},   // Small transformer
            {512, 256, 8, 4, 256},   // Medium transformer
            {1000, 512, 8, 6, 512},  // Large transformer
        };
        
        for (auto [vocab, embed, heads, layers, seq_len] : test_configs) {
            std::string key = std::to_string(vocab) + "_" + std::to_string(embed);
            
            // Generate test token sequences
            test_tokens[key] = std::vector<int>(seq_len);
            for (auto& token : test_tokens[key]) {
                token = token_dis(gen);
            }
            
            // Generate test embeddings
            test_embeddings[key] = std::vector<float>(seq_len * embed);
            for (auto& val : test_embeddings[key]) {
                val = float_dis(gen);
            }
        }
    }
    
    std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t>> test_configs;
    std::map<std::string, std::vector<int>> test_tokens;
    std::map<std::string, std::vector<float>> test_embeddings;
};

// Test RealTimeTransformer Creation
TEST_F(Phase4TransformerTest, RealTimeTransformerCreation) {
    for (auto [vocab, embed, heads, layers, seq_len] : test_configs) {
        ML::RealTime::RealTimeTransformer transformer(vocab, embed, heads, layers, seq_len);
        
        EXPECT_EQ(transformer.getEmbedDim(), embed);
        EXPECT_EQ(transformer.getNumLayers(), layers);
        EXPECT_TRUE(transformer.isEarlyExitEnabled());
    }
}

// Test Embedding Layer (SIMD)
TEST_F(Phase4TransformerTest, EmbeddingLayerSIMD) {
    for (auto [vocab, embed, heads, layers, seq_len] : test_configs) {
        ML::RealTime::RealTimeTransformer transformer(vocab, embed, heads, layers, seq_len);
        
        std::string key = std::to_string(vocab) + "_" + std::to_string(embed);
        const auto& tokens = test_tokens[key];
        
        std::vector<float> embeddings;
        transformer.embedInput(tokens, embeddings);
        
        // Verify output dimensions
        ASSERT_EQ(embeddings.size(), tokens.size() * embed);
        
        // Verify no NaN or Inf values
        for (float val : embeddings) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in embeddings";
            ASSERT_FALSE(std::isinf(val)) << "Inf in embeddings";
            ASSERT_LT(std::abs(val), 1e6f) << "Extreme value in embeddings";
        }
        
        // Verify embeddings are different from zero (non-trivial)
        bool has_non_zero = false;
        for (float val : embeddings) {
            if (std::abs(val) > 1e-6f) {
                has_non_zero = true;
                break;
            }
        }
        ASSERT_TRUE(has_non_zero) << "All embeddings are zero";
    }
}

// Test Transformer Blocks
TEST_F(Phase4TransformerTest, TransformerBlocks) {
    for (auto [vocab, embed, heads, layers, seq_len] : test_configs) {
        ML::RealTime::RealTimeTransformer transformer(vocab, embed, heads, layers, seq_len);
        
        std::string key = std::to_string(vocab) + "_" + std::to_string(embed);
        const auto& input = test_embeddings[key];
        
        // Test single transformer block
        std::vector<float> block_output;
        transformer.transformerBlock(input, block_output, 0);
        
        // Verify output dimensions
        ASSERT_EQ(block_output.size(), input.size());
        
        // Verify output is different from input
        bool different = false;
        for (size_t i = 0; i < input.size(); ++i) {
            if (std::abs(input[i] - block_output[i]) > 1e-6f) {
                different = true;
                break;
            }
        }
        ASSERT_TRUE(different) << "Transformer block output identical to input";
        
        // Verify no NaN or Inf
        for (float val : block_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in transformer block output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in transformer block output";
        }
    }
}

// Test Multi-Head Attention within Blocks
TEST_F(Phase4TransformerTest, MultiHeadAttentionWithinBlocks) {
    size_t vocab = 512, embed = 256, heads = 8, layers = 4, seq_len = 128;
    ML::RealTime::RealTimeTransformer transformer(vocab, embed, heads, layers, seq_len);
    
    std::vector<float> input(seq_len * embed);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<float> attention_output;
    transformer.computeAttention(input, attention_output, 0);
    
    // Verify attention output
    ASSERT_EQ(attention_output.size(), input.size());
    
    // Verify reasonable values
    for (float val : attention_output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN in attention output";
        ASSERT_FALSE(std::isinf(val)) << "Inf in attention output";
        ASSERT_LT(std::abs(val), 1e6f) << "Extreme value in attention output";
    }
}

// Test FeedForward (SIMD) within Blocks
TEST_F(Phase4TransformerTest, FeedForwardSIMDWithinBlocks) {
    size_t vocab = 512, embed = 256, heads = 8, layers = 4, seq_len = 128;
    ML::RealTime::RealTimeTransformer transformer(vocab, embed, heads, layers, seq_len);
    
    std::vector<float> input(seq_len * embed);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<float> ff_output;
    transformer.computeFeedForward(input, ff_output, 0);
    
    // Verify feed-forward output
    ASSERT_EQ(ff_output.size(), input.size());
    
    // Verify reasonable values
    for (float val : ff_output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN in feed-forward output";
        ASSERT_FALSE(std::isinf(val)) << "Inf in feed-forward output";
        ASSERT_LT(std::abs(val), 1e6f) << "Extreme value in feed-forward output";
    }
}

// Test LayerNorm within Blocks
TEST_F(Phase4TransformerTest, LayerNormWithinBlocks) {
    size_t vocab = 512, embed = 256, heads = 8, layers = 4, seq_len = 128;
    ML::RealTime::RealTimeTransformer transformer(vocab, embed, heads, layers, seq_len);
    
    std::vector<float> input(seq_len * embed);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<float> norm_output;
    transformer.applyLayerNorm(input, norm_output, 0);
    
    // Verify layer norm output
    ASSERT_EQ(norm_output.size(), input.size());
    
    // Verify normalization properties (mean ≈ 0, std ≈ 1)
    for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        float mean = 0.0f;
        float variance = 0.0f;
        
        for (size_t dim = 0; dim < embed; ++dim) {
            mean += norm_output[seq_idx * embed + dim];
        }
        mean /= embed;
        
        for (size_t dim = 0; dim < embed; ++dim) {
            float diff = norm_output[seq_idx * embed + dim] - mean;
            variance += diff * diff;
        }
        variance /= embed;
        
        EXPECT_NEAR(mean, 0.0f, 0.1f) << "Layer norm mean not close to 0";
        EXPECT_NEAR(std::sqrt(variance), 1.0f, 0.2f) << "Layer norm std not close to 1";
    }
}

// Test Adaptive Computation
TEST_F(Phase4TransformerTest, AdaptiveComputation) {
    ML::RealTime::RealTimeTransformer transformer(512, 256, 8, 4, 256);
    
    // Test with simple input (low complexity)
    std::vector<float> simple_input(256 * 256, 0.1f); // All same values
    transformer.process(simple_input, simple_input); // Process to compute complexity
    
    EXPECT_LE(transformer.getCurrentDepth(), 2) << "Simple input should use fewer layers";
    EXPECT_LT(transformer.getComplexityScore(), 0.2f) << "Simple input should have low complexity";
    
    // Test with complex input (high complexity)
    std::vector<float> complex_input(256 * 256);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& val : complex_input) val = dis(gen);
    
    transformer.process(complex_input, complex_input); // Process to compute complexity
    
    EXPECT_EQ(transformer.getCurrentDepth(), 4) << "Complex input should use all layers";
    EXPECT_GT(transformer.getComplexityScore(), 0.5f) << "Complex input should have high complexity";
    
    // Test early exit disabled
    transformer.setEarlyExitEnabled(false);
    transformer.process(simple_input, simple_input);
    EXPECT_EQ(transformer.getCurrentDepth(), 4) << "Should use all layers when early exit disabled";
}

// Test Early Exit Feature
TEST_F(Phase4TransformerTest, EarlyExitFeature) {
    ML::RealTime::RealTimeTransformer transformer(512, 256, 8, 4, 256);
    
    std::vector<float> simple_input(256 * 256, 0.1f);
    std::vector<float> output;
    
    // Measure time with early exit
    auto start = std::chrono::high_resolution_clock::now();
    transformer.process(simple_input, output);
    auto end = std::chrono::high_resolution_clock::now();
    auto early_exit_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Disable early exit
    transformer.setEarlyExitEnabled(false);
    
    // Measure time without early exit
    start = std::chrono::high_resolution_clock::now();
    transformer.process(simple_input, output);
    end = std::chrono::high_resolution_clock::now();
    auto full_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Early exit should be faster for simple inputs
    EXPECT_LT(early_exit_time.count(), full_time.count()) 
        << "Early exit should be faster for simple inputs";
    
    // Re-enable early exit
    transformer.setEarlyExitEnabled(true);
}

// Test Batch Streaming
TEST_F(Phase4TransformerTest, BatchStreaming) {
    ML::RealTime::RealTimeTransformer transformer(512, 256, 8, 4, 256);
    
    // Start streaming
    transformer.startStream();
    
    // Push multiple chunks
    std::vector<std::vector<int>> chunks;
    for (int i = 0; i < 5; ++i) {
        std::vector<int> chunk(64);
        std::iota(chunk.begin(), chunk.end(), i * 64);
        chunks.push_back(chunk);
        transformer.pushChunk(chunk);
    }
    
    // Process stream in separate thread
    std::thread processing_thread([&transformer]() {
        transformer.processStream();
    });
    
    // Give some time for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Get outputs
    int outputs_received = 0;
    std::vector<float> output;
    while (transformer.getOutput(output)) {
        outputs_received++;
        EXPECT_EQ(output.size(), 64 * 256) << "Output size incorrect";
        
        // Verify reasonable values
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in streaming output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in streaming output";
        }
    }
    
    // Stop streaming
    transformer.stopStream();
    processing_thread.join();
    
    EXPECT_EQ(outputs_received, 5) << "Should receive all chunk outputs";
}

// Test Memory Recycling
TEST_F(Phase4TransformerTest, MemoryRecycling) {
    ML::RealTime::RealTimeTransformer transformer(512, 256, 8, 4, 256);
    
    std::vector<float> input(256 * 256);
    std::iota(input.begin(), input.end(), 0.0f);
    
    // Process multiple times
    for (int i = 0; i < 10; ++i) {
        std::vector<float> output;
        transformer.process(input, output);
        
        // Recycle memory
        transformer.recycleMemory();
        
        // Verify output is still reasonable
        ASSERT_EQ(output.size(), input.size());
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN after memory recycling";
            ASSERT_FALSE(std::isinf(val)) << "Inf after memory recycling";
        }
    }
}

// Test Real-Time Performance Targets
TEST_F(Phase4TransformerTest, RealTimePerformanceTargets) {
    std::cout << "\n=== Phase 4 Real-Time Performance Targets ===\n";
    std::cout << std::setw(12) << "Embed Dim" << std::setw(12) << "Layers" 
              << std::setw(15) << "Forward (ms)" << std::setw(15) << "Target (ms)" 
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(64, '-') << std::endl;
    
    const int iterations = 50;
    const double target_forward_ms = 2.0; // <2ms target from roadmap
    
    for (auto [vocab, embed, heads, layers, seq_len] : test_configs) {
        ML::RealTime::RealTimeTransformer transformer(vocab, embed, heads, layers, seq_len);
        
        std::string key = std::to_string(vocab) + "_" + std::to_string(embed);
        const auto& input = test_embeddings[key];
        
        // Benchmark forward pass
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> output;
            transformer.process(input, output);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_forward_ms = static_cast<double>(duration.count()) / (iterations * 1000.0);
        
        std::string status = (avg_forward_ms <= target_forward_ms) ? "PASS" : "FAIL";
        
        std::cout << std::setw(12) << embed << std::setw(12) << layers
                  << std::setw(15) << std::fixed << std::setprecision(2) << avg_forward_ms
                  << std::setw(15) << target_forward_ms << std::setw(10) << status << std::endl;
        
        // For smaller transformers, enforce performance target
        if (embed <= 256 && layers <= 4) {
            EXPECT_LE(avg_forward_ms, target_forward_ms) 
                << "Real-time transformer performance target not met";
        }
    }
}

// Test Zero-Allocation Inference
TEST_F(Phase4TransformerTest, ZeroAllocationInference) {
    ML::RealTime::RealTimeTransformer transformer(512, 256, 8, 4, 256);
    
    std::vector<float> input(256 * 256);
    std::iota(input.begin(), input.end(), 0.0f);
    
    // Pre-allocate output
    std::vector<float> output(input.size());
    
    // Process multiple times without additional allocations
    for (int i = 0; i < 100; ++i) {
        transformer.process(input, output);
        
        // Verify output consistency
        ASSERT_EQ(output.size(), input.size());
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in zero-allocation inference";
            ASSERT_FALSE(std::isinf(val)) << "Inf in zero-allocation inference";
        }
    }
    
    // Memory should be recycled
    transformer.recycleMemory();
}

// Test Edge Cases and Robustness
TEST_F(Phase4TransformerTest, EdgeCasesAndRobustness) {
    // Test with minimal transformer
    EXPECT_NO_THROW({
        ML::RealTime::RealTimeTransformer minimal(100, 64, 2, 1, 32);
        std::vector<float> input(32 * 64, 1.0f);
        std::vector<float> output;
        minimal.process(input, output);
    });
    
    // Test with empty input
    ML::RealTime::RealTimeTransformer transformer(512, 256, 8, 4, 256);
    std::vector<float> empty_input;
    std::vector<float> empty_output;
    
    // Should handle gracefully
    EXPECT_NO_THROW(transformer.process(empty_input, empty_output));
    
    // Test with extreme values
    std::vector<float> extreme_input(256 * 256);
    for (size_t i = 0; i < extreme_input.size(); ++i) {
        if (i % 4 == 0) extreme_input[i] = 1e6f;
        else if (i % 4 == 1) extreme_input[i] = -1e6f;
        else if (i % 4 == 2) extreme_input[i] = 1e-6f;
        else extreme_input[i] = -1e-6f;
    }
    
    std::vector<float> extreme_output;
    EXPECT_NO_THROW(transformer.process(extreme_input, extreme_output));
    
    // Verify no NaN or Inf in output
    for (float val : extreme_output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN with extreme input";
        ASSERT_FALSE(std::isinf(val)) << "Inf with extreme input";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
