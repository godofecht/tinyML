//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 2 Attention Mechanism Core Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

// Mock/Placeholder includes for attention mechanism components
// These would be implemented according to the roadmap
#include "SIMDOperations.h"

namespace ML {
namespace Attention {

// Mock LightweightAttention class (to be implemented in Phase 2)
class LightweightAttention {
private:
    size_t seq_len;
    size_t embed_dim;
    size_t num_heads;
    float scale_factor;
    
    // Internal matrices for Q, K, V projections
    std::vector<float> q_proj_weights;
    std::vector<float> k_proj_weights; 
    std::vector<float> v_proj_weights;
    std::vector<float> output_proj_weights;
    
public:
    LightweightAttention(size_t seq_len, size_t embed_dim, size_t num_heads = 8)
        : seq_len(seq_len), embed_dim(embed_dim), num_heads(num_heads) {
        scale_factor = 1.0f / std::sqrt(embed_dim / num_heads);
        
        // Initialize projection weights
        size_t head_dim = embed_dim / num_heads;
        q_proj_weights.resize(embed_dim * embed_dim);
        k_proj_weights.resize(embed_dim * embed_dim);
        v_proj_weights.resize(embed_dim * embed_dim);
        output_proj_weights.resize(embed_dim * embed_dim);
        
        // Random initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
        
        for (auto& w : q_proj_weights) w = dis(gen);
        for (auto& w : k_proj_weights) w = dis(gen);
        for (auto& w : v_proj_weights) w = dis(gen);
        for (auto& w : output_proj_weights) w = dis(gen);
    }
    
    // QKV Projection (SIMD-optimized)
    void computeQKV(const std::vector<float>& input, 
                    std::vector<float>& q, 
                    std::vector<float>& k, 
                    std::vector<float>& v) const {
        q.resize(seq_len * embed_dim);
        k.resize(seq_len * embed_dim);
        v.resize(seq_len * embed_dim);
        
        // Use SIMD matrix-vector multiplication
        for (size_t i = 0; i < seq_len; ++i) {
            // Extract input sequence element
            std::vector<float> input_seq(embed_dim);
            for (size_t j = 0; j < embed_dim; ++j) {
                input_seq[j] = input[i * embed_dim + j];
            }
            
            // Compute Q, K, V projections
            ML::SIMD::VectorOps::matrix_vector_multiply(
                q_proj_weights.data(), input_seq.data(), 
                q.data() + i * embed_dim, embed_dim, embed_dim
            );
            
            ML::SIMD::VectorOps::matrix_vector_multiply(
                k_proj_weights.data(), input_seq.data(), 
                k.data() + i * embed_dim, embed_dim, embed_dim
            );
            
            ML::SIMD::VectorOps::matrix_vector_multiply(
                v_proj_weights.data(), input_seq.data(), 
                v.data() + i * embed_dim, embed_dim, embed_dim
            );
        }
    }
    
    // Scaled Dot-Product Attention
    void scaledDotProductAttention(const std::vector<float>& q,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  std::vector<float>& output) const {
        output.resize(seq_len * embed_dim);
        
        size_t head_dim = embed_dim / num_heads;
        
        // For each head
        for (size_t head = 0; head < num_heads; ++head) {
            size_t head_offset = head * head_dim;
            
            // Compute attention scores: Q * K^T
            std::vector<float> attention_scores(seq_len * seq_len);
            
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    // Dot product of Q[i] and K[j]
                    float score = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        score += q[i * embed_dim + head_offset + d] * 
                                k[j * embed_dim + head_offset + d];
                    }
                    attention_scores[i * seq_len + j] = score * scale_factor;
                }
            }
            
            // Apply softmax to get attention weights
            std::vector<float> attention_weights(seq_len * seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                // Find max for numerical stability
                float max_score = attention_scores[i * seq_len];
                for (size_t j = 1; j < seq_len; ++j) {
                    max_score = std::max(max_score, attention_scores[i * seq_len + j]);
                }
                
                // Compute exp and sum
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    float exp_val = std::exp(attention_scores[i * seq_len + j] - max_score);
                    attention_weights[i * seq_len + j] = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize
                for (size_t j = 0; j < seq_len; ++j) {
                    attention_weights[i * seq_len + j] /= sum_exp;
                }
            }
            
            // Apply attention weights to V: AttentionWeights * V
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    float weighted_sum = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        weighted_sum += attention_weights[i * seq_len + j] * 
                                       v[j * embed_dim + head_offset + d];
                    }
                    output[i * embed_dim + head_offset + d] = weighted_sum;
                }
            }
        }
    }
    
    // Multi-Head Concatenation and Output Projection
    void multiHeadConcatenation(const std::vector<float>& attention_output,
                               std::vector<float>& final_output) const {
        final_output.resize(seq_len * embed_dim);
        
        // Output projection
        for (size_t i = 0; i < seq_len; ++i) {
            std::vector<float> input_vec(embed_dim);
            for (size_t j = 0; j < embed_dim; ++j) {
                input_vec[j] = attention_output[i * embed_dim + j];
            }
            
            ML::SIMD::VectorOps::matrix_vector_multiply(
                output_proj_weights.data(), input_vec.data(),
                final_output.data() + i * embed_dim, embed_dim, embed_dim
            );
        }
    }
    
    // Complete forward pass
    void forward(const std::vector<float>& input, std::vector<float>& output) const {
        std::vector<float> q, k, v, attention_output;
        
        computeQKV(input, q, k, v);
        scaledDotProductAttention(q, k, v, attention_output);
        multiHeadConcatenation(attention_output, output);
    }
    
    // Getters for testing
    size_t getSeqLen() const { return seq_len; }
    size_t getEmbedDim() const { return embed_dim; }
    size_t getNumHeads() const { return num_heads; }
};

} // namespace Attention
} // namespace ML

class Phase2AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test configurations from roadmap
        test_configs = {
            {128, 64, 4},   // Small: 128 seq, 64 embed, 4 heads
            {256, 128, 8},  // Medium: 256 seq, 128 embed, 8 heads  
            {512, 256, 8},  // Large: 512 seq, 256 embed, 8 heads
        };
        
        for (auto [seq_len, embed_dim, num_heads] : test_configs) {
            std::string key = std::to_string(seq_len) + "_" + std::to_string(embed_dim);
            test_inputs[key] = std::vector<float>(seq_len * embed_dim);
            
            for (size_t i = 0; i < seq_len * embed_dim; ++i) {
                test_inputs[key][i] = dis(gen);
            }
        }
    }
    
    std::vector<std::tuple<size_t, size_t, size_t>> test_configs;
    std::map<std::string, std::vector<float>> test_inputs;
};

// Test LightweightAttention Class Creation
TEST_F(Phase2AttentionTest, LightweightAttentionCreation) {
    for (auto [seq_len, embed_dim, num_heads] : test_configs) {
        ML::Attention::LightweightAttention attention(seq_len, embed_dim, num_heads);
        
        EXPECT_EQ(attention.getSeqLen(), seq_len);
        EXPECT_EQ(attention.getEmbedDim(), embed_dim);
        EXPECT_EQ(attention.getNumHeads(), num_heads);
    }
}

// Test QKV Projection Implementation
TEST_F(Phase2AttentionTest, QKVProjection) {
    for (auto [seq_len, embed_dim, num_heads] : test_configs) {
        ML::Attention::LightweightAttention attention(seq_len, embed_dim, num_heads);
        
        std::string key = std::to_string(seq_len) + "_" + std::to_string(embed_dim);
        const auto& input = test_inputs[key];
        
        std::vector<float> q, k, v;
        attention.computeQKV(input, q, k, v);
        
        // Verify output dimensions
        ASSERT_EQ(q.size(), seq_len * embed_dim);
        ASSERT_EQ(k.size(), seq_len * embed_dim);
        ASSERT_EQ(v.size(), seq_len * embed_dim);
        
        // Verify no NaN or Inf values
        for (size_t i = 0; i < q.size(); ++i) {
            ASSERT_FALSE(std::isnan(q[i])) << "NaN in Q at index " << i;
            ASSERT_FALSE(std::isinf(q[i])) << "Inf in Q at index " << i;
            ASSERT_FALSE(std::isnan(k[i])) << "NaN in K at index " << i;
            ASSERT_FALSE(std::isinf(k[i])) << "Inf in K at index " << i;
            ASSERT_FALSE(std::isnan(v[i])) << "NaN in V at index " << i;
            ASSERT_FALSE(std::isinf(v[i])) << "Inf in V at index " << i;
        }
    }
}

// Test Scaled Dot-Product Attention
TEST_F(Phase2AttentionTest, ScaledDotProductAttention) {
    for (auto [seq_len, embed_dim, num_heads] : test_configs) {
        ML::Attention::LightweightAttention attention(seq_len, embed_dim, num_heads);
        
        std::string key = std::to_string(seq_len) + "_" + std::to_string(embed_dim);
        const auto& input = test_inputs[key];
        
        std::vector<float> q, k, v, attention_output;
        attention.computeQKV(input, q, k, v);
        attention.scaledDotProductAttention(q, k, v, attention_output);
        
        // Verify output dimensions
        ASSERT_EQ(attention_output.size(), seq_len * embed_dim);
        
        // Verify output is reasonable (not all zeros, not extreme values)
        bool has_non_zero = false;
        for (float val : attention_output) {
            if (std::abs(val) > 1e-6f) {
                has_non_zero = true;
                break;
            }
        }
        ASSERT_TRUE(has_non_zero) << "Attention output is all zeros";
        
        // Verify no NaN or Inf
        for (float val : attention_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in attention output";
            ASSERT_FALSE(std::isinf(val)) << "Inf in attention output";
            ASSERT_LT(std::abs(val), 1e6f) << "Extreme value in attention output: " << val;
        }
    }
}

// Test Multi-Head Concatenation
TEST_F(Phase2AttentionTest, MultiHeadConcatenation) {
    for (auto [seq_len, embed_dim, num_heads] : test_configs) {
        ML::Attention::LightweightAttention attention(seq_len, embed_dim, num_heads);
        
        std::string key = std::to_string(seq_len) + "_" + std::to_string(embed_dim);
        const auto& input = test_inputs[key];
        
        std::vector<float> output;
        attention.forward(input, output);
        
        // Verify final output dimensions
        ASSERT_EQ(output.size(), seq_len * embed_dim);
        
        // Verify output is different from input (transformation occurred)
        bool different = false;
        for (size_t i = 0; i < input.size(); ++i) {
            if (std::abs(input[i] - output[i]) > 1e-6f) {
                different = true;
                break;
            }
        }
        ASSERT_TRUE(different) << "Output identical to input - no transformation";
    }
}

// Test Real-Time Performance Targets (<1ms latency for 512-sequence)
TEST_F(Phase2AttentionTest, RealTimePerformanceTargets) {
    std::cout << "\n=== Phase 2 Real-Time Performance Targets ===\n";
    std::cout << std::setw(12) << "Seq Len" << std::setw(12) << "Embed Dim" 
              << std::setw(15) << "Latency (ms)" << std::setw(12) << "Target (ms)" 
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(61, '-') << std::endl;
    
    const int iterations = 100;
    const double target_latency_ms = 1.0; // <1ms target from roadmap
    
    for (auto [seq_len, embed_dim, num_heads] : test_configs) {
        ML::Attention::LightweightAttention attention(seq_len, embed_dim, num_heads);
        
        std::string key = std::to_string(seq_len) + "_" + std::to_string(embed_dim);
        const auto& input = test_inputs[key];
        
        // Benchmark forward pass
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> output;
            attention.forward(input, output);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_latency_ms = static_cast<double>(duration.count()) / (iterations * 1000.0);
        
        std::string status = (avg_latency_ms <= target_latency_ms) ? "PASS" : "FAIL";
        
        std::cout << std::setw(12) << seq_len << std::setw(12) << embed_dim
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_latency_ms
                  << std::setw(12) << target_latency_ms << std::setw(10) << status << std::endl;
        
        // For 512-sequence, enforce the <1ms target
        if (seq_len == 512) {
            EXPECT_LE(avg_latency_ms, target_latency_ms) 
                << "512-sequence attention latency target not met: " 
                << avg_latency_ms << "ms > " << target_latency_ms << "ms";
        }
    }
}

// Test Memory Efficiency (<10MB memory for complete transformer)
TEST_F(Phase2AttentionTest, MemoryEfficiencyValidation) {
    std::cout << "\n=== Memory Efficiency Validation ===\n";
    std::cout << std::setw(12) << "Seq Len" << std::setw(12) << "Embed Dim" 
              << std::setw(15) << "Est. Memory (MB)" << std::setw(12) << "Target (MB)" 
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(61, '-') << std::endl;
    
    const double target_memory_mb = 10.0; // <10MB target from roadmap
    
    for (auto [seq_len, embed_dim, num_heads] : test_configs) {
        // Estimate memory usage for attention mechanism
        size_t input_memory = seq_len * embed_dim * sizeof(float);
        size_t qkv_memory = 3 * seq_len * embed_dim * sizeof(float);
        size_t attention_scores_memory = seq_len * seq_len * sizeof(float);
        size_t output_memory = seq_len * embed_dim * sizeof(float);
        size_t weight_memory = 4 * embed_dim * embed_dim * sizeof(float); // Q,K,V,O projections
        
        size_t total_memory_bytes = input_memory + qkv_memory + attention_scores_memory + 
                                  output_memory + weight_memory;
        double total_memory_mb = static_cast<double>(total_memory_bytes) / (1024 * 1024);
        
        std::string status = (total_memory_mb <= target_memory_mb) ? "PASS" : "FAIL";
        
        std::cout << std::setw(12) << seq_len << std::setw(12) << embed_dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << total_memory_mb
                  << std::setw(12) << target_memory_mb << std::setw(10) << status << std::endl;
        
        EXPECT_LE(total_memory_mb, target_memory_mb) 
            << "Memory efficiency target not met: " << total_memory_mb << "MB > " << target_memory_mb << "MB";
    }
}

// Test Throughput (>1000 sequences/second)
TEST_F(Phase2AttentionTest, ThroughputValidation) {
    std::cout << "\n=== Throughput Validation ===\n";
    std::cout << std::setw(12) << "Seq Len" << std::setw(12) << "Embed Dim" 
              << std::setw(15) << "Throughput (seq/s)" << std::setw(12) << "Target (seq/s)" 
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(61, '-') << std::endl;
    
    const double target_throughput = 1000.0; // >1000 sequences/second from roadmap
    
    for (auto [seq_len, embed_dim, num_heads] : test_configs) {
        ML::Attention::LightweightAttention attention(seq_len, embed_dim, num_heads);
        
        std::string key = std::to_string(seq_len) + "_" + std::to_string(embed_dim);
        const auto& input = test_inputs[key];
        
        // Measure throughput
        const int test_duration_ms = 1000; // 1 second test
        auto start_time = std::chrono::high_resolution_clock::now();
        int sequences_processed = 0;
        
        while (true) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            
            if (elapsed.count() >= test_duration_ms) break;
            
            std::vector<float> output;
            attention.forward(input, output);
            sequences_processed++;
        }
        
        double throughput = static_cast<double>(sequences_processed) / (test_duration_ms / 1000.0);
        std::string status = (throughput >= target_throughput) ? "PASS" : "FAIL";
        
        std::cout << std::setw(12) << seq_len << std::setw(12) << embed_dim
                  << std::setw(15) << std::fixed << std::setprecision(0) << throughput
                  << std::setw(12) << target_throughput << std::setw(10) << status << std::endl;
        
        // For smaller sequences, enforce throughput target
        if (seq_len <= 256) {
            EXPECT_GE(throughput, target_throughput) 
                << "Throughput target not met: " << throughput << " seq/s < " << target_throughput << " seq/s";
        }
    }
}

// Test SIMD Optimization Integration
TEST_F(Phase2AttentionTest, SIMDOptimizationIntegration) {
    // Test that attention mechanism uses SIMD operations effectively
    size_t seq_len = 512;
    size_t embed_dim = 256;
    size_t num_heads = 8;
    
    ML::Attention::LightweightAttention attention(seq_len, embed_dim, num_heads);
    
    std::string key = std::to_string(seq_len) + "_" + std::to_string(embed_dim);
    const auto& input = test_inputs[key];
    
    // Verify that SIMD operations are being used by checking performance
    // This is a proxy test - actual SIMD usage would be verified in implementation
    std::vector<float> output;
    auto start = std::chrono::high_resolution_clock::now();
    attention.forward(input, output);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete in reasonable time (indicating SIMD optimization)
    EXPECT_LT(duration.count(), 10000) << "Attention forward pass too slow - likely not using SIMD";
    
    // Verify output correctness
    ASSERT_EQ(output.size(), seq_len * embed_dim);
    
    // Check for reasonable output values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN in attention output";
        ASSERT_FALSE(std::isinf(val)) << "Inf in attention output";
    }
}

// Test Edge Cases
TEST_F(Phase2AttentionTest, EdgeCases) {
    // Test with minimal sequence length
    ML::Attention::LightweightAttention minimal_attention(1, 64, 4);
    std::vector<float> minimal_input(64, 1.0f);
    std::vector<float> minimal_output;
    
    minimal_attention.forward(minimal_input, minimal_output);
    
    ASSERT_EQ(minimal_output.size(), 64);
    
    // Test with single head
    ML::Attention::LightweightAttention single_head_attention(128, 64, 1);
    std::vector<float> single_head_input(128 * 64);
    std::iota(single_head_input.begin(), single_head_input.end(), 0.0f);
    std::vector<float> single_head_output;
    
    single_head_attention.forward(single_head_input, single_head_output);
    
    ASSERT_EQ(single_head_output.size(), 128 * 64);
    
    // Verify no crashes with edge cases
    EXPECT_NO_THROW({
        ML::Attention::LightweightAttention edge_attention(2, 4, 2);
        std::vector<float> edge_input(8, 0.5f);
        std::vector<float> edge_output;
        edge_attention.forward(edge_input, edge_output);
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
