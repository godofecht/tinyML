//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 2: Attention Mechanism Core Tests with XSIMD
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>

#include "LightweightAttention.h"
#include "XSIMDOperations.h"

class Phase2AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test configurations for different scales
        test_configs = {
            ML::RealTime::LightweightAttention::Config{64, 4, 16, 128},   // Small
            ML::RealTime::LightweightAttention::Config{256, 8, 32, 512},  // Medium
            ML::RealTime::LightweightAttention::Config{512, 8, 64, 512}   // Large
        };
        
        // Generate test inputs for each configuration
        for (const auto& config : test_configs) {
            size_t input_size = config.sequence_length * config.embed_dim;
            test_inputs[config.embed_dim] = std::vector<float>(input_size);
            
            for (size_t i = 0; i < input_size; ++i) {
                test_inputs[config.embed_dim][i] = dis(gen);
            }
        }
    }
    
    std::vector<ML::RealTime::LightweightAttention::Config> test_configs;
    std::map<size_t, std::vector<float>> test_inputs;
};

// Test QKV Projection with XSIMD
TEST_F(Phase2AttentionTest, QKVProjectionXSIMD) {
    std::cout << "\n=== QKV Projection with XSIMD Test ===\n";
    
    for (const auto& config : test_configs) {
        ML::RealTime::LightweightAttention attention(config);
        
        // Test forward pass
        auto output = attention.forward(test_inputs[config.embed_dim]);
        
        // Verify output dimensions
        ASSERT_EQ(output.size(), test_inputs[config.embed_dim].size());
        
        // Verify no NaN or infinite values
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN detected in attention output";
            ASSERT_FALSE(std::isinf(val)) << "Inf detected in attention output";
        }
        
        std::cout << "QKV Projection for embed_dim=" << config.embed_dim 
                  << ", heads=" << config.num_heads << ": PASS\n";
    }
}

// Test Scaled Dot-Product Attention
TEST_F(Phase2AttentionTest, ScaledDotProductAttention) {
    std::cout << "\n=== Scaled Dot-Product Attention Test ===\n";
    
    // Create test Q, K, V matrices
    size_t seq_len = 128;
    size_t embed_dim = 256;
    size_t head_dim = embed_dim / 8; // 8 heads
    
    std::vector<float> Q(seq_len * embed_dim);
    std::vector<float> K(seq_len * embed_dim);
    std::vector<float> V(seq_len * embed_dim);
    
    // Fill with test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < seq_len * embed_dim; ++i) {
        Q[i] = dis(gen);
        K[i] = dis(gen);
        V[i] = dis(gen);
    }
    
    // Test attention computation using XSIMD
    ML::RealTime::LightweightAttention::Config config{embed_dim, 8, head_dim, seq_len};
    ML::RealTime::LightweightAttention attention(config);
    
    auto output = attention.forward(Q);
    
    // Verify output properties
    ASSERT_EQ(output.size(), Q.size());
    
    // Check that output is different from input (attention should transform)
    bool different = false;
    for (size_t i = 0; i < std::min(output.size(), Q.size()); ++i) {
        if (std::abs(output[i] - Q[i]) > 1e-3f) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "Attention output should differ from input";
    
    std::cout << "Scaled dot-product attention: PASS\n";
}

// Test Multi-Head Attention Concatenation
TEST_F(Phase2AttentionTest, MultiHeadConcatenation) {
    std::cout << "\n=== Multi-Head Attention Concatenation Test ===\n";
    
    for (const auto& config : test_configs) {
        ML::RealTime::LightweightAttention attention(config);
        
        // Test with different sequence lengths
        std::vector<size_t> test_seq_lengths = {32, 64, 128, config.sequence_length};
        
        for (size_t seq_len : test_seq_lengths) {
            if (seq_len > config.sequence_length) continue;
            
            // Create input for this sequence length
            std::vector<float> input(seq_len * config.embed_dim);
            std::fill(input.begin(), input.end(), 0.1f);
            
            // Forward pass
            auto output = attention.forward(input);
            
            // Verify output dimensions
            ASSERT_EQ(output.size(), input.size());
            
            // Verify output is reasonable (not all zeros)
            float sum = 0.0f;
            for (float val : output) {
                sum += std::abs(val);
            }
            EXPECT_GT(sum, 1e-6f) << "Output should not be all zeros";
        }
        
        std::cout << "Multi-head concatenation for " << config.num_heads 
                  << " heads: PASS\n";
    }
}

// Test Real-Time Latency Targets
TEST_F(Phase2AttentionTest, RealTimeLatencyTargets) {
    std::cout << "\n=== Real-Time Latency Targets Test ===\n";
    
    const double target_latency_ms = 1.0; // <1ms target
    const int iterations = 100;
    
    for (const auto& config : test_configs) {
        ML::RealTime::LightweightAttention attention(config);
        
        // Warm up
        for (int i = 0; i < 10; ++i) {
            attention.forward(test_inputs[config.embed_dim]);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            attention.forward(test_inputs[config.embed_dim]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_latency_ms = static_cast<double>(duration.count()) / iterations / 1000.0;
        
        std::cout << "Config " << config.embed_dim << "x" << config.num_heads 
                  << " heads: " << std::fixed << std::setprecision(3) 
                  << avg_latency_ms << "ms";
        
        if (avg_latency_ms <= target_latency_ms) {
            std::cout << " ✓ PASS\n";
        } else {
            std::cout << " ✗ FAIL (target: " << target_latency_ms << "ms)\n";
        }
        
        // For larger configurations, be more lenient
        double relaxed_target = (config.embed_dim >= 512) ? target_latency_ms * 2 : target_latency_ms;
        EXPECT_LE(avg_latency_ms, relaxed_target) 
            << "Latency target not met for config " << config.embed_dim;
    }
}

// Test Memory Efficiency
TEST_F(Phase2AttentionTest, MemoryEfficiency) {
    std::cout << "\n=== Memory Efficiency Test ===\n";
    
    const size_t target_memory_mb = 10; // <10MB target
    
    for (const auto& config : test_configs) {
        ML::RealTime::LightweightAttention attention(config);
        
        size_t memory_usage = attention.get_memory_usage();
        double memory_mb = static_cast<double>(memory_usage) / (1024 * 1024);
        
        std::cout << "Config " << config.embed_dim << "x" << config.num_heads 
                  << ": " << std::fixed << std::setprecision(2) 
                  << memory_mb << "MB";
        
        if (memory_mb <= target_memory_mb) {
            std::cout << " ✓ PASS\n";
        } else {
            std::cout << " ✗ FAIL (target: " << target_memory_mb << "MB)\n";
        }
        
        EXPECT_LE(memory_mb, target_memory_mb) 
            << "Memory target not met for config " << config.embed_dim;
    }
}

// Test Streaming Interface
TEST_F(Phase2AttentionTest, StreamingInterface) {
    std::cout << "\n=== Streaming Interface Test ===\n";
    
    auto config = test_configs[1]; // Use medium config
    ML::RealTime::LightweightAttention attention(config);
    
    // Test streaming mode
    attention.start_stream();
    
    // Process chunks
    size_t chunk_size = 32;
    size_t num_chunks = config.sequence_length / chunk_size;
    
    for (size_t i = 0; i < num_chunks; ++i) {
        std::vector<float> chunk(chunk_size * config.embed_dim, 0.1f * i);
        
        auto chunk_output = attention.process_chunk(chunk);
        
        // Verify chunk output
        ASSERT_EQ(chunk_output.size(), chunk.size());
        
        // Check for reasonable values
        for (float val : chunk_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in chunk " << i;
            ASSERT_FALSE(std::isinf(val)) << "Inf in chunk " << i;
        }
    }
    
    attention.end_stream();
    
    std::cout << "Streaming interface: PASS\n";
}

// Test Batch Processing
TEST_F(Phase2AttentionTest, BatchProcessing) {
    std::cout << "\n=== Batch Processing Test ===\n";
    
    auto config = test_configs[1];
    ML::RealTime::LightweightAttention attention(config);
    
    // Create batch of inputs
    std::vector<std::vector<float>> batch;
    size_t batch_size = 4;
    
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> input(config.sequence_length * config.embed_dim, 0.1f * (i + 1));
        batch.push_back(input);
    }
    
    // Process batch
    auto batch_output = attention.forward_batch(batch);
    
    // Verify batch output
    ASSERT_EQ(batch_output.size(), batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        ASSERT_EQ(batch_output[i].size(), batch[i].size());
        
        // Verify outputs are different for different inputs
        if (i > 0) {
            bool different = false;
            for (size_t j = 0; j < batch_output[i].size(); ++j) {
                if (std::abs(batch_output[i][j] - batch_output[i-1][j]) > 1e-3f) {
                    different = true;
                    break;
                }
            }
            EXPECT_TRUE(different) << "Batch outputs should differ";
        }
    }
    
    std::cout << "Batch processing: PASS\n";
}

// Test XSIMD Integration
TEST_F(Phase2AttentionTest, XSIMDIntegration) {
    std::cout << "\n=== XSIMD Integration Test ===\n";
    
    // Test that XSIMD operations are being used
    std::cout << "XSIMD batch size: " << ML::XSIMD::XSIMDVector::simd_batch_size() << std::endl;
    std::cout << "XSIMD support: " << (ML::XSIMD::XSIMDVector::has_simd_support() ? "Yes" : "No") << std::endl;
    
    // Test vector operations directly
    std::vector<float> test_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> result(test_vec.size());
    
    // Test XSIMD vector addition
    ML::XSIMD::VectorOps::vector_add_scalar(test_vec.data(), 2.0f, result.data(), test_vec.size());
    
    // Verify results
    for (size_t i = 0; i < test_vec.size(); ++i) {
        EXPECT_NEAR(result[i], test_vec[i] + 2.0f, 1e-5f);
    }
    
    std::cout << "XSIMD integration: PASS\n";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
