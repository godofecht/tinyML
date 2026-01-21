//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 4: Simple Real-Time Transformer Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

#include "LightweightAttention.h"
#include "DynamicNeuralNetwork.h"
#include "XSIMDOperations.h"

class Phase4SimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test data
        embed_dim = 256;
        sequence_length = 64;
        
        // Create test sequence
        test_sequence.resize(sequence_length);
        for (size_t i = 0; i < sequence_length; ++i) {
            test_sequence[i].resize(embed_dim);
            for (size_t j = 0; j < embed_dim; ++j) {
                test_sequence[i][j] = dis(gen);
            }
        }
        
        // Single token embedding
        test_token.resize(embed_dim);
        for (size_t i = 0; i < embed_dim; ++i) {
            test_token[i] = dis(gen);
        }
    }
    
    size_t embed_dim;
    size_t sequence_length;
    std::vector<std::vector<float>> test_sequence;
    std::vector<float> test_token;
};

// Test transformer block composition
TEST_F(Phase4SimpleTest, TransformerBlockComposition) {
    std::cout << "\n=== Transformer Block Composition Test ===\n";
    
    // Create attention layer
    ML::RealTime::LightweightAttention::Config attn_config{embed_dim, 8, 32, 512};
    ML::RealTime::LightweightAttention attention(attn_config);
    
    // Create feed-forward layers
    ML::Dynamic::DynamicLayer::Config ff1_config{embed_dim, embed_dim * 4, "relu", false, 0.0f};
    ML::Dynamic::DynamicLayer::Config ff2_config{embed_dim * 4, embed_dim, "tanh", false, 0.0f};
    
    ML::Dynamic::DynamicDenseLayer ff1(ff1_config);
    ML::Dynamic::DynamicDenseLayer ff2(ff2_config);
    
    // Process through transformer block
    auto attn_output = attention.forward(test_token);
    
    // Add residual connection
    std::vector<float> hidden1 = test_token;
    for (size_t i = 0; i < embed_dim; ++i) {
        hidden1[i] += attn_output[i];
    }
    
    // Feed-forward
    auto ff_hidden = ff1.forward(hidden1);
    auto ff_output = ff2.forward(ff_hidden);
    
    // Add residual connection
    std::vector<float> output = hidden1;
    for (size_t i = 0; i < embed_dim; ++i) {
        output[i] += ff_output[i];
    }
    
    // Verify output
    ASSERT_EQ(output.size(), embed_dim);
    
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val));
        ASSERT_FALSE(std::isinf(val));
    }
    
    std::cout << "Transformer block composition: PASS\n";
}

// Test multi-layer transformer
TEST_F(Phase4SimpleTest, MultiLayerTransformer) {
    std::cout << "\n=== Multi-Layer Transformer Test ===\n";
    
    const size_t num_layers = 4;
    std::vector<std::unique_ptr<ML::RealTime::LightweightAttention>> attention_layers;
    std::vector<std::unique_ptr<ML::Dynamic::DynamicDenseLayer>> ff1_layers, ff2_layers;
    
    // Create layers
    for (size_t i = 0; i < num_layers; ++i) {
        ML::RealTime::LightweightAttention::Config attn_config{embed_dim, 8, 32, 512};
        attention_layers.push_back(std::make_unique<ML::RealTime::LightweightAttention>(attn_config));
        
        ML::Dynamic::DynamicLayer::Config ff1_config{embed_dim, embed_dim * 4, "relu", false, 0.0f};
        ML::Dynamic::DynamicLayer::Config ff2_config{embed_dim * 4, embed_dim, "tanh", false, 0.0f};
        
        ff1_layers.push_back(std::make_unique<ML::Dynamic::DynamicDenseLayer>(ff1_config));
        ff2_layers.push_back(std::make_unique<ML::Dynamic::DynamicDenseLayer>(ff2_config));
    }
    
    // Process through all layers
    std::vector<float> hidden = test_token;
    
    for (size_t layer = 0; layer < num_layers; ++layer) {
        // Attention
        auto attn_output = attention_layers[layer]->forward(hidden);
        
        // Residual connection
        for (size_t i = 0; i < embed_dim; ++i) {
            hidden[i] += attn_output[i];
        }
        
        // Feed-forward
        auto ff_hidden = ff1_layers[layer]->forward(hidden);
        auto ff_output = ff2_layers[layer]->forward(ff_hidden);
        
        // Residual connection
        for (size_t i = 0; i < embed_dim; ++i) {
            hidden[i] += ff_output[i];
        }
    }
    
    // Verify final output
    ASSERT_EQ(hidden.size(), embed_dim);
    
    for (float val : hidden) {
        ASSERT_FALSE(std::isnan(val));
        ASSERT_FALSE(std::isinf(val));
    }
    
    std::cout << "Multi-layer transformer: PASS\n";
}

// Test sequence processing
TEST_F(Phase4SimpleTest, SequenceProcessing) {
    std::cout << "\n=== Sequence Processing Test ===\n";
    
    ML::RealTime::LightweightAttention::Config attn_config{embed_dim, 8, 32, 512};
    ML::RealTime::LightweightAttention attention(attn_config);
    
    // Process sequence token by token
    std::vector<std::vector<float>> outputs;
    
    for (const auto& token : test_sequence) {
        auto output = attention.forward(token);
        outputs.push_back(output);
    }
    
    // Verify all outputs
    ASSERT_EQ(outputs.size(), sequence_length);
    
    for (const auto& output : outputs) {
        ASSERT_EQ(output.size(), embed_dim);
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val));
            ASSERT_FALSE(std::isinf(val));
        }
    }
    
    std::cout << "Sequence processing: PASS\n";
}

// Test performance benchmarks
TEST_F(Phase4SimpleTest, PerformanceBenchmarks) {
    std::cout << "\n=== Performance Benchmarks Test ===\n";
#ifndef NDEBUG
    GTEST_SKIP() << "Performance targets require a Release build (-O3).";
#endif
    
    // Create a 4-layer transformer
    std::vector<std::unique_ptr<ML::RealTime::LightweightAttention>> attention_layers;
    std::vector<std::unique_ptr<ML::Dynamic::DynamicDenseLayer>> ff1_layers, ff2_layers;
    
    const size_t num_layers = 4;
    for (size_t i = 0; i < num_layers; ++i) {
        ML::RealTime::LightweightAttention::Config attn_config{embed_dim, 8, 32, 512};
        attention_layers.push_back(std::make_unique<ML::RealTime::LightweightAttention>(attn_config));
        
        ML::Dynamic::DynamicLayer::Config ff1_config{embed_dim, embed_dim * 4, "relu", false, 0.0f};
        ML::Dynamic::DynamicLayer::Config ff2_config{embed_dim * 4, embed_dim, "tanh", false, 0.0f};
        
        ff1_layers.push_back(std::make_unique<ML::Dynamic::DynamicDenseLayer>(ff1_config));
        ff2_layers.push_back(std::make_unique<ML::Dynamic::DynamicDenseLayer>(ff2_config));
    }
    
    // Benchmark single token processing
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        std::vector<float> hidden = test_token;
        
        for (size_t layer = 0; layer < num_layers; ++layer) {
            auto attn_output = attention_layers[layer]->forward(hidden);
            
            // Residual connection
            for (size_t j = 0; j < embed_dim; ++j) {
                hidden[j] += attn_output[j];
            }
            
            auto ff_hidden = ff1_layers[layer]->forward(hidden);
            auto ff_output = ff2_layers[layer]->forward(ff_hidden);
            
            // Residual connection
            for (size_t j = 0; j < embed_dim; ++j) {
                hidden[j] += ff_output[j];
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_latency_us = static_cast<double>(duration.count()) / iterations;
    double avg_latency_ms = avg_latency_us / 1000.0;
    
    std::cout << "4-layer transformer (" << embed_dim << "D):\n";
    std::cout << "  Latency: " << std::fixed << std::setprecision(3) << avg_latency_ms << "ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << (1000.0 / avg_latency_ms) << " tokens/sec\n";
    
    // Performance targets
    EXPECT_LT(avg_latency_ms, 10.0) << "Latency should be under 10ms";
    EXPECT_GT(1000.0 / avg_latency_ms, 100.0) << "Throughput should be at least 100 tokens/sec";
    
    std::cout << "Performance benchmarks: PASS\n";
}

// Test memory efficiency
TEST_F(Phase4SimpleTest, MemoryEfficiency) {
    std::cout << "\n=== Memory Efficiency Test ===\n";
    
    std::vector<size_t> model_sizes = {64, 128, 256, 512};
    
    for (size_t d_model : model_sizes) {
        // Create transformer components
        ML::RealTime::LightweightAttention::Config attn_config{d_model, 8, d_model/8, 512};
        ML::RealTime::LightweightAttention attention(attn_config);
        
        ML::Dynamic::DynamicLayer::Config ff1_config{d_model, d_model * 4, "relu", false, 0.0f};
        ML::Dynamic::DynamicLayer::Config ff2_config{d_model * 4, d_model, "tanh", false, 0.0f};
        
        ML::Dynamic::DynamicDenseLayer ff1(ff1_config);
        ML::Dynamic::DynamicDenseLayer ff2(ff2_config);
        
        // Calculate memory usage
        size_t total_memory = attention.get_memory_usage() + ff1.get_memory_usage() + ff2.get_memory_usage();
        double memory_mb = static_cast<double>(total_memory) / (1024 * 1024);
        
        std::cout << "Model size " << d_model << "D: " << std::fixed << std::setprecision(2) 
                  << memory_mb << "MB\n";
        
        // Memory should be reasonable
        EXPECT_LT(memory_mb, 50.0) << "Memory usage seems too high for " << d_model << "D model";
    }
    
    std::cout << "Memory efficiency: PASS\n";
}

// Test streaming simulation
TEST_F(Phase4SimpleTest, StreamingSimulation) {
    std::cout << "\n=== Streaming Simulation Test ===\n";
#ifndef NDEBUG
    GTEST_SKIP() << "Streaming timing consistency requires a Release build (-O3).";
#endif
    
    ML::RealTime::LightweightAttention::Config attn_config{embed_dim, 8, 32, 512};
    ML::RealTime::LightweightAttention attention(attn_config);
    
    // Simulate streaming processing
    std::vector<float> processing_times;
    
    for (size_t i = 0; i < 100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process token
        auto output = attention.forward(test_token);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        processing_times.push_back(static_cast<float>(duration.count()));
        
        // Verify output
        ASSERT_EQ(output.size(), embed_dim);
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val));
            ASSERT_FALSE(std::isinf(val));
        }
    }
    
    // Calculate statistics
    float avg_time = 0.0f;
    for (float time : processing_times) {
        avg_time += time;
    }
    avg_time /= processing_times.size();
    
    float max_time = *std::max_element(processing_times.begin(), processing_times.end());
    float min_time = *std::min_element(processing_times.begin(), processing_times.end());
    
    std::cout << "Streaming statistics (100 tokens):\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(3) << avg_time << "μs\n";
    std::cout << "  Min: " << std::fixed << std::setprecision(3) << min_time << "μs\n";
    std::cout << "  Max: " << std::fixed << std::setprecision(3) << max_time << "μs\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << (1000000.0 / avg_time) << " tokens/sec\n";
    
    // Streaming should be consistent
    EXPECT_LT(max_time / min_time, 3.0) << "Processing times should be relatively consistent";
    
    std::cout << "Streaming simulation: PASS\n";
}

// Test different model configurations
TEST_F(Phase4SimpleTest, ModelConfigurations) {
    std::cout << "\n=== Model Configurations Test ===\n";
    
    std::vector<std::tuple<size_t, size_t, size_t>> configs = {
        {64, 2, 32},    // Small: 64D, 2 heads, 32 head_dim
        {128, 4, 32},   // Medium: 128D, 4 heads, 32 head_dim  
        {256, 8, 32},   // Large: 256D, 8 heads, 32 head_dim
        {512, 8, 64}    // Very Large: 512D, 8 heads, 64 head_dim
    };
    
    for (const auto& [d_model, n_heads, head_dim] : configs) {
        ML::RealTime::LightweightAttention::Config attn_config{d_model, n_heads, head_dim, 512};
        ML::RealTime::LightweightAttention attention(attn_config);
        
        // Create appropriate sized input
        std::vector<float> input(d_model, 0.1f);
        
        // Test forward pass
        auto start = std::chrono::high_resolution_clock::now();
        auto output = attention.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float latency_ms = static_cast<float>(duration.count()) / 1000.0f;
        
        // Verify output
        ASSERT_EQ(output.size(), d_model);
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val));
            ASSERT_FALSE(std::isinf(val));
        }
        
        std::cout << "Config " << d_model << "D, " << n_heads << " heads: " 
                  << std::fixed << std::setprecision(3) << latency_ms << "ms\n";
    }
    
    std::cout << "Model configurations: PASS\n";
}

// Test XSIMD integration in transformer
TEST_F(Phase4SimpleTest, XSIMDIntegration) {
    std::cout << "\n=== XSIMD Integration in Transformer Test ===\n";
    
    std::cout << "XSIMD batch size: " << ML::XSIMD::XSIMDVector::simd_batch_size() << std::endl;
    std::cout << "XSIMD support: " << (ML::XSIMD::XSIMDVector::has_simd_support() ? "Yes" : "No") << std::endl;
    
    // Create transformer components
    ML::RealTime::LightweightAttention::Config attn_config{embed_dim, 8, 32, 512};
    ML::RealTime::LightweightAttention attention(attn_config);
    
    ML::Dynamic::DynamicLayer::Config ff_config{embed_dim, embed_dim, "tanh", false, 0.0f};
    ML::Dynamic::DynamicDenseLayer ff_layer(ff_config);
    
    // Test multiple forward passes to ensure SIMD consistency
    for (int i = 0; i < 100; ++i) {
        auto attn_output = attention.forward(test_token);
        auto ff_output = ff_layer.forward(test_token);
        
        for (float val : attn_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in attention iteration " << i;
            ASSERT_FALSE(std::isinf(val)) << "Inf in attention iteration " << i;
        }
        
        for (float val : ff_output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in feed-forward iteration " << i;
            ASSERT_FALSE(std::isinf(val)) << "Inf in feed-forward iteration " << i;
        }
    }
    
    std::cout << "XSIMD integration in transformer: PASS\n";
}
