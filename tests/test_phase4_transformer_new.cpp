//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 4: Real-Time Transformer Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <thread>
#include <future>

#include "RealTimeTransformer.h"
#include "XSIMDOperations.h"

class Phase4TransformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test data
        vocab_size = 1000;
        embed_dim = 256;
        test_embedding.resize(embed_dim);
        
        for (size_t i = 0; i < embed_dim; ++i) {
            test_embedding[i] = dis(gen);
        }
        
        // Create test sequence
        test_sequence.resize(10);
        for (size_t i = 0; i < 10; ++i) {
            test_sequence[i].resize(embed_dim);
            for (size_t j = 0; j < embed_dim; ++j) {
                test_sequence[i][j] = dis(gen);
            }
        }
    }
    
    size_t vocab_size;
    size_t embed_dim;
    std::vector<float> test_embedding;
    std::vector<std::vector<float>> test_sequence;
};

// Test basic transformer functionality
TEST_F(Phase4TransformerTest, BasicTransformer) {
    std::cout << "\n=== Basic Transformer Test ===\n";
    
    ML::RealTime::StreamingTransformer::Config config{
        vocab_size, embed_dim, 8, 4, 1024, 0.1f, 512, 1.0f, 5
    };
    
    ML::RealTime::StreamingTransformer transformer(config);
    
    // Test single token processing
    auto output = transformer.forward_single(test_embedding);
    
    // Verify output dimensions
    ASSERT_EQ(output.size(), embed_dim);
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in transformer output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in transformer output";
    }
    
    std::cout << "Basic transformer: PASS\n";
}

// Test sequence processing
TEST_F(Phase4TransformerTest, SequenceProcessing) {
    std::cout << "\n=== Sequence Processing Test ===\n";
    
    ML::RealTime::StreamingTransformer::Config config{
        vocab_size, embed_dim, 4, 2, 512, 0.1f, 256, 2.0f, 3
    };
    
    ML::RealTime::StreamingTransformer transformer(config);
    
    // Test sequence processing
    auto output = transformer.forward(test_sequence);
    
    // Verify output dimensions (should be sequence_length * embed_dim)
    ASSERT_EQ(output.size(), test_sequence.size() * embed_dim);
    
    // Verify output is reasonable
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val));
        ASSERT_FALSE(std::isinf(val));
    }
    
    std::cout << "Sequence processing: PASS\n";
}

// Test streaming interface
TEST_F(Phase4TransformerTest, StreamingInterface) {
    std::cout << "\n=== Streaming Interface Test ===\n";
    
    ML::RealTime::StreamingTransformer::Config config{
        vocab_size, embed_dim, 4, 2, 512, 0.1f, 128, 1.5f, 2
    };
    
    ML::RealTime::StreamingTransformer transformer(config);
    
    // Start streaming
    transformer.start_stream();
    
    // Process tokens asynchronously
    std::vector<std::future<void>> futures;
    std::vector<std::vector<float>> outputs;
    
    for (size_t i = 0; i < 5; ++i) {
        auto future = std::async(std::launch::async, [&transformer, &test_embedding, i]() {
            transformer.process_token(test_embedding);
        });
        futures.push_back(std::move(future));
    }
    
    // Wait for all tokens to be processed
    for (auto& future : futures) {
        future.wait();
    }
    
    // Collect outputs
    for (size_t i = 0; i < 5; ++i) {
        auto output = transformer.get_next_output();
        if (!output.empty()) {
            outputs.push_back(output);
        }
    }
    
    // Stop streaming
    transformer.stop_stream();
    
    // Verify outputs
    ASSERT_GE(outputs.size(), 3); // At least some outputs should be available
    
    for (const auto& output : outputs) {
        ASSERT_EQ(output.size(), embed_dim);
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val));
            ASSERT_FALSE(std::isinf(val));
        }
    }
    
    std::cout << "Streaming interface: PASS\n";
}

// Test latency targets
TEST_F(Phase4TransformerTest, LatencyTargets) {
    std::cout << "\n=== Latency Targets Test ===\n";
    
    std::vector<ML::RealTime::StreamingTransformer::Config> configs = {
        {vocab_size, 64, 2, 2, 256, 0.1f, 128, 5.0f, 1},   // Mobile: 5ms target
        {vocab_size, 128, 4, 2, 512, 0.1f, 256, 2.0f, 2},  // Edge: 2ms target
        {vocab_size, 256, 8, 4, 1024, 0.1f, 512, 1.0f, 5}  // Server: 1ms target
    };
    
    std::vector<std::string> config_names = {"Mobile", "Edge", "Server"};
    
    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& config = configs[i];
        ML::RealTime::StreamingTransformer transformer(config);
        
        // Warm up
        for (int j = 0; j < 10; ++j) {
            transformer.forward_single(test_embedding);
        }
        
        // Benchmark
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int j = 0; j < iterations; ++j) {
            transformer.forward_single(test_embedding);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_latency_ms = static_cast<double>(duration.count()) / iterations / 1000.0;
        
        std::cout << config_names[i] << " (" << config.d_model << "D): " 
                  << std::fixed << std::setprecision(3) << avg_latency_ms << "ms";
        
        if (avg_latency_ms <= config.target_latency_ms) {
            std::cout << " ✓ PASS\n";
        } else {
            std::cout << " ✗ FAIL (target: " << config.target_latency_ms << "ms)\n";
        }
        
        // Check if transformer meets target
        EXPECT_LE(avg_latency_ms, config.target_latency_ms * 1.5) 
            << "Latency significantly exceeds target for " << config_names[i];
    }
}

// Test memory efficiency
TEST_F(Phase4TransformerTest, MemoryEfficiency) {
    std::cout << "\n=== Memory Efficiency Test ===\n";
    
    std::vector<ML::RealTime::StreamingTransformer::Config> configs = {
        {vocab_size, 64, 2, 2, 256, 0.1f, 128, 5.0f, 1},   // Mobile: 1MB target
        {vocab_size, 128, 4, 2, 512, 0.1f, 256, 2.0f, 2},  // Edge: 2MB target
        {vocab_size, 256, 8, 4, 1024, 0.1f, 512, 1.0f, 5}  // Server: 5MB target
    };
    
    std::vector<std::string> config_names = {"Mobile", "Edge", "Server"};
    
    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& config = configs[i];
        ML::RealTime::StreamingTransformer transformer(config);
        
        size_t memory_usage = transformer.get_memory_usage();
        double memory_mb = static_cast<double>(memory_usage) / (1024 * 1024);
        
        std::cout << config_names[i] << " (" << config.d_model << "D): " 
                  << std::fixed << std::setprecision(2) << memory_mb << "MB";
        
        if (memory_mb <= config.max_memory_mb) {
            std::cout << " ✓ PASS\n";
        } else {
            std::cout << " ✗ FAIL (target: " << config.max_memory_mb << "MB)\n";
        }
        
        // Check memory target
        EXPECT_LE(memory_mb, config.max_memory_mb * 1.5) 
            << "Memory usage significantly exceeds target for " << config_names[i];
    }
}

// Test throughput
TEST_F(Phase4TransformerTest, Throughput) {
    std::cout << "\n=== Throughput Test ===\n";
    
    ML::RealTime::StreamingTransformer::Config config{
        vocab_size, embed_dim, 8, 4, 1024, 0.1f, 512, 1.0f, 5
    };
    
    ML::RealTime::StreamingTransformer transformer(config);
    
    // Start streaming for throughput measurement
    transformer.start_stream();
    
    // Process tokens rapidly
    const int num_tokens = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_tokens; ++i) {
        transformer.process_token(test_embedding);
    }
    
    // Wait for processing to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float throughput = static_cast<float>(num_tokens) / duration.count() * 1000.0f;
    
    std::cout << "Processed " << num_tokens << " tokens in " << duration.count() << "ms\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << throughput << " tokens/sec\n";
    
    transformer.stop_stream();
    
    // Should achieve reasonable throughput
    EXPECT_GT(throughput, 100.0f) << "Throughput should be at least 100 tokens/sec";
}

// Test factory patterns
TEST_F(Phase4TransformerTest, FactoryPatterns) {
    std::cout << "\n=== Factory Patterns Test ===\n";
    
    // Test edge transformer
    auto edge_transformer = ML::RealTime::RealTimeTransformerFactory::create_for_edge(vocab_size, 2.0f);
    ASSERT_NE(edge_transformer, nullptr);
    
    // Test mobile transformer
    auto mobile_transformer = ML::RealTime::RealTimeTransformerFactory::create_for_mobile(vocab_size, 5.0f);
    ASSERT_NE(mobile_transformer, nullptr);
    
    // Test server transformer
    auto server_transformer = ML::RealTime::RealTimeTransformerFactory::create_for_server(vocab_size, 0.5f);
    ASSERT_NE(server_transformer, nullptr);
    
    // Test each transformer
    std::vector<std::pair<std::string, std::unique_ptr<ML::RealTime::StreamingTransformer>>> transformers = {
        {"Edge", std::move(edge_transformer)},
        {"Mobile", std::move(mobile_transformer)},
        {"Server", std::move(server_transformer)}
    };
    
    for (const auto& [name, transformer] : transformers) {
        auto output = transformer->forward_single(test_embedding);
        ASSERT_EQ(output.size(), transformer->get_config().d_model);
        
        std::cout << name << " transformer (" << transformer->get_config().d_model << "D): PASS\n";
    }
}

// Test optimization modes
TEST_F(Phase4TransformerTest, OptimizationModes) {
    std::cout << "\n=== Optimization Modes Test ===\n";
    
    ML::RealTime::StreamingTransformer::Config config{
        vocab_size, embed_dim, 8, 4, 1024, 0.1f, 512, 1.0f, 5
    };
    
    ML::RealTime::StreamingTransformer transformer(config);
    
    // Test latency optimization
    transformer.optimize_for_latency();
    auto latency_output = transformer.forward_single(test_embedding);
    ASSERT_EQ(latency_output.size(), embed_dim);
    
    // Test memory optimization
    transformer.optimize_for_memory();
    auto memory_output = transformer.forward_single(test_embedding);
    ASSERT_EQ(memory_output.size(), embed_dim);
    
    // Verify outputs are still valid
    for (float val : latency_output) {
        ASSERT_FALSE(std::isnan(val));
        ASSERT_FALSE(std::isinf(val));
    }
    
    for (float val : memory_output) {
        ASSERT_FALSE(std::isnan(val));
        ASSERT_FALSE(std::isinf(val));
    }
    
    std::cout << "Optimization modes: PASS\n";
}

// Test XSIMD integration
TEST_F(Phase4TransformerTest, XSIMDIntegration) {
    std::cout << "\n=== XSIMD Integration in Transformer Test ===\n";
    
    std::cout << "XSIMD batch size: " << ML::XSIMD::XSIMDVector::simd_batch_size() << std::endl;
    std::cout << "XSIMD support: " << (ML::XSIMD::XSIMDVector::has_simd_support() ? "Yes" : "No") << std::endl;
    
    ML::RealTime::StreamingTransformer::Config config{
        vocab_size, embed_dim, 8, 4, 1024, 0.1f, 512, 1.0f, 5
    };
    
    ML::RealTime::StreamingTransformer transformer(config);
    
    // Test multiple forward passes to ensure SIMD consistency
    for (int i = 0; i < 50; ++i) {
        auto output = transformer.forward_single(test_embedding);
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in iteration " << i;
            ASSERT_FALSE(std::isinf(val)) << "Inf in iteration " << i;
        }
    }
    
    std::cout << "XSIMD integration in transformer: PASS\n";
}

// Test performance benchmarks
TEST_F(Phase4TransformerTest, PerformanceBenchmarks) {
    std::cout << "\n=== Performance Benchmarks Test ===\n";
    
    std::vector<ML::RealTime::StreamingTransformer::Config> configs = {
        {vocab_size, 64, 2, 2, 256, 0.1f, 128, 5.0f, 1},   // Mobile
        {vocab_size, 128, 4, 2, 512, 0.1f, 256, 2.0f, 2},  // Edge
        {vocab_size, 256, 8, 4, 1024, 0.1f, 512, 1.0f, 5}  // Server
    };
    
    std::vector<std::string> config_names = {"Mobile", "Edge", "Server"};
    
    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& config = configs[i];
        ML::RealTime::StreamingTransformer transformer(config);
        
        // Benchmark forward pass
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int j = 0; j < iterations; ++j) {
            transformer.forward_single(test_embedding);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_latency_us = static_cast<double>(duration.count()) / iterations;
        double avg_latency_ms = avg_latency_us / 1000.0;
        
        // Memory usage
        size_t memory_usage = transformer.get_memory_usage();
        double memory_mb = static_cast<double>(memory_usage) / (1024 * 1024);
        
        std::cout << config_names[i] << " (" << config.d_model << "D):\n";
        std::cout << "  Latency: " << std::fixed << std::setprecision(3) << avg_latency_ms << "ms\n";
        std::cout << "  Memory: " << std::fixed << std::setprecision(2) << memory_mb << "MB\n";
        
        // Performance assertions
        EXPECT_LT(avg_latency_ms, 10.0) << "Latency should be under 10ms";
        EXPECT_LT(memory_mb, 20.0) << "Memory usage should be under 20MB";
    }
    
    std::cout << "Performance benchmarks: PASS\n";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
