//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 2: Simple Attention Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

#include "LightweightAttention.h"
#include "XSIMDOperations.h"

class Phase2SimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test with single embedding vectors (current implementation)
        embed_dim = 256;
        test_input.resize(embed_dim);
        
        for (size_t i = 0; i < embed_dim; ++i) {
            test_input[i] = dis(gen);
        }
    }
    
    size_t embed_dim;
    std::vector<float> test_input;
};

// Test basic attention functionality
TEST_F(Phase2SimpleTest, BasicAttention) {
    std::cout << "\n=== Basic Attention Test ===\n";
    
    ML::RealTime::LightweightAttention::Config config{embed_dim, 8, 32, 512};
    ML::RealTime::LightweightAttention attention(config);
    
    // Test forward pass
    auto output = attention.forward(test_input);
    
    // Verify output dimensions
    ASSERT_EQ(output.size(), test_input.size());
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in attention output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in attention output";
    }
    
    std::cout << "Basic attention: PASS\n";
}

// Test XSIMD integration in attention
TEST_F(Phase2SimpleTest, AttentionXSIMDIntegration) {
    std::cout << "\n=== Attention XSIMD Integration Test ===\n";
    
    std::cout << "XSIMD batch size: " << ML::XSIMD::XSIMDVector::simd_batch_size() << std::endl;
    std::cout << "XSIMD support: " << (ML::XSIMD::XSIMDVector::has_simd_support() ? "Yes" : "No") << std::endl;
    
    ML::RealTime::LightweightAttention::Config config{embed_dim, 8, 32, 512};
    ML::RealTime::LightweightAttention attention(config);
    
    // Test multiple forward passes
    const int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        auto output = attention.forward(test_input);
        
        // Verify output is reasonable
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in iteration " << i;
            ASSERT_FALSE(std::isinf(val)) << "Inf in iteration " << i;
        }
    }
    
    std::cout << "Attention XSIMD integration: PASS\n";
}

// Test different configurations
TEST_F(Phase2SimpleTest, DifferentConfigurations) {
    std::cout << "\n=== Different Configurations Test ===\n";
    
    std::vector<ML::RealTime::LightweightAttention::Config> configs = {
        {64, 4, 16, 128},   // Small
        {128, 4, 32, 256},  // Medium
        {256, 8, 32, 512},  // Large
        {512, 8, 64, 512}   // Very Large
    };
    
    for (const auto& config : configs) {
        // Create appropriate sized input
        std::vector<float> input(config.embed_dim);
        std::fill(input.begin(), input.end(), 0.1f);
        
        ML::RealTime::LightweightAttention attention(config);
        
        // Test forward pass
        auto output = attention.forward(input);
        
        // Verify output
        ASSERT_EQ(output.size(), input.size());
        
        // Check for reasonable values
        float sum = 0.0f;
        for (float val : output) {
            sum += std::abs(val);
            ASSERT_FALSE(std::isnan(val));
            ASSERT_FALSE(std::isinf(val));
        }
        
        EXPECT_GT(sum, 1e-6f) << "Output should not be all zeros";
        
        std::cout << "Config " << config.embed_dim << "x" << config.num_heads << ": PASS\n";
    }
}

// Test memory usage
TEST_F(Phase2SimpleTest, MemoryUsage) {
    std::cout << "\n=== Memory Usage Test ===\n";
    
    std::vector<ML::RealTime::LightweightAttention::Config> configs = {
        {64, 4, 16, 128},
        {128, 4, 32, 256},
        {256, 8, 32, 512}
    };
    
    for (const auto& config : configs) {
        ML::RealTime::LightweightAttention attention(config);
        
        size_t memory_usage = attention.get_memory_usage();
        double memory_mb = static_cast<double>(memory_usage) / (1024 * 1024);
        
        std::cout << "Config " << config.embed_dim << "x" << config.num_heads 
                  << ": " << std::fixed << std::setprecision(2) 
                  << memory_mb << "MB\n";
        
        // Memory should be reasonable
        EXPECT_LT(memory_mb, 50.0) << "Memory usage seems too high";
    }
}

// Test performance
TEST_F(Phase2SimpleTest, Performance) {
    std::cout << "\n=== Performance Test ===\n";
    
    ML::RealTime::LightweightAttention::Config config{embed_dim, 8, 32, 512};
    ML::RealTime::LightweightAttention attention(config);
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        attention.forward(test_input);
    }
    
    // Benchmark
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        attention.forward(test_input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_latency_us = static_cast<double>(duration.count()) / iterations;
    double avg_latency_ms = avg_latency_us / 1000.0;
    
    std::cout << "Average latency: " << std::fixed << std::setprecision(3) 
              << avg_latency_ms << "ms (" << avg_latency_us << "μs)\n";
    
    // Performance should be reasonable
    EXPECT_LT(avg_latency_ms, 10.0) << "Latency seems too high";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
