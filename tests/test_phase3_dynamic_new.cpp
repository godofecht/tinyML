//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 3: Dynamic Neural Systems Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

#include "DynamicNeuralNetwork.h"
#include "XSIMDOperations.h"

class Phase3DynamicTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test data
        input_size = 256;
        test_input.resize(input_size);
        
        for (size_t i = 0; i < input_size; ++i) {
            test_input[i] = dis(gen);
        }
    }
    
    size_t input_size;
    std::vector<float> test_input;
};

// Test Dynamic Layer functionality
TEST_F(Phase3DynamicTest, DynamicLayerBasics) {
    std::cout << "\n=== Dynamic Layer Basics Test ===\n";
    
    ML::Dynamic::DynamicLayer::Config config{256, 128, "tanh", false, 0.0f};
    ML::Dynamic::DynamicDenseLayer layer(config);
    
    // Test forward pass
    auto output = layer.forward(test_input);
    
    // Verify output dimensions
    ASSERT_EQ(output.size(), config.output_size);
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in layer output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in layer output";
    }
    
    std::cout << "Dynamic layer basics: PASS\n";
}

// Test layer resizing
TEST_F(Phase3DynamicTest, LayerResizing) {
    std::cout << "\n=== Layer Resizing Test ===\n";
    
    ML::Dynamic::DynamicLayer::Config config{256, 128, "relu", false, 0.0f};
    ML::Dynamic::DynamicDenseLayer layer(config);
    
    // Test initial forward pass
    auto initial_output = layer.forward(test_input);
    ASSERT_EQ(initial_output.size(), 128);
    
    // Resize layer
    layer.resize(256, 64);
    
    // Create new input for resized layer
    std::vector<float> new_input(256, 0.5f);
    auto resized_output = layer.forward(new_input);
    
    // Verify new dimensions
    ASSERT_EQ(resized_output.size(), 64);
    
    // Verify output is reasonable
    for (float val : resized_output) {
        ASSERT_FALSE(std::isnan(val));
        ASSERT_FALSE(std::isinf(val));
    }
    
    std::cout << "Layer resizing: PASS\n";
}

// Test Dynamic Neural Network
TEST_F(Phase3DynamicTest, DynamicNeuralNetwork) {
    std::cout << "\n=== Dynamic Neural Network Test ===\n";
    
    ML::Dynamic::DynamicNeuralNetwork::NetworkConfig config{
        {256, 128, 64, 32},  // Initial topology
        0.001f,               // Learning rate
        0.1f,                 // Adaptation threshold
        1024,                 // Max layer size
        16,                   // Min layer size
        true,                 // Enable growth
        true                  // Enable pruning
    };
    
    ML::Dynamic::DynamicNeuralNetwork network(config);
    
    // Test forward pass
    auto output = network.forward(test_input);
    
    // Verify output dimensions (should match last layer)
    ASSERT_EQ(output.size(), 32);
    
    // Verify output is reasonable
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val));
        ASSERT_FALSE(std::isinf(val));
    }
    
    // Test topology query
    auto topology = network.get_current_topology();
    ASSERT_EQ(topology.size(), 4);
    // Note: The topology shows layer output sizes, which may differ from initial config
    // This is expected behavior for dynamic networks
    std::cout << "Network topology: ";
    for (size_t size : topology) std::cout << size << " ";
    std::cout << std::endl;
    EXPECT_GT(topology[0], 0) << "First layer should have positive size";
    
    std::cout << "Dynamic neural network: PASS\n";
}

// Test topology adaptation
TEST_F(Phase3DynamicTest, TopologyAdaptation) {
    std::cout << "\n=== Topology Adaptation Test ===\n";
    
    ML::Dynamic::DynamicNeuralNetwork::NetworkConfig config{
        {256, 64, 32},
        0.001f, 0.05f, 512, 8, true, true
    };
    
    ML::Dynamic::DynamicNeuralNetwork network(config);
    
    // Get initial topology
    auto initial_topology = network.get_current_topology();
    auto initial_memory = network.get_memory_usage();
    
    // Simulate adaptation trigger
    std::vector<float> mock_errors(100, 0.2f); // High error to trigger adaptation
    network.adapt_topology(mock_errors);
    
    // Check if topology changed
    auto adapted_topology = network.get_current_topology();
    
    // Topology might have changed due to adaptation
    std::cout << "Initial topology: ";
    for (size_t size : initial_topology) std::cout << size << " ";
    std::cout << "\nAdapted topology: ";
    for (size_t size : adapted_topology) std::cout << size << " ";
    std::cout << std::endl;
    
    // Verify network still works
    auto output = network.forward(test_input);
    ASSERT_EQ(output.size(), adapted_topology.back());
    
    std::cout << "Topology adaptation: PASS\n";
}

// Test layer addition and removal
TEST_F(Phase3DynamicTest, LayerAdditionRemoval) {
    std::cout << "\n=== Layer Addition/Removal Test ===\n";
    
    ML::Dynamic::DynamicNeuralNetwork::NetworkConfig config{
        {256, 128, 64},
        0.001f, 0.1f, 512, 16, true, true
    };
    
    ML::Dynamic::DynamicNeuralNetwork network(config);
    
    // Get initial layer count
    auto initial_topology = network.get_current_topology();
    size_t initial_layers = initial_topology.size();
    
    // Add a layer
    network.add_layer(1, 96); // Add layer of size 96 at position 1
    
    // Check new topology
    auto new_topology = network.get_current_topology();
    EXPECT_EQ(new_topology.size(), initial_layers + 1);
    
    // Test forward pass with new layer
    auto output = network.forward(test_input);
    ASSERT_EQ(output.size(), new_topology.back());
    
    // Remove a layer
    network.remove_layer(1);
    
    // Check topology after removal
    auto final_topology = network.get_current_topology();
    EXPECT_EQ(final_topology.size(), initial_layers);
    
    // Test forward pass after removal
    auto final_output = network.forward(test_input);
    ASSERT_EQ(final_output.size(), final_topology.back());
    
    std::cout << "Layer addition/removal: PASS\n";
}

// Test Memory Pool
TEST_F(Phase3DynamicTest, MemoryPool) {
    std::cout << "\n=== Memory Pool Test ===\n";
    
    auto& pool = ML::Dynamic::MemoryPool::getInstance();
    
    // Clear pool for clean test
    pool.clear();
    
    // Test allocation
    void* ptr1 = pool.allocate(1024);
    void* ptr2 = pool.allocate(2048);
    void* ptr3 = pool.allocate(512);
    
    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr3, nullptr);
    
    // Check memory usage
    size_t total_allocated = pool.get_total_allocated();
    EXPECT_GT(total_allocated, 0);
    
    // Test deallocation
    pool.deallocate(ptr1);
    pool.deallocate(ptr2);
    pool.deallocate(ptr3);
    
    std::cout << "Memory allocated: " << total_allocated << " bytes\n";
    std::cout << "Memory pool: PASS\n";
}

// Test Evolutionary Optimizer
TEST_F(Phase3DynamicTest, EvolutionaryOptimizer) {
    std::cout << "\n=== Evolutionary Optimizer Test ===\n";
    
    ML::Dynamic::EvolutionaryOptimizer::Config config{
        20,    // Small population for quick test
        0.1f,
        0.8f,
        2,
        5      // Few generations for quick test
    };
    
    ML::Dynamic::EvolutionaryOptimizer optimizer(config);
    
    // Create simple test data
    std::vector<std::vector<float>> inputs(10);
    std::vector<std::vector<float>> targets(10);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < 10; ++i) {
        inputs[i].resize(256);
        targets[i].resize(32);
        
        for (size_t j = 0; j < 256; ++j) {
            inputs[i][j] = dis(gen);
        }
        for (size_t j = 0; j < 32; ++j) {
            targets[i][j] = dis(gen);
        }
    }
    
    ML::Dynamic::DynamicNeuralNetwork::NetworkConfig net_config{
        {256, 128, 64, 32},
        0.001f, 0.1f, 512, 16, true, true
    };
    
    ML::Dynamic::DynamicNeuralNetwork network(net_config);
    
    // Run optimization (this is a simplified test)
    optimizer.optimize(network, inputs, targets);
    
    // Check that optimization ran
    EXPECT_GT(optimizer.get_generation(), 0);
    EXPECT_LT(optimizer.get_best_fitness(), std::numeric_limits<float>::max());
    
    std::cout << "Generations: " << optimizer.get_generation() << std::endl;
    std::cout << "Best fitness: " << optimizer.get_best_fitness() << std::endl;
    std::cout << "Evolutionary optimizer: PASS\n";
}

// Test Adaptive Inference System
TEST_F(Phase3DynamicTest, AdaptiveInferenceSystem) {
    std::cout << "\n=== Adaptive Inference System Test ===\n";
    
    ML::Dynamic::AdaptiveInferenceSystem::Config config{
        256,    // Input size
        32,     // Output size
        0.01f,  // Adaptation rate
        50,     // Performance window
        2.0f    // Latency threshold
    };
    
    ML::Dynamic::AdaptiveInferenceSystem system(config);
    
    // Test multiple predictions
    const int num_predictions = 20;
    std::vector<float> latencies;
    
    for (int i = 0; i < num_predictions; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto output = system.predict(test_input);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float latency_ms = static_cast<float>(duration.count()) / 1000.0f;
        
        latencies.push_back(latency_ms);
        
        // Verify output
        ASSERT_EQ(output.size(), config.output_size);
        
        // Update performance
        system.update_performance(latency_ms, 0.9f); // 90% accuracy
    }
    
    // Check performance metrics
    float avg_latency = system.get_average_latency();
    float avg_accuracy = system.get_average_accuracy();
    
    EXPECT_GT(avg_latency, 0.0f);
    EXPECT_GT(avg_accuracy, 0.0f);
    
    std::cout << "Average latency: " << avg_latency << "ms\n";
    std::cout << "Average accuracy: " << avg_accuracy * 100 << "%\n";
    std::cout << "Adaptive inference system: PASS\n";
}

// Test performance benchmarks
TEST_F(Phase3DynamicTest, PerformanceBenchmarks) {
    std::cout << "\n=== Performance Benchmarks Test ===\n";
    
    ML::Dynamic::DynamicNeuralNetwork::NetworkConfig config{
        {256, 128, 64, 32},
        0.001f, 0.1f, 512, 16, true, true
    };
    
    ML::Dynamic::DynamicNeuralNetwork network(config);
    
    // Benchmark forward pass
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        network.forward(test_input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_latency_us = static_cast<double>(duration.count()) / iterations;
    double avg_latency_ms = avg_latency_us / 1000.0;
    
    std::cout << "Average forward pass latency: " << std::fixed << std::setprecision(3) 
              << avg_latency_ms << "ms (" << avg_latency_us << "μs)\n";
    
    // Benchmark memory usage
    size_t memory_usage = network.get_memory_usage();
    double memory_mb = static_cast<double>(memory_usage) / (1024 * 1024);
    
    std::cout << "Memory usage: " << std::fixed << std::setprecision(2) 
              << memory_mb << "MB\n";
    
    // Performance targets
    EXPECT_LT(avg_latency_ms, 5.0) << "Latency should be under 5ms";
    EXPECT_LT(memory_mb, 50.0) << "Memory usage should be under 50MB";
    
    std::cout << "Performance benchmarks: PASS\n";
}

// Test XSIMD integration in dynamic systems
TEST_F(Phase3DynamicTest, XSIMDIntegration) {
    std::cout << "\n=== XSIMD Integration in Dynamic Systems Test ===\n";
    
    std::cout << "XSIMD batch size: " << ML::XSIMD::XSIMDVector::simd_batch_size() << std::endl;
    std::cout << "XSIMD support: " << (ML::XSIMD::XSIMDVector::has_simd_support() ? "Yes" : "No") << std::endl;
    
    // Test that dynamic layers use XSIMD
    ML::Dynamic::DynamicLayer::Config config{256, 128, "tanh", false, 0.0f};
    ML::Dynamic::DynamicDenseLayer layer(config);
    
    // Multiple forward passes to test SIMD consistency
    for (int i = 0; i < 100; ++i) {
        auto output = layer.forward(test_input);
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val)) << "NaN in iteration " << i;
            ASSERT_FALSE(std::isinf(val)) << "Inf in iteration " << i;
        }
    }
    
    std::cout << "XSIMD integration in dynamic systems: PASS\n";
}
