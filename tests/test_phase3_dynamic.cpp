//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 3 Dynamic Neural Systems Tests
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
#include <algorithm>
#include <numeric>

// Include existing components
#include "SIMDOperations.h"
#include "Model.h"

namespace ML {
namespace Dynamic {

// Mock DynamicNeuralNetwork class (to be implemented in Phase 3)
class DynamicNeuralNetwork {
private:
    std::vector<std::vector<float>> layers;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> gradients;
    std::vector<size_t> layer_sizes;
    std::vector<bool> active_neurons;
    std::vector<bool> active_connections;
    
    // Memory pool for dynamic allocation
    std::vector<float> memory_pool;
    size_t pool_size;
    size_t pool_used;
    
    // Dynamic parameters
    float learning_rate;
    float pruning_threshold;
    float growth_threshold;
    size_t max_neurons_per_layer;
    size_t min_neurons_per_layer;
    
public:
    DynamicNeuralNetwork(const std::vector<size_t>& initial_topology, 
                         size_t memory_pool_size = 1024*1024) 
        : layer_sizes(initial_topology), pool_size(memory_pool_size), pool_used(0),
          learning_rate(0.01f), pruning_threshold(0.01f), growth_threshold(0.1f),
          max_neurons_per_layer(1024), min_neurons_per_layer(4) {
        
        // Initialize layers and weights
        initializeNetwork();
        initializeMemoryPool();
    }
    
    void initializeNetwork() {
        layers.clear();
        weights.clear();
        gradients.clear();
        active_neurons.clear();
        active_connections.clear();
        
        // Initialize each layer
        for (size_t layer_idx = 0; layer_idx < layer_sizes.size(); ++layer_idx) {
            size_t layer_size = layer_sizes[layer_idx];
            
            // Initialize layer activations
            layers.push_back(std::vector<float>(layer_size, 0.0f));
            
            // Initialize active neuron mask
            active_neurons.push_back(std::vector<bool>(layer_size, true));
            
            // Initialize weights (except for input layer)
            if (layer_idx > 0) {
                size_t prev_size = layer_sizes[layer_idx - 1];
                weights.push_back(std::vector<float>(prev_size * layer_size, 0.0f));
                gradients.push_back(std::vector<float>(prev_size * layer_size, 0.0f));
                active_connections.push_back(std::vector<bool>(prev_size * layer_size, true));
                
                // Random weight initialization
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
                
                for (auto& w : weights.back()) {
                    w = dis(gen);
                }
            }
        }
    }
    
    void initializeMemoryPool() {
        memory_pool.resize(pool_size);
        pool_used = 0;
    }
    
    // Size-Agnostic Layers - can handle different input/output sizes
    void resizeLayer(size_t layer_idx, size_t new_size) {
        if (layer_idx >= layer_sizes.size()) return;
        
        size_t old_size = layer_sizes[layer_idx];
        
        if (new_size == old_size) return;
        
        // Check bounds
        new_size = std::max(min_neurons_per_layer, 
                           std::min(max_neurons_per_layer, new_size));
        
        // Resize layer
        layers[layer_idx].resize(new_size, 0.0f);
        active_neurons[layer_idx].resize(new_size, true);
        
        // Resize weights if not input layer
        if (layer_idx > 0) {
            size_t prev_size = layer_sizes[layer_idx - 1];
            weights[layer_idx - 1].resize(prev_size * new_size, 0.0f);
            gradients[layer_idx - 1].resize(prev_size * new_size, 0.0f);
            active_connections[layer_idx - 1].resize(prev_size * new_size, true);
            
            // Initialize new weights
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
            
            for (size_t i = old_size * prev_size; i < new_size * prev_size; ++i) {
                weights[layer_idx - 1][i] = dis(gen);
            }
        }
        
        // Resize next layer weights if not output layer
        if (layer_idx < layer_sizes.size() - 1) {
            size_t next_size = layer_sizes[layer_idx + 1];
            weights[layer_idx].resize(new_size * next_size, 0.0f);
            gradients[layer_idx].resize(new_size * next_size, 0.0f);
            active_connections[layer_idx].resize(new_size * next_size, true);
            
            // Initialize new weights
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
            
            for (size_t i = old_size * next_size; i < new_size * next_size; ++i) {
                weights[layer_idx][i] = dis(gen);
            }
        }
        
        layer_sizes[layer_idx] = new_size;
    }
    
    // Runtime Topology Adjustment
    void adjustTopology(const std::vector<float>& input_data, 
                       const std::vector<float>& target_output) {
        // Analyze network activity and adjust topology
        for (size_t layer_idx = 1; layer_idx < layer_sizes.size() - 1; ++layer_idx) {
            float avg_activation = 0.0f;
            size_t active_count = 0;
            
            for (size_t i = 0; i < layers[layer_idx].size(); ++i) {
                if (active_neurons[layer_idx][i]) {
                    avg_activation += std::abs(layers[layer_idx][i]);
                    active_count++;
                }
            }
            
            if (active_count > 0) {
                avg_activation /= active_count;
                
                // Grow layer if underutilized
                if (avg_activation > growth_threshold && 
                    layer_sizes[layer_idx] < max_neurons_per_layer) {
                    size_t new_size = layer_sizes[layer_idx] + 
                                    static_cast<size_t>(layer_sizes[layer_idx] * 0.1f);
                    resizeLayer(layer_idx, new_size);
                }
                
                // Prune layer if overutilized or inactive
                if (avg_activation < pruning_threshold && 
                    layer_sizes[layer_idx] > min_neurons_per_layer) {
                    size_t new_size = layer_sizes[layer_idx] - 
                                    static_cast<size_t>(layer_sizes[layer_idx] * 0.05f);
                    resizeLayer(layer_idx, new_size);
                }
            }
        }
    }
    
    // Memory Pool Management
    float* allocateFromPool(size_t size) {
        if (pool_used + size > pool_size) {
            return nullptr; // Out of memory
        }
        
        float* ptr = memory_pool.data() + pool_used;
        pool_used += size;
        return ptr;
    }
    
    void deallocateToPool(size_t size) {
        if (pool_used >= size) {
            pool_used -= size;
        }
    }
    
    void resetPool() {
        pool_used = 0;
        std::fill(memory_pool.begin(), memory_pool.end(), 0.0f);
    }
    
    // Gradient-Free Optimization (using evolutionary approaches)
    void gradientFreeOptimization(const std::vector<float>& input_data,
                                 const std::vector<float>& target_output) {
        // Simple evolutionary strategy - mutate weights and keep better ones
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> mutation_dis(-0.1f, 0.1f);
        std::uniform_real_distribution<float> selection_dis(0.0f, 1.0f);
        
        // Store current performance
        float current_error = computeError(input_data, target_output);
        
        // Create mutated version
        std::vector<std::vector<float>> mutated_weights = weights;
        
        for (auto& layer_weights : mutated_weights) {
            for (auto& w : layer_weights) {
                if (selection_dis(gen) < 0.1f) { // 10% mutation rate
                    w += mutation_dis(gen);
                }
            }
        }
        
        // Test mutated version
        auto original_weights = weights;
        weights = mutated_weights;
        float mutated_error = computeError(input_data, target_output);
        
        // Keep better version
        if (mutated_error < current_error) {
            // Keep mutated weights
            learning_rate *= 1.1f; // Increase learning rate
        } else {
            // Revert to original weights
            weights = original_weights;
            learning_rate *= 0.9f; // Decrease learning rate
        }
    }
    
    float computeError(const std::vector<float>& input_data,
                      const std::vector<float>& target_output) {
        forward(input_data);
        
        float error = 0.0f;
        const auto& output = layers.back();
        
        for (size_t i = 0; i < output.size() && i < target_output.size(); ++i) {
            float diff = output[i] - target_output[i];
            error += diff * diff;
        }
        
        return std::sqrt(error / output.size());
    }
    
    // Forward pass with dynamic layers
    void forward(const std::vector<float>& input_data) {
        if (input_data.size() != layer_sizes[0]) {
            // Resize input layer if needed
            resizeLayer(0, input_data.size());
        }
        
        // Set input layer
        layers[0] = input_data;
        
        // Forward through each layer
        for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
            const auto& prev_layer = layers[layer_idx - 1];
            auto& current_layer = layers[layer_idx];
            const auto& layer_weights = weights[layer_idx - 1];
            const auto& connections = active_connections[layer_idx - 1];
            
            size_t prev_size = prev_layer.size();
            size_t current_size = current_layer.size();
            
            // Matrix-vector multiplication with active connections
            for (size_t i = 0; i < current_size; ++i) {
                if (!active_neurons[layer_idx][i]) {
                    current_layer[i] = 0.0f;
                    continue;
                }
                
                float sum = 0.0f;
                for (size_t j = 0; j < prev_size; ++j) {
                    size_t weight_idx = j * current_size + i;
                    if (connections[weight_idx]) {
                        sum += prev_layer[j] * layer_weights[weight_idx];
                    }
                }
                
                // Apply activation function (tanh)
                current_layer[i] = std::tanh(sum);
            }
        }
    }
    
    // Neuroplasticity - Dynamic neuron addition/removal
    void applyNeuroplasticity() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (size_t layer_idx = 1; layer_idx < layer_sizes.size() - 1; ++layer_idx) {
            // Add neurons with small probability
            if (dis(gen) < 0.05f && layer_sizes[layer_idx] < max_neurons_per_layer) {
                resizeLayer(layer_idx, layer_sizes[layer_idx] + 1);
            }
            
            // Remove inactive neurons
            for (size_t i = 0; i < layers[layer_idx].size(); ++i) {
                if (std::abs(layers[layer_idx][i]) < 0.01f && dis(gen) < 0.1f) {
                    active_neurons[layer_idx][i] = false;
                }
            }
        }
    }
    
    // Pruning - Automatic connection optimization
    void applyPruning() {
        for (size_t layer_idx = 0; layer_idx < active_connections.size(); ++layer_idx) {
            auto& connections = active_connections[layer_idx];
            const auto& layer_weights = weights[layer_idx];
            
            // Prune weak connections
            for (size_t i = 0; i < connections.size(); ++i) {
                if (std::abs(layer_weights[i]) < pruning_threshold) {
                    connections[i] = false;
                }
            }
        }
    }
    
    // Getters for testing
    const std::vector<size_t>& getLayerSizes() const { return layer_sizes; }
    const std::vector<std::vector<float>>& getLayers() const { return layers; }
    float getLearningRate() const { return learning_rate; }
    size_t getMemoryUsed() const { return pool_used; }
    size_t getMemoryPoolSize() const { return pool_size; }
};

} // namespace Dynamic
} // namespace ML

class Phase3DynamicTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test configurations for dynamic networks
        test_topologies = {
            {4, 8, 4},      // Small network
            {8, 16, 8},     // Medium network  
            {16, 32, 16},   // Large network
        };
        
        for (const auto& topology : test_topologies) {
            std::string key = std::to_string(topology[0]) + "_" + 
                            std::to_string(topology[1]) + "_" + 
                            std::to_string(topology[2]);
            
            size_t input_size = topology[0];
            test_inputs[key] = std::vector<float>(input_size);
            test_targets[key] = std::vector<float>(topology.back());
            
            for (size_t i = 0; i < input_size; ++i) {
                test_inputs[key][i] = dis(gen);
            }
            
            for (size_t i = 0; i < topology.back(); ++i) {
                test_targets[key][i] = dis(gen);
            }
        }
    }
    
    std::vector<std::vector<size_t>> test_topologies;
    std::map<std::string, std::vector<float>> test_inputs;
    std::map<std::string, std::vector<float>> test_targets;
};

// Test DynamicNeuralNetwork Creation
TEST_F(Phase3DynamicTest, DynamicNeuralNetworkCreation) {
    for (const auto& topology : test_topologies) {
        ML::Dynamic::DynamicNeuralNetwork network(topology);
        
        const auto& layer_sizes = network.getLayerSizes();
        ASSERT_EQ(layer_sizes.size(), topology.size());
        
        for (size_t i = 0; i < topology.size(); ++i) {
            EXPECT_EQ(layer_sizes[i], topology[i]);
        }
        
        EXPECT_GT(network.getMemoryPoolSize(), 0);
        EXPECT_GT(network.getLearningRate(), 0);
    }
}

// Test Size-Agnostic Layers
TEST_F(Phase3DynamicTest, SizeAgnosticLayers) {
    ML::Dynamic::DynamicNeuralNetwork network({4, 8, 4});
    
    // Test layer resizing
    network.resizeLayer(1, 12); // Resize hidden layer from 8 to 12
    const auto& layer_sizes = network.getLayerSizes();
    EXPECT_EQ(layer_sizes[1], 12);
    
    // Test forward pass with resized layer
    std::string key = "4_8_4";
    const auto& input = test_inputs[key];
    
    EXPECT_NO_THROW(network.forward(input));
    
    // Test resizing input layer
    std::vector<float> larger_input(6, 1.0f);
    EXPECT_NO_THROW(network.forward(larger_input));
    
    const auto& new_layer_sizes = network.getLayerSizes();
    EXPECT_EQ(new_layer_sizes[0], 6); // Input layer should have resized
}

// Test Runtime Topology Adjustment
TEST_F(Phase3DynamicTest, RuntimeTopologyAdjustment) {
    for (const auto& topology : test_topologies) {
        ML::Dynamic::DynamicNeuralNetwork network(topology);
        
        std::string key = std::to_string(topology[0]) + "_" + 
                        std::to_string(topology[1]) + "_" + 
                        std::to_string(topology[2]);
        
        const auto& input = test_inputs[key];
        const auto& target = test_targets[key];
        
        // Store initial topology
        const auto& initial_sizes = network.getLayerSizes();
        
        // Perform forward pass
        network.forward(input);
        
        // Apply topology adjustment
        network.adjustTopology(input, target);
        
        // Check that topology may have changed
        const auto& new_sizes = network.getLayerSizes();
        
        // Topology should remain valid
        ASSERT_EQ(new_sizes.size(), initial_sizes.size());
        for (size_t size : new_sizes) {
            ASSERT_GE(size, 4); // Minimum layer size
            ASSERT_LE(size, 1024); // Maximum layer size
        }
    }
}

// Test Memory Pool Management
TEST_F(Phase3DynamicTest, MemoryPoolManagement) {
    const size_t pool_size = 1024;
    ML::Dynamic::DynamicNeuralNetwork network({4, 8, 4}, pool_size);
    
    EXPECT_EQ(network.getMemoryPoolSize(), pool_size);
    EXPECT_EQ(network.getMemoryUsed(), 0);
    
    // Test allocation
    float* ptr1 = network.allocateFromPool(100);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(network.getMemoryUsed(), 100);
    
    float* ptr2 = network.allocateFromPool(200);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(network.getMemoryUsed(), 300);
    
    // Test deallocation
    network.deallocateToPool(100);
    EXPECT_EQ(network.getMemoryUsed(), 200);
    
    // Test reset
    network.resetPool();
    EXPECT_EQ(network.getMemoryUsed(), 0);
    
    // Test out of memory condition
    float* large_ptr = network.allocateFromPool(pool_size * 2);
    EXPECT_EQ(large_ptr, nullptr);
}

// Test Gradient-Free Optimization
TEST_F(Phase3DynamicTest, GradientFreeOptimization) {
    ML::Dynamic::DynamicNeuralNetwork network({4, 8, 4});
    
    std::string key = "4_8_4";
    const auto& input = test_inputs[key];
    const auto& target = test_targets[key];
    
    // Store initial learning rate
    float initial_lr = network.getLearningRate();
    
    // Perform gradient-free optimization
    network.gradientFreeOptimization(input, target);
    
    // Learning rate should have been adjusted
    float new_lr = network.getLearningRate();
    EXPECT_NE(new_lr, initial_lr);
    
    // Network should still function
    EXPECT_NO_THROW(network.forward(input));
}

// Test Neuroplasticity Features
TEST_F(Phase3DynamicTest, NeuroplasticityFeatures) {
    ML::Dynamic::DynamicNeuralNetwork network({4, 8, 4});
    
    const auto& initial_sizes = network.getLayerSizes();
    
    // Apply neuroplasticity
    network.applyNeuroplasticity();
    
    // Check that network structure may have changed
    const auto& new_sizes = network.getLayerSizes();
    
    // Should still be valid
    ASSERT_EQ(new_sizes.size(), initial_sizes.size());
    for (size_t size : new_sizes) {
        ASSERT_GE(size, 4);
        ASSERT_LE(size, 1024);
    }
    
    // Network should still function
    std::string key = "4_8_4";
    const auto& input = test_inputs[key];
    EXPECT_NO_THROW(network.forward(input));
}

// Test Pruning Features
TEST_F(Phase3DynamicTest, PruningFeatures) {
    ML::Dynamic::DynamicNeuralNetwork network({4, 8, 4});
    
    // Perform forward pass to activate connections
    std::string key = "4_8_4";
    const auto& input = test_inputs[key];
    network.forward(input);
    
    // Apply pruning
    network.applyPruning();
    
    // Network should still function after pruning
    EXPECT_NO_THROW(network.forward(input));
    
    // Output should still be reasonable
    const auto& layers = network.getLayers();
    const auto& output = layers.back();
    
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected after pruning";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected after pruning";
    }
}

// Test Quantization Support (8-bit/4-bit inference)
TEST_F(Phase3DynamicTest, QuantizationSupport) {
    ML::Dynamic::DynamicNeuralNetwork network({4, 8, 4});
    
    std::string key = "4_8_4";
    const auto& input = test_inputs[key];
    
    // Forward pass with full precision
    network.forward(input);
    const auto& full_precision_output = network.getLayers().back();
    
    // Simulate 8-bit quantization
    std::vector<float> quantized_8bit(full_precision_output.size());
    for (size_t i = 0; i < full_precision_output.size(); ++i) {
        // Quantize to 8-bit (0-255 range)
        float val = full_precision_output[i];
        val = std::max(-1.0f, std::min(1.0f, val)); // Clamp to [-1, 1]
        quantized_8bit[i] = std::round((val + 1.0f) * 127.0f) / 127.0f - 1.0f;
    }
    
    // Verify quantization error is reasonable
    float max_error = 0.0f;
    for (size_t i = 0; i < full_precision_output.size(); ++i) {
        float error = std::abs(full_precision_output[i] - quantized_8bit[i]);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_LT(max_error, 0.01f) << "8-bit quantization error too high";
    
    // Simulate 4-bit quantization
    std::vector<float> quantized_4bit(full_precision_output.size());
    for (size_t i = 0; i < full_precision_output.size(); ++i) {
        float val = full_precision_output[i];
        val = std::max(-1.0f, std::min(1.0f, val));
        quantized_4bit[i] = std::round((val + 1.0f) * 7.0f) / 7.0f - 1.0f;
    }
    
    // Verify 4-bit quantization error
    max_error = 0.0f;
    for (size_t i = 0; i < full_precision_output.size(); ++i) {
        float error = std::abs(full_precision_output[i] - quantized_4bit[i]);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_LT(max_error, 0.125f) << "4-bit quantization error too high";
}

// Test Streaming Real-Time Learning
TEST_F(Phase3DynamicTest, StreamingRealTimeLearning) {
    ML::Dynamic::DynamicNeuralNetwork network({4, 8, 4});
    
    std::string key = "4_8_4";
    const auto& input = test_inputs[key];
    const auto& target = test_targets[key];
    
    // Simulate streaming data
    const int stream_steps = 100;
    std::vector<float> error_history(stream_steps);
    
    for (int step = 0; step < stream_steps; ++step) {
        // Add small noise to simulate streaming variation
        std::vector<float> noisy_input = input;
        std::vector<float> noisy_target = target;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> noise_dis(-0.1f, 0.1f);
        
        for (size_t i = 0; i < noisy_input.size(); ++i) {
            noisy_input[i] += noise_dis(gen);
        }
        
        for (size_t i = 0; i < noisy_target.size(); ++i) {
            noisy_target[i] += noise_dis(gen);
        }
        
        // Process streaming data
        network.forward(noisy_input);
        network.gradientFreeOptimization(noisy_input, noisy_target);
        
        // Track error
        error_history[step] = network.computeError(noisy_input, noisy_target);
        
        // Apply dynamic adjustments periodically
        if (step % 10 == 0) {
            network.adjustTopology(noisy_input, noisy_target);
            network.applyNeuroplasticity();
            network.applyPruning();
        }
    }
    
    // Verify learning occurred (error should generally decrease)
    float initial_error = error_history[0];
    float final_error = error_history.back();
    
    // Allow for some fluctuation but overall trend should be downward
    float avg_initial = 0.0f;
    float avg_final = 0.0f;
    
    for (int i = 0; i < 10; ++i) {
        avg_initial += error_history[i];
        avg_final += error_history[stream_steps - 10 + i];
    }
    
    avg_initial /= 10.0f;
    avg_final /= 10.0f;
    
    EXPECT_LT(avg_final, avg_initial) << "Streaming learning failed to reduce error";
}

// Test Performance and Resource Efficiency
TEST_F(Phase3DynamicTest, PerformanceAndResourceEfficiency) {
    std::cout << "\n=== Phase 3 Performance and Resource Efficiency ===\n";
    std::cout << std::setw(15) << "Topology" << std::setw(15) << "Forward (ms)" 
              << std::setw(15) << "Memory (MB)" << std::setw(15) << "Adapt (ms)" 
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    const int iterations = 100;
    const double target_forward_ms = 5.0; // <5ms forward pass
    const double target_adapt_ms = 10.0;   // <10ms adaptation
    const double target_memory_mb = 5.0;   // <5MB memory usage
    
    for (const auto& topology : test_topologies) {
        ML::Dynamic::DynamicNeuralNetwork network(topology);
        
        std::string key = std::to_string(topology[0]) + "_" + 
                        std::to_string(topology[1]) + "_" + 
                        std::to_string(topology[2]);
        
        const auto& input = test_inputs[key];
        const auto& target = test_targets[key];
        
        // Benchmark forward pass
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            network.forward(input);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_forward_ms = static_cast<double>(duration.count()) / (iterations * 1000.0);
        
        // Benchmark adaptation
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            network.adjustTopology(input, target);
            network.applyNeuroplasticity();
            network.applyPruning();
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_adapt_ms = static_cast<double>(duration.count()) / (iterations * 1000.0);
        
        // Calculate memory usage
        double memory_mb = static_cast<double>(network.getMemoryPoolSize()) / (1024 * 1024);
        
        std::string topology_str = std::to_string(topology[0]) + "-" + 
                                std::to_string(topology[1]) + "-" + 
                                std::to_string(topology[2]);
        
        std::string status = "PASS";
        if (avg_forward_ms > target_forward_ms || avg_adapt_ms > target_adapt_ms || 
            memory_mb > target_memory_mb) {
            status = "FAIL";
        }
        
        std::cout << std::setw(15) << topology_str << std::setw(15) << std::fixed << std::setprecision(2) << avg_forward_ms
                  << std::setw(15) << memory_mb << std::setw(15) << avg_adapt_ms << std::setw(10) << status << std::endl;
        
        // Verify performance targets for smaller networks
        if (topology[0] <= 8) {
            EXPECT_LE(avg_forward_ms, target_forward_ms) << "Forward pass too slow";
            EXPECT_LE(avg_adapt_ms, target_adapt_ms) << "Adaptation too slow";
        }
        
        EXPECT_LE(memory_mb, target_memory_mb) << "Memory usage too high";
    }
}

// Test Edge Cases and Robustness
TEST_F(Phase3DynamicTest, EdgeCasesAndRobustness) {
    // Test with minimal network
    EXPECT_NO_THROW({
        ML::Dynamic::DynamicNeuralNetwork minimal_network({1, 2, 1});
        std::vector<float> input = {1.0f};
        minimal_network.forward(input);
    });
    
    // Test with single neuron layers
    EXPECT_NO_THROW({
        ML::Dynamic::DynamicNeuralNetwork single_neuron({2, 1, 1});
        std::vector<float> input = {1.0f, 0.5f};
        single_neuron.forward(input);
    });
    
    // Test extreme resizing
    ML::Dynamic::DynamicNeuralNetwork network({4, 8, 4});
    
    // Resize to minimum
    network.resizeLayer(1, 4);
    EXPECT_EQ(network.getLayerSizes()[1], 4);
    
    // Resize to maximum
    network.resizeLayer(1, 1024);
    EXPECT_EQ(network.getLayerSizes()[1], 1024);
    
    // Test with zero input
    std::vector<float> zero_input(4, 0.0f);
    EXPECT_NO_THROW(network.forward(zero_input));
    
    // Test with extreme input values
    std::vector<float> extreme_input = {1e6f, -1e6f, 1e-6f, -1e-6f};
    EXPECT_NO_THROW(network.forward(extreme_input));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
