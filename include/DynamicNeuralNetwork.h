//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Dynamic Neural Systems - Phase 3 Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef DYNAMIC_NEURAL_NETWORK_H
#define DYNAMIC_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
#include <random>
#include "XSIMDOperations.h"

namespace ML {
namespace Dynamic {

// Dynamic layer that can resize at runtime
class DynamicLayer {
public:
    struct Config {
        size_t input_size = 0;
        size_t output_size = 0;
        std::string activation = "tanh";
        bool use_batch_norm = false;
        float dropout_rate = 0.0f;
    };
    
    DynamicLayer(const Config& config);
    virtual ~DynamicLayer() = default;
    
    // Core operations
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    virtual void resize(size_t new_input_size, size_t new_output_size);
    virtual size_t get_memory_usage() const;
    virtual void perturb_weights(float sigma, unsigned int seed, float direction = 1.0f) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, sigma);
        for (auto& w : weights_) w += direction * dist(gen);
        for (auto& b : biases_) b += direction * dist(gen);
    }

    virtual std::vector<float> get_weights() const {
        std::vector<float> all_params = weights_;
        all_params.insert(all_params.end(), biases_.begin(), biases_.end());
        return all_params;
    }

    virtual std::vector<float> get_weights_structured() const {
        return weights_;
    }
    
    // Configuration
    size_t get_input_size() const { return config_.input_size; }
    size_t get_output_size() const { return config_.output_size; }
    
protected:
    Config config_;
    std::vector<float> weights_;
    std::vector<float> biases_;
    bool use_simd_;
    
    void initialize_weights();
    void apply_activation(std::vector<float>& data);
};

// Dynamic fully connected layer with XSIMD optimization
class DynamicDenseLayer : public DynamicLayer {
public:
    DynamicDenseLayer(const Config& config);
    
    std::vector<float> forward(const std::vector<float>& input) override;
    void resize(size_t new_input_size, size_t new_output_size) override;
    
private:
    std::vector<float> output_buffer_;
};

// Dynamic attention layer
class DynamicAttentionLayer : public DynamicLayer {
public:
    struct AttentionConfig : public Config {
        size_t num_heads = 8;
        size_t sequence_length = 512;
        bool use_causal_mask = false;
    };
    
    DynamicAttentionLayer(const AttentionConfig& config);
    
    std::vector<float> forward(const std::vector<float>& input) override;
    void resize(size_t new_input_size, size_t new_output_size) override;
    
private:
    AttentionConfig attn_config_;
    std::vector<float> q_buffer_, k_buffer_, v_buffer_;
    std::vector<float> attention_scores_;
    std::vector<float> context_buffer_;
};

// Dynamic neural network with adaptive topology
class DynamicNeuralNetwork {
public:
    struct NetworkConfig {
        std::vector<size_t> initial_topology = {256, 128, 64, 32};
        float learning_rate = 0.001f;
        float adaptation_threshold = 0.1f;
        size_t max_layer_size = 1024;
        size_t min_layer_size = 16;
        bool enable_growth = true;
        bool enable_pruning = true;
    };
    
    DynamicNeuralNetwork();
    DynamicNeuralNetwork(const NetworkConfig& config);
    ~DynamicNeuralNetwork() = default;
    
    // Core operations
    std::vector<float> forward(const std::vector<float>& input);
    void adapt_topology(const std::vector<float>& recent_errors);
    
    // Dynamic resizing
    void resize_layer(size_t layer_idx, size_t new_size);
    void addLayer(size_t size);
    void add_layer(size_t position, size_t size);
    void remove_layer(size_t position);
    
    // Memory management
    void allocate_memory_pool(size_t total_size);
    void deallocate_memory_pool();
    size_t get_memory_usage() const;
    
    // Optimization
    void optimize_for_inference();
    void optimize_for_training();
    
    // Monitoring
    std::vector<size_t> get_current_topology() const;
    float get_efficiency_score() const;
    
private:
    NetworkConfig config_;
    std::vector<std::unique_ptr<DynamicLayer>> layers_;
    std::vector<size_t> manual_topology_;
    std::vector<float> memory_pool_;
    size_t pool_offset_;
    
    // Adaptation metrics
    std::vector<float> layer_efficiencies_;
    std::vector<float> recent_errors_;
    
    void initialize_layers();
    void rebuild_layers_from_manual_topology();
    void compute_layer_efficiencies();
    bool should_grow_layer(size_t layer_idx) const;
    bool should_shrink_layer(size_t layer_idx) const;
};

// Memory pool manager for dynamic allocations
class MemoryPool {
public:
    static MemoryPool& getInstance();
    
    void* allocate(size_t size, size_t alignment = 16);
    void deallocate(void* ptr);
    void clear();
    
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_peak_usage() const { return peak_usage_; }
    
private:
    MemoryPool() = default;
    ~MemoryPool() = default;
    
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks_;
    size_t total_allocated_ = 0;
    size_t peak_usage_ = 0;
    
    Block* find_free_block(size_t size);
    void* allocate_new_block(size_t size);
};

// Gradient-free optimization using evolutionary strategies
class EvolutionaryOptimizer {
public:
    struct Config {
        size_t population_size = 50;
        float mutation_rate = 0.1f;
        float crossover_rate = 0.8f;
        size_t elite_size = 5;
        size_t max_generations = 100;
    };
    
    EvolutionaryOptimizer(const Config& config);
    
    // Optimize network parameters without gradients
    void optimize(DynamicNeuralNetwork& network, 
                 const std::vector<std::vector<float>>& inputs,
                 const std::vector<std::vector<float>>& targets);
    
    float get_best_fitness() const { return best_fitness_; }
    size_t get_generation() const { return current_generation_; }
    
private:
    Config config_;
    std::vector<std::vector<float>> population_;
    std::vector<float> fitness_scores_;
    float best_fitness_;
    size_t current_generation_;
    
    void initialize_population(DynamicNeuralNetwork& network);
    void evaluate_fitness(DynamicNeuralNetwork& network,
                        const std::vector<std::vector<float>>& inputs,
                        const std::vector<std::vector<float>>& targets);
    void selection();
    void crossover();
    void mutation();
    
    float compute_network_error(DynamicNeuralNetwork& network,
                               const std::vector<std::vector<float>>& inputs,
                               const std::vector<std::vector<float>>& targets);
};

// Real-time adaptive inference system
class AdaptiveInferenceSystem {
public:
    struct Config {
        size_t input_size = 256;
        size_t output_size = 64;
        float adaptation_rate = 0.01f;
        size_t performance_window = 100;
        float latency_threshold_ms = 5.0f;
    };
    
    AdaptiveInferenceSystem(const Config& config);
    
    // Real-time inference with adaptation
    std::vector<float> predict(const std::vector<float>& input);
    void update_performance(float latency_ms, float accuracy);
    
    // System monitoring
    float get_average_latency() const;
    float get_average_accuracy() const;
    bool is_adapting() const { return is_adapting_; }
    
private:
    Config config_;
    std::unique_ptr<DynamicNeuralNetwork> network_;
    std::unique_ptr<EvolutionaryOptimizer> optimizer_;
    
    // Performance tracking
    std::vector<float> latency_history_;
    std::vector<float> accuracy_history_;
    size_t performance_index_;
    bool is_adapting_;
    
    void trigger_adaptation();
    void update_performance_window(float latency, float accuracy);
};

} // namespace Dynamic
} // namespace ML

#endif // DYNAMIC_NEURAL_NETWORK_H
