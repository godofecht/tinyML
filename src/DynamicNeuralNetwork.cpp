//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Dynamic Neural Systems Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "DynamicNeuralNetwork.h"
#include <algorithm>
#include <random>
#include <cstring>
#include <iostream>
#include <cmath>

namespace ML {
namespace Dynamic {

// DynamicLayer Implementation
DynamicLayer::DynamicLayer(const Config& config)
    : config_(config), use_simd_(XSIMD::XSIMDVector::has_simd_support()) {
    initialize_weights();
}

void DynamicLayer::initialize_weights() {
    size_t total_size = config_.input_size * config_.output_size;
    weights_.resize(total_size);
    biases_.resize(config_.output_size);
    
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (config_.input_size + config_.output_size));
    std::normal_distribution<float> dis(0.0f, scale);
    
    for (size_t i = 0; i < total_size; ++i) {
        weights_[i] = dis(gen);
    }
    
    for (size_t i = 0; i < config_.output_size; ++i) {
        biases_[i] = 0.0f;
    }
}

void DynamicLayer::apply_activation(std::vector<float>& data) {
    if (config_.activation == "tanh") {
        XSIMD::VectorOps::tanh_batch(data.data(), data.data(), data.size());
    } else if (config_.activation == "relu") {
        XSIMD::VectorOps::relu_batch(data.data(), data.data(), data.size());
    } else if (config_.activation == "sigmoid") {
        XSIMD::VectorOps::sigmoid_batch(data.data(), data.data(), data.size());
    }
}

void DynamicLayer::resize(size_t new_input_size, size_t new_output_size) {
    config_.input_size = new_input_size;
    config_.output_size = new_output_size;
    initialize_weights();
}

size_t DynamicLayer::get_memory_usage() const {
    return (weights_.size() + biases_.size()) * sizeof(float);
}

// DynamicDenseLayer Implementation
DynamicDenseLayer::DynamicDenseLayer(const Config& config)
    : DynamicLayer(config), output_buffer_(config.output_size) {
}

std::vector<float> DynamicDenseLayer::forward(const std::vector<float>& input) {
    if (input.size() != config_.input_size) {
        throw std::invalid_argument("Input size mismatch");
    }
    
    // Output = input * weights + biases
    if (use_simd_) {
        XSIMD::VectorOps::matrix_vector_multiply(
            weights_.data(), input.data(), output_buffer_.data(),
            config_.output_size, config_.input_size);
        XSIMD::VectorOps::vector_add_vector(
            output_buffer_.data(), biases_.data(), output_buffer_.data(),
            config_.output_size);
    } else {
        // Scalar fallback
        for (size_t i = 0; i < config_.output_size; ++i) {
            float sum = biases_[i];
            for (size_t j = 0; j < config_.input_size; ++j) {
                sum += input[j] * weights_[i * config_.input_size + j];
            }
            output_buffer_[i] = sum;
        }
    }
    
    // Apply activation
    apply_activation(output_buffer_);
    
    return output_buffer_;
}

void DynamicDenseLayer::resize(size_t new_input_size, size_t new_output_size) {
    DynamicLayer::resize(new_input_size, new_output_size);
    output_buffer_.resize(new_output_size);
}

// DynamicAttentionLayer Implementation
DynamicAttentionLayer::DynamicAttentionLayer(const AttentionConfig& config)
    : DynamicLayer(config), attn_config_(config) {
    
    // Allocate attention buffers
    size_t head_dim = config.output_size / config.num_heads;
    q_buffer_.resize(config.output_size);
    k_buffer_.resize(config.output_size);
    v_buffer_.resize(config.output_size);
    attention_scores_.resize(config.num_heads * config.sequence_length * config.sequence_length);
    context_buffer_.resize(config.num_heads * config.sequence_length * head_dim);
}

std::vector<float> DynamicAttentionLayer::forward(const std::vector<float>& input) {
    if (input.size() != config_.input_size) {
        throw std::invalid_argument("Input size mismatch");
    }
    
    // Simplified attention computation
    // In a full implementation, this would compute QKV projections and attention
    
    std::vector<float> output(config_.output_size, 0.0f);
    
    // Apply attention weights (simplified)
    if (use_simd_) {
        XSIMD::VectorOps::vector_mul_vector(
            input.data(), weights_.data(), output.data(),
            config_.input_size);
    } else {
        for (size_t i = 0; i < config_.input_size; ++i) {
            output[i] = input[i] * weights_[i];
        }
    }
    
    apply_activation(output);
    return output;
}

void DynamicAttentionLayer::resize(size_t new_input_size, size_t new_output_size) {
    DynamicLayer::resize(new_input_size, new_output_size);
    
    // Resize attention buffers
    size_t head_dim = new_output_size / attn_config_.num_heads;
    q_buffer_.resize(new_output_size);
    k_buffer_.resize(new_output_size);
    v_buffer_.resize(new_output_size);
    context_buffer_.resize(attn_config_.num_heads * attn_config_.sequence_length * head_dim);
}

// DynamicNeuralNetwork Implementation
DynamicNeuralNetwork::DynamicNeuralNetwork(const NetworkConfig& config)
    : config_(config), pool_offset_(0) {
    initialize_layers();
    layer_efficiencies_.resize(layers_.size(), 1.0f);
    recent_errors_.resize(100, 0.0f);
}

void DynamicNeuralNetwork::initialize_layers() {
    layers_.clear();
    
    for (size_t i = 0; i < config_.initial_topology.size(); ++i) {
        size_t input_size = (i == 0) ? config_.initial_topology[0] : config_.initial_topology[i-1];
        size_t output_size = config_.initial_topology[i];
        
        DynamicLayer::Config layer_config{input_size, output_size, "tanh", false, 0.0f};
        layers_.push_back(std::make_unique<DynamicDenseLayer>(layer_config));
    }
}

std::vector<float> DynamicNeuralNetwork::forward(const std::vector<float>& input) {
    std::vector<float> current = input;
    
    for (const auto& layer : layers_) {
        current = layer->forward(current);
    }
    
    return current;
}

void DynamicNeuralNetwork::adapt_topology(const std::vector<float>& recent_errors) {
    compute_layer_efficiencies();
    
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (config_.enable_growth && should_grow_layer(i)) {
            size_t new_size = std::min(layers_[i]->get_output_size() * 2, config_.max_layer_size);
            resize_layer(i, new_size);
        } else if (config_.enable_pruning && should_shrink_layer(i)) {
            size_t new_size = std::max(layers_[i]->get_output_size() / 2, config_.min_layer_size);
            resize_layer(i, new_size);
        }
    }
}

void DynamicNeuralNetwork::resize_layer(size_t layer_idx, size_t new_size) {
    if (layer_idx >= layers_.size()) return;
    
    size_t input_size = layers_[layer_idx]->get_input_size();
    layers_[layer_idx]->resize(input_size, new_size);
    
    // Update next layer's input size if it exists
    if (layer_idx + 1 < layers_.size()) {
        size_t next_output_size = layers_[layer_idx + 1]->get_output_size();
        layers_[layer_idx + 1]->resize(new_size, next_output_size);
    }
}

void DynamicNeuralNetwork::add_layer(size_t position, size_t size) {
    if (position > layers_.size()) return;
    
    size_t input_size = (position == 0) ? config_.initial_topology[0] : layers_[position-1]->get_output_size();
    size_t output_size = (position < layers_.size()) ? layers_[position]->get_input_size() : size;
    
    DynamicLayer::Config layer_config{input_size, output_size, "tanh", false, 0.0f};
    layers_.insert(layers_.begin() + position, std::make_unique<DynamicDenseLayer>(layer_config));
    
    // Update adjacent layers
    if (position < layers_.size() - 1) {
        layers_[position + 1]->resize(output_size, layers_[position + 1]->get_output_size());
    }
}

void DynamicNeuralNetwork::remove_layer(size_t position) {
    if (position >= layers_.size() || layers_.size() <= 1) return;
    
    layers_.erase(layers_.begin() + position);
    
    // Update adjacent layer
    if (position < layers_.size()) {
        size_t input_size = (position == 0) ? config_.initial_topology[0] : layers_[position-1]->get_output_size();
        layers_[position]->resize(input_size, layers_[position]->get_output_size());
    }
}

void DynamicNeuralNetwork::allocate_memory_pool(size_t total_size) {
    memory_pool_.resize(total_size);
    pool_offset_ = 0;
}

void DynamicNeuralNetwork::deallocate_memory_pool() {
    memory_pool_.clear();
    pool_offset_ = 0;
}

size_t DynamicNeuralNetwork::get_memory_usage() const {
    size_t total = 0;
    for (const auto& layer : layers_) {
        total += layer->get_memory_usage();
    }
    total += memory_pool_.size() * sizeof(float);
    return total;
}

void DynamicNeuralNetwork::optimize_for_inference() {
    // Optimize for inference: freeze topology, use SIMD
    for (auto& layer : layers_) {
        // Layers are already using SIMD when available
    }
}

void DynamicNeuralNetwork::optimize_for_training() {
    // Optimize for training: enable dynamic resizing
    // Already enabled through config
}

std::vector<size_t> DynamicNeuralNetwork::get_current_topology() const {
    std::vector<size_t> topology;
    for (const auto& layer : layers_) {
        topology.push_back(layer->get_output_size());
    }
    return topology;
}

float DynamicNeuralNetwork::get_efficiency_score() const {
    float total_efficiency = 0.0f;
    for (float efficiency : layer_efficiencies_) {
        total_efficiency += efficiency;
    }
    return total_efficiency / layer_efficiencies_.size();
}

void DynamicNeuralNetwork::compute_layer_efficiencies() {
    // Simplified efficiency computation based on layer utilization
    for (size_t i = 0; i < layers_.size(); ++i) {
        size_t layer_size = layers_[i]->get_output_size();
        size_t max_size = config_.max_layer_size;
        
        // Efficiency based on how well-utilized the layer is
        float utilization = static_cast<float>(layer_size) / max_size;
        layer_efficiencies_[i] = 1.0f - std::abs(0.5f - utilization) * 2.0f; // Peak at 50% utilization
    }
}

bool DynamicNeuralNetwork::should_grow_layer(size_t layer_idx) const {
    if (layer_idx >= layer_efficiencies_.size()) return false;
    
    float efficiency = layer_efficiencies_[layer_idx];
    size_t current_size = layers_[layer_idx]->get_output_size();
    
    return efficiency < 0.3f && current_size < config_.max_layer_size;
}

bool DynamicNeuralNetwork::should_shrink_layer(size_t layer_idx) const {
    if (layer_idx >= layer_efficiencies_.size()) return false;
    
    float efficiency = layer_efficiencies_[layer_idx];
    size_t current_size = layers_[layer_idx]->get_output_size();
    
    return efficiency < 0.2f && current_size > config_.min_layer_size;
}

// MemoryPool Implementation
MemoryPool& MemoryPool::getInstance() {
    static MemoryPool instance;
    return instance;
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    Block* block = find_free_block(size);
    if (block) {
        block->in_use = true;
        return block->ptr;
    }
    
    return allocate_new_block(size);
}

void MemoryPool::deallocate(void* ptr) {
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            return;
        }
    }
}

MemoryPool::Block* MemoryPool::find_free_block(size_t size) {
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            return &block;
        }
    }
    return nullptr;
}

void* MemoryPool::allocate_new_block(size_t size) {
    void* ptr = std::aligned_alloc(16, size);
    if (!ptr) return nullptr;
    
    blocks_.push_back({ptr, size, true});
    total_allocated_ += size;
    peak_usage_ = std::max(peak_usage_, total_allocated_);
    
    return ptr;
}

void MemoryPool::clear() {
    for (auto& block : blocks_) {
        std::free(block.ptr);
    }
    blocks_.clear();
    total_allocated_ = 0;
}

// EvolutionaryOptimizer Implementation
EvolutionaryOptimizer::EvolutionaryOptimizer(const Config& config)
    : config_(config), best_fitness_(std::numeric_limits<float>::max()), current_generation_(0) {
}

void EvolutionaryOptimizer::optimize(DynamicNeuralNetwork& network,
                                     const std::vector<std::vector<float>>& inputs,
                                     const std::vector<std::vector<float>>& targets) {
    initialize_population(network);
    
    for (current_generation_ = 0; current_generation_ < config_.max_generations; ++current_generation_) {
        evaluate_fitness(network, inputs, targets);
        selection();
        crossover();
        mutation();
        
        if (current_generation_ % 10 == 0) {
            std::cout << "Generation " << current_generation_ 
                      << ", Best Fitness: " << best_fitness_ << std::endl;
        }
    }
}

void EvolutionaryOptimizer::initialize_population(DynamicNeuralNetwork& network) {
    population_.resize(config_.population_size);
    fitness_scores_.resize(config_.population_size);
    
    // Initialize with random variations of the current network
    for (size_t i = 0; i < config_.population_size; ++i) {
        // In a full implementation, this would copy network parameters with mutations
        population_[i] = std::vector<float>(100, 0.0f); // Placeholder
    }
}

void EvolutionaryOptimizer::evaluate_fitness(DynamicNeuralNetwork& network,
                                           const std::vector<std::vector<float>>& inputs,
                                           const std::vector<std::vector<float>>& targets) {
    for (size_t i = 0; i < population_.size(); ++i) {
        fitness_scores_[i] = compute_network_error(network, inputs, targets);
    }
    
    // Track best fitness
    auto min_it = std::min_element(fitness_scores_.begin(), fitness_scores_.end());
    best_fitness_ = std::min(best_fitness_, *min_it);
}

float EvolutionaryOptimizer::compute_network_error(DynamicNeuralNetwork& network,
                                                const std::vector<std::vector<float>>& inputs,
                                                const std::vector<std::vector<float>>& targets) {
    float total_error = 0.0f;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = network.forward(inputs[i]);
        
        // Compute mean squared error
        float sample_error = 0.0f;
        for (size_t j = 0; j < output.size(); ++j) {
            float diff = output[j] - targets[i][j];
            sample_error += diff * diff;
        }
        total_error += sample_error / output.size();
    }
    
    return total_error / inputs.size();
}

void EvolutionaryOptimizer::selection() {
    // Tournament selection
    std::vector<std::vector<float>> new_population;
    new_population.reserve(config_.population_size);
    
    // Keep elite
    std::vector<size_t> indices(config_.population_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [this](size_t a, size_t b) { return fitness_scores_[a] < fitness_scores_[b]; });
    
    for (size_t i = 0; i < config_.elite_size; ++i) {
        new_population.push_back(population_[indices[i]]);
    }
    
    // Fill rest with tournament selection
    while (new_population.size() < config_.population_size) {
        size_t a = rand() % config_.population_size;
        size_t b = rand() % config_.population_size;
        size_t winner = (fitness_scores_[a] < fitness_scores_[b]) ? a : b;
        new_population.push_back(population_[winner]);
    }
    
    population_ = new_population;
}

void EvolutionaryOptimizer::crossover() {
    for (size_t i = config_.elite_size; i < population_.size(); i += 2) {
        if (i + 1 < population_.size() && (static_cast<float>(rand()) / RAND_MAX) < config_.crossover_rate) {
            // Simple single-point crossover
            size_t crossover_point = rand() % population_[i].size();
            std::swap_ranges(population_[i].begin() + crossover_point, 
                           population_[i].end(), 
                           population_[i+1].begin() + crossover_point);
        }
    }
}

void EvolutionaryOptimizer::mutation() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> mutation(0.0f, 0.1f);
    
    for (size_t i = config_.elite_size; i < population_.size(); ++i) {
        for (size_t j = 0; j < population_[i].size(); ++j) {
            if ((static_cast<float>(rand()) / RAND_MAX) < config_.mutation_rate) {
                population_[i][j] += mutation(gen);
            }
        }
    }
}

// AdaptiveInferenceSystem Implementation
AdaptiveInferenceSystem::AdaptiveInferenceSystem(const Config& config)
    : config_(config), performance_index_(0), is_adapting_(false) {
    
    DynamicNeuralNetwork::NetworkConfig net_config{
        {config.input_size, 128, 64, config.output_size},
        0.001f, 0.1f, 1024, 16, true, true
    };
    
    network_ = std::make_unique<DynamicNeuralNetwork>(net_config);
    
    EvolutionaryOptimizer::Config opt_config{50, 0.1f, 0.8f, 5, 100};
    optimizer_ = std::make_unique<EvolutionaryOptimizer>(opt_config);
    
    latency_history_.reserve(config.performance_window);
    accuracy_history_.reserve(config.performance_window);
}

std::vector<float> AdaptiveInferenceSystem::predict(const std::vector<float>& input) {
    auto start = std::chrono::high_resolution_clock::now();
    
    auto output = network_->forward(input);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float latency_ms = static_cast<float>(duration.count()) / 1000.0f;
    
    update_performance_window(latency_ms, 1.0f); // Assume 100% accuracy for now
    
    return output;
}

void AdaptiveInferenceSystem::update_performance(float latency_ms, float accuracy) {
    update_performance_window(latency_ms, accuracy);
    
    // Check if adaptation is needed
    float avg_latency = get_average_latency();
    if (avg_latency > config_.latency_threshold_ms) {
        trigger_adaptation();
    }
}

void AdaptiveInferenceSystem::update_performance_window(float latency, float accuracy) {
    latency_history_.push_back(latency);
    accuracy_history_.push_back(accuracy);
    
    if (latency_history_.size() > config_.performance_window) {
        latency_history_.erase(latency_history_.begin());
        accuracy_history_.erase(accuracy_history_.begin());
    }
    
    performance_index_ = (performance_index_ + 1) % config_.performance_window;
}

float AdaptiveInferenceSystem::get_average_latency() const {
    if (latency_history_.empty()) return 0.0f;
    
    float sum = 0.0f;
    for (float latency : latency_history_) {
        sum += latency;
    }
    return sum / latency_history_.size();
}

float AdaptiveInferenceSystem::get_average_accuracy() const {
    if (accuracy_history_.empty()) return 0.0f;
    
    float sum = 0.0f;
    for (float accuracy : accuracy_history_) {
        sum += accuracy;
    }
    return sum / accuracy_history_.size();
}

void AdaptiveInferenceSystem::trigger_adaptation() {
    if (!is_adapting_) {
        is_adapting_ = true;
        std::cout << "Triggering network adaptation..." << std::endl;
        
        // Simplified adaptation: resize layers based on performance
        auto current_topology = network_->get_current_topology();
        for (size_t i = 0; i < current_topology.size(); ++i) {
            if (current_topology[i] > 64) {
                network_->resize_layer(i, current_topology[i] / 2);
            }
        }
        
        is_adapting_ = false;
    }
}

} // namespace Dynamic
} // namespace ML
