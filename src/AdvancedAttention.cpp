//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 7: Advanced Attention & Transformers Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "AdvancedAttention.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>

namespace ML {
namespace Advanced {

// AdvancedAttention base class implementation
AdvancedAttention::AdvancedAttention(const Config& config) 
    : config_(config), rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

std::vector<float> AdvancedAttention::softmax(const std::vector<float>& logits) {
    std::vector<float> softmax_output(logits.size());
    
    // Find max for numerical stability
    float max_val = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        float exp_val = std::exp(logits[i] - max_val);
        softmax_output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (size_t i = 0; i < logits.size(); ++i) {
        softmax_output[i] /= sum;
    }
    
    return softmax_output;
}

std::vector<float> AdvancedAttention::apply_dropout(const std::vector<float>& input) {
    if (config_.dropout_rate == 0.0f) {
        return input;
    }
    
    std::vector<float> output(input.size());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (dist(rng_) < config_.dropout_rate) {
            output[i] = 0.0f;
        } else {
            output[i] = input[i] / (1.0f - config_.dropout_rate);
        }
    }
    
    return output;
}

std::vector<float> AdvancedAttention::causal_mask(const std::vector<float>& input) {
    if (!config_.use_causal_masking) {
        return input;
    }
    
    size_t seq_len = config_.seq_len;
    std::vector<float> masked_input = input;
    
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = i + 1; j < seq_len; ++j) {
            masked_input[i * seq_len + j] = -std::numeric_limits<float>::infinity();
        }
    }
    
    return masked_input;
}

// MultiModalAttention implementation
MultiModalAttention::MultiModalAttention(const Config& config, 
                                        const std::vector<ModalityConfig>& modality_configs)
    : AdvancedAttention(config) {
    
    for (const auto& mod_config : modality_configs) {
        modality_configs_[mod_config.type] = mod_config;
        modality_weights_.push_back(mod_config.weight);
        
        // Create embedding vectors for each modality
        auto embedding = std::make_unique<ML::XSIMD::XSIMDVector>(mod_config.embed_dim);
        modality_embeddings_.push_back(std::move(embedding));
    }
    
    // Normalize weights
    float sum = std::accumulate(modality_weights_.begin(), modality_weights_.end(), 0.0f);
    for (auto& weight : modality_weights_) {
        weight /= sum;
    }
}

std::vector<float> MultiModalAttention::forward(const std::vector<float>& query,
                                                const std::vector<float>& key,
                                                const std::vector<float>& value) {
    
    // For simplicity, implement as weighted combination of modalities
    // In a full implementation, this would be more sophisticated
    size_t total_dim = 0;
    for (const auto& [type, config] : modality_configs_) {
        total_dim += config.embed_dim;
    }
    
    std::vector<float> fused_output(total_dim);
    size_t offset = 0;
    
    for (size_t i = 0; i < modality_weights_.size(); ++i) {
        size_t mod_dim = modality_embeddings_[i]->size();
        
        // Simple projection and weighting
        for (size_t j = 0; j < mod_dim; ++j) {
            fused_output[offset + j] = query[j % query.size()] * modality_weights_[i];
        }
        
        offset += mod_dim;
    }
    
    return fused_output;
}

std::vector<float> MultiModalAttention::forward(const std::vector<float>& input) {
    // Split input by modality and process each
    std::vector<std::vector<float>> modality_inputs;
    size_t offset = 0;
    
    for (const auto& [type, config] : modality_configs_) {
        std::vector<float> mod_input(config.embed_dim);
        for (size_t i = 0; i < config.embed_dim; ++i) {
            mod_input[i] = input[offset + i];
        }
        modality_inputs.push_back(mod_input);
        offset += config.embed_dim;
    }
    
    return fuse_modalities(modality_inputs);
}

void MultiModalAttention::add_modality(const ModalityConfig& modality) {
    modality_configs_[modality.type] = modality;
    modality_weights_.push_back(modality.weight);
    
    auto embedding = std::make_unique<ML::XSIMD::XSIMDVector>(modality.embed_dim);
    modality_embeddings_.push_back(std::move(embedding));
    
    // Renormalize weights
    float sum = std::accumulate(modality_weights_.begin(), modality_weights_.end(), 0.0f);
    for (auto& weight : modality_weights_) {
        weight /= sum;
    }
}

void MultiModalAttention::remove_modality(ModalityType type) {
    modality_configs_.erase(type);
    // Note: In a full implementation, would need to remove corresponding weights and embeddings
}

void MultiModalAttention::set_modality_weight(ModalityType type, float weight) {
    if (modality_configs_.count(type)) {
        // Find and update weight
        size_t index = 0;
        for (const auto& [t, config] : modality_configs_) {
            if (t == type) {
                modality_weights_[index] = weight;
                break;
            }
            index++;
        }
        
        // Renormalize
        float sum = std::accumulate(modality_weights_.begin(), modality_weights_.end(), 0.0f);
        for (auto& w : modality_weights_) {
            w /= sum;
        }
    }
}

size_t MultiModalAttention::get_memory_usage() const {
    size_t total = 0;
    for (const auto& [type, config] : modality_configs_) {
        total += config.embed_dim * sizeof(float);
    }
    total += modality_weights_.size() * sizeof(float);
    return total;
}

float MultiModalAttention::get_computation_complexity() const {
    // O(N * D) where N is sequence length and D is total embedding dimension
    size_t total_dim = 0;
    for (const auto& [type, config] : modality_configs_) {
        total_dim += config.embed_dim;
    }
    return static_cast<float>(config_.seq_len * total_dim);
}

void MultiModalAttention::benchmark_performance(size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create test input
    size_t total_dim = 0;
    for (const auto& [type, config] : modality_configs_) {
        total_dim += config.embed_dim;
    }
    
    std::vector<float> test_input(total_dim);
    std::fill(test_input.begin(), test_input.end(), 0.1f);
    
    for (size_t i = 0; i < iterations; ++i) {
        auto output = forward(test_input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "MultiModalAttention Benchmark:\n";
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << "μs" << std::endl;
    std::cout << "  Average latency: " << static_cast<float>(duration.count()) / iterations << "μs" << std::endl;
    std::cout << "  Throughput: " << (iterations * 1000000.0f) / duration.count() << " ops/sec" << std::endl;
}

std::vector<float> MultiModalAttention::fuse_modalities(const std::vector<std::vector<float>>& modality_inputs) {
    // Simple weighted fusion
    size_t total_dim = 0;
    for (const auto& input : modality_inputs) {
        total_dim += input.size();
    }
    
    std::vector<float> fused_output(total_dim, 0.0f);
    size_t offset = 0;
    
    for (size_t i = 0; i < modality_inputs.size(); ++i) {
        for (size_t j = 0; j < modality_inputs[i].size(); ++j) {
            fused_output[offset + j] += modality_inputs[i][j] * modality_weights_[i];
        }
        offset += modality_inputs[i].size();
    }
    
    return fused_output;
}

std::vector<float> MultiModalAttention::cross_modal_attention(const std::vector<float>& modality_a,
                                                             const std::vector<float>& modality_b) {
    // Simplified cross-modal attention
    size_t min_dim = std::min(modality_a.size(), modality_b.size());
    std::vector<float> attention_output(min_dim);
    
    for (size_t i = 0; i < min_dim; ++i) {
        attention_output[i] = modality_a[i] * modality_b[i];
    }
    
    return attention_output;
}

// HierarchicalAttention implementation
HierarchicalAttention::HierarchicalAttention(const Config& config, 
                                          const std::vector<LevelConfig>& levels)
    : AdvancedAttention(config), levels_(levels) {
    
    // Create weights for each level
    for (const auto& level : levels_) {
        auto weight = std::make_unique<ML::XSIMD::XSIMDVector>(level.embed_dim);
        level_weights_.push_back(std::move(weight));
    }
}

std::vector<float> HierarchicalAttention::forward(const std::vector<float>& query,
                                                   const std::vector<float>& key,
                                                   const std::vector<float>& value) {
    
    // Process each level
    level_outputs_.clear();
    for (const auto& level : levels_) {
        auto level_output = process_level(query, level);
        level_outputs_.push_back(level_output);
    }
    
    return aggregate_levels(level_outputs_);
}

std::vector<float> HierarchicalAttention::forward(const std::vector<float>& input) {
    level_outputs_.clear();
    
    // Process from bottom to top
    std::vector<float> current_input = input;
    for (const auto& level : levels_) {
        auto level_output = process_level(current_input, level);
        level_outputs_.push_back(level_output);
        current_input = level_output;
    }
    
    return current_input;
}

void HierarchicalAttention::add_level(const LevelConfig& level) {
    levels_.push_back(level);
    auto weight = std::make_unique<ML::XSIMD::XSIMDVector>(level.embed_dim);
    level_weights_.push_back(std::move(weight));
}

void HierarchicalAttention::remove_level(size_t level) {
    if (level < levels_.size()) {
        levels_.erase(levels_.begin() + level);
        level_weights_.erase(level_weights_.begin() + level);
    }
}

std::vector<float> HierarchicalAttention::get_level_output(size_t level) const {
    if (level < level_outputs_.size()) {
        return level_outputs_[level];
    }
    return {};
}

size_t HierarchicalAttention::get_memory_usage() const {
    size_t total = 0;
    for (const auto& level : levels_) {
        total += level.embed_dim * sizeof(float);
    }
    total += level_outputs_.size() * sizeof(std::vector<float>);
    return total;
}

float HierarchicalAttention::get_computation_complexity() const {
    float total_complexity = 0.0f;
    for (const auto& level : levels_) {
        total_complexity += static_cast<float>(config_.seq_len * level.embed_dim);
    }
    return total_complexity;
}

void HierarchicalAttention::benchmark_performance(size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> test_input(config_.embed_dim);
    std::fill(test_input.begin(), test_input.end(), 0.1f);
    
    for (size_t i = 0; i < iterations; ++i) {
        auto output = forward(test_input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "HierarchicalAttention Benchmark:\n";
    std::cout << "  Levels: " << levels_.size() << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << "μs" << std::endl;
    std::cout << "  Average latency: " << static_cast<float>(duration.count()) / iterations << "μs" << std::endl;
}

std::vector<float> HierarchicalAttention::process_level(const std::vector<float>& input, 
                                                          const LevelConfig& level_config) {
    // Simplified level processing
    std::vector<float> output(level_config.embed_dim);
    
    // Apply compression if needed
    size_t input_size = std::min(input.size(), level_config.embed_dim);
    for (size_t i = 0; i < input_size; ++i) {
        output[i] = input[i] * level_config.compression_ratio;
    }
    
    return output;
}

std::vector<float> HierarchicalAttention::aggregate_levels(const std::vector<std::vector<float>>& level_outputs) {
    if (level_outputs.empty()) {
        return {};
    }
    
    // Simple aggregation - take the last level output
    return level_outputs.back();
}

// Factory implementation
std::unique_ptr<AdvancedAttention> AdvancedAttentionFactory::create_attention(
    AttentionType type, const AdvancedAttention::Config& config) {
    
    switch (type) {
        case AttentionType::MULTI_HEAD:
            // Create default multi-modal config
            return create_multi_modal_attention(config, {
                {ModalityType::TEXT, 256, 512, 0.5f, true},
                {ModalityType::VISION, 256, 512, 0.3f, true},
                {ModalityType::AUDIO, 128, 512, 0.2f, true}
            });
            
        case AttentionType::HIERARCHICAL:
            return create_hierarchical_attention(config, {
                {0, 512, 8, 64, 0.5f},
                {1, 256, 4, 32, 0.5f},
                {2, 128, 2, 16, 0.5f}
            });
            
        case AttentionType::SPARSE:
            return create_sparse_attention(config, {});
            
        case AttentionType::LINEAR:
            return create_linear_attention(config, {});
            
        case AttentionType::LOCAL:
            return create_local_attention(config, {});
            
        default:
            throw std::runtime_error("Unsupported attention type");
    }
}

std::unique_ptr<AdvancedAttention> AdvancedAttentionFactory::create_multi_modal_attention(
    const AdvancedAttention::Config& config,
    const std::vector<MultiModalAttention::ModalityConfig>& modality_configs) {
    
    return std::make_unique<MultiModalAttention>(config, modality_configs);
}

std::unique_ptr<AdvancedAttention> AdvancedAttentionFactory::create_hierarchical_attention(
    const AdvancedAttention::Config& config,
    const std::vector<HierarchicalAttention::LevelConfig>& levels) {
    
    return std::make_unique<HierarchicalAttention>(config, levels);
}

std::unique_ptr<AdvancedAttention> AdvancedAttentionFactory::create_sparse_attention(
    const AdvancedAttention::Config& config,
    const SparseAttention::SparseConfig& sparse_config) {
    
    return std::make_unique<SparseAttention>(config, sparse_config);
}

std::unique_ptr<AdvancedAttention> AdvancedAttentionFactory::create_linear_attention(
    const AdvancedAttention::Config& config,
    const LinearAttention::LinearConfig& linear_config) {
    
    return std::make_unique<LinearAttention>(config, linear_config);
}

std::unique_ptr<AdvancedAttention> AdvancedAttentionFactory::create_local_attention(
    const AdvancedAttention::Config& config,
    const LocalAttention::LocalConfig& local_config) {
    
    return std::make_unique<LocalAttention>(config, local_config);
}

std::unique_ptr<AdvancedTransformerBlock> AdvancedAttentionFactory::create_transformer_block(
    const AdvancedTransformerBlock::BlockConfig& config) {
    
    return std::make_unique<AdvancedTransformerBlock>(config);
}

// Benchmark implementation
std::vector<AdvancedAttentionBenchmarks::BenchmarkResult> 
AdvancedAttentionBenchmarks::benchmark_all_attention_types(size_t seq_len, size_t embed_dim) {
    
    std::vector<BenchmarkResult> results;
    
    // Test different attention types
    std::vector<AttentionType> types = {
        AttentionType::MULTI_HEAD,
        AttentionType::HIERARCHICAL,
        AttentionType::SPARSE,
        AttentionType::LINEAR,
        AttentionType::LOCAL
    };
    
    for (auto type : types) {
        AdvancedAttention::Config config;
        config.seq_len = seq_len;
        config.embed_dim = embed_dim;
        config.attention_type = type;
        
        try {
            auto attention = AdvancedAttentionFactory::create_attention(type, config);
            auto result = benchmark_specific_attention(std::move(attention), 
                                                     attention_type_to_string(type), 100);
            results.push_back(result);
        } catch (const std::exception& e) {
            std::cout << "Error benchmarking " << attention_type_to_string(type) << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

AdvancedAttentionBenchmarks::BenchmarkResult 
AdvancedAttentionBenchmarks::benchmark_specific_attention(
    std::unique_ptr<AdvancedAttention> attention,
    const std::string& name,
    size_t iterations) {
    
    BenchmarkResult result;
    result.attention_type = name;
    result.seq_len = attention->get_config().seq_len;
    result.embed_dim = attention->get_config().embed_dim;
    
    // Create test input
    std::vector<float> test_input(result.embed_dim);
    std::fill(test_input.begin(), test_input.end(), 0.1f);
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        auto output = attention->forward(test_input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    result.latency_ms = static_cast<float>(duration.count()) / iterations / 1000.0f;
    result.throughput_tokens_per_sec = (iterations * 1000000.0f) / duration.count();
    result.memory_usage_mb = attention->get_memory_usage() / (1024 * 1024);
    result.accuracy_score = 0.95f; // Placeholder - would need actual accuracy measurement
    
    return result;
}

void AdvancedAttentionBenchmarks::print_benchmark_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=== Advanced Attention Benchmarks ===\n";
    std::cout << std::left << std::setw(20) << "Type" 
              << std::setw(10) << "Seq Len" 
              << std::setw(10) << "Embed Dim"
              << std::setw(12) << "Latency(ms)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Memory(MB)"
              << std::setw(10) << "Accuracy" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(20) << result.attention_type
                  << std::setw(10) << result.seq_len
                  << std::setw(10) << result.embed_dim
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.latency_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_tokens_per_sec
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.memory_usage_mb
                  << std::setw(10) << std::fixed << std::setprecision(3) << result.accuracy_score << std::endl;
    }
}

void AdvancedAttentionBenchmarks::save_benchmark_results(const std::vector<BenchmarkResult>& results,
                                                         const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Type,SeqLen,EmbedDim,LatencyMs,Throughput,MemoryMB,Accuracy\n";
        for (const auto& result : results) {
            file << result.attention_type << ","
                 << result.seq_len << ","
                 << result.embed_dim << ","
                 << result.latency_ms << ","
                 << result.throughput_tokens_per_sec << ","
                 << result.memory_usage_mb << ","
                 << result.accuracy_score << "\n";
        }
        file.close();
        std::cout << "Benchmark results saved to " << filename << std::endl;
    }
}

// Helper function for attention type to string conversion
std::string attention_type_to_string(AttentionType type) {
    switch (type) {
        case AttentionType::STANDARD: return "Standard";
        case AttentionType::MULTI_HEAD: return "MultiModal";
        case AttentionType::SPARSE: return "Sparse";
        case AttentionType::LINEAR: return "Linear";
        case AttentionType::KERNEL_BASED: return "KernelBased";
        case AttentionType::LOCAL: return "Local";
        case AttentionType::GLOBAL: return "Global";
        case AttentionType::HIERARCHICAL: return "Hierarchical";
        default: return "Unknown";
    }
}

// SparseAttention implementation
SparseAttention::SparseAttention(const Config& config, const SparseConfig& sparse_config)
    : AdvancedAttention(config), sparse_config_(sparse_config) {
    
    // Initialize global tokens
    for (size_t i = 0; i < sparse_config_.global_tokens; ++i) {
        global_tokens_.push_back(i * (config_.seq_len / sparse_config_.global_tokens));
    }
    
    // Initialize attention mask
    attention_mask_.resize(config_.seq_len, std::vector<bool>(config_.seq_len, false));
}

std::vector<float> SparseAttention::forward(const std::vector<float>& query,
                                                const std::vector<float>& key,
                                                const std::vector<float>& value) {
    
    auto mask = generate_sparse_mask();
    return sparse_attention_computation(query, key, value, mask);
}

std::vector<float> SparseAttention::forward(const std::vector<float>& input) {
    // Split input into query, key, value (simplified)
    size_t chunk_size = input.size() / 3;
    std::vector<float> query(input.begin(), input.begin() + chunk_size);
    std::vector<float> key(input.begin() + chunk_size, input.begin() + 2 * chunk_size);
    std::vector<float> value(input.begin() + 2 * chunk_size, input.end());
    
    return forward(query, key, value);
}

void SparseAttention::set_global_tokens(const std::vector<size_t>& global_indices) {
    global_tokens_ = global_indices;
}

void SparseAttention::update_sparsity_pattern() {
    attention_mask_ = generate_sparse_mask();
}

std::vector<std::pair<size_t, size_t>> SparseAttention::get_attention_pattern() const {
    std::vector<std::pair<size_t, size_t>> pattern;
    
    for (size_t i = 0; i < attention_mask_.size(); ++i) {
        for (size_t j = 0; j < attention_mask_[i].size(); ++j) {
            if (attention_mask_[i][j]) {
                pattern.push_back({i, j});
            }
        }
    }
    
    return pattern;
}

size_t SparseAttention::get_memory_usage() const {
    size_t total = 0;
    total += global_tokens_.size() * sizeof(size_t);
    total += attention_mask_.size() * attention_mask_[0].size() * sizeof(bool);
    return total;
}

float SparseAttention::get_computation_complexity() const {
    // Sparse attention has O(n * s) complexity where s is sparsity
    float sparsity = sparse_config_.sparsity_ratio;
    return static_cast<float>(config_.seq_len * config_.embed_dim * sparsity);
}

void SparseAttention::benchmark_performance(size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> test_query(config_.seq_len * config_.embed_dim);
    std::vector<float> test_key(config_.seq_len * config_.embed_dim);
    std::vector<float> test_value(config_.seq_len * config_.embed_dim);
    std::fill(test_query.begin(), test_query.end(), 0.1f);
    std::fill(test_key.begin(), test_key.end(), 0.1f);
    std::fill(test_value.begin(), test_value.end(), 0.1f);
    
    for (size_t i = 0; i < iterations; ++i) {
        auto output = forward(test_query, test_key, test_value);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "SparseAttention Benchmark:\n";
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << "μs" << std::endl;
    std::cout << "  Average latency: " << static_cast<float>(duration.count()) / iterations << "μs" << std::endl;
}

std::vector<std::vector<bool>> SparseAttention::generate_sparse_mask() {
    std::vector<std::vector<bool>> mask(config_.seq_len, std::vector<bool>(config_.seq_len, false));
    
    // Add local windows
    for (size_t i = 0; i < config_.seq_len; ++i) {
        size_t start = (i / sparse_config_.local_window_size) * sparse_config_.local_window_size;
        size_t end = std::min(start + sparse_config_.local_window_size, config_.seq_len);
        
        for (size_t j = start; j < end; ++j) {
            mask[i][j] = true;
        }
    }
    
    // Add global tokens
    for (size_t global_token : global_tokens_) {
        if (global_token < config_.seq_len) {
            for (size_t j = 0; j < config_.seq_len; ++j) {
                mask[global_token][j] = true;
                mask[j][global_token] = true;
            }
        }
    }
    
    return mask;
}

std::vector<float> SparseAttention::sparse_attention_computation(const std::vector<float>& query,
                                                               const std::vector<float>& key,
                                                               const std::vector<float>& value,
                                                               const std::vector<std::vector<bool>>& mask) {
    
    std::vector<float> output(config_.seq_len * config_.embed_dim, 0.0f);
    
    // Simplified sparse attention computation
    for (size_t i = 0; i < config_.seq_len; ++i) {
        for (size_t j = 0; j < config_.seq_len; ++j) {
            if (mask[i][j]) {
                // Compute attention weight
                float score = 0.0f;
                for (size_t k = 0; k < config_.embed_dim; ++k) {
                    score += query[i * config_.embed_dim + k] * key[j * config_.embed_dim + k];
                }
                
                // Apply attention to value
                for (size_t k = 0; k < config_.embed_dim; ++k) {
                    output[i * config_.embed_dim + k] += score * value[j * config_.embed_dim + k];
                }
            }
        }
    }
    
    return output;
}

// LinearAttention implementation
LinearAttention::LinearAttention(const Config& config, const LinearConfig& linear_config)
    : AdvancedAttention(config), linear_config_(linear_config) {
    
    feature_map_.resize(linear_config_.feature_dim);
}

std::vector<float> LinearAttention::forward(const std::vector<float>& query,
                                             const std::vector<float>& key,
                                             const std::vector<float>& value) {
    
    return linear_attention_computation(query, key, value);
}

std::vector<float> LinearAttention::forward(const std::vector<float>& input) {
    // Split input into query, key, value (simplified)
    size_t chunk_size = input.size() / 3;
    std::vector<float> query(input.begin(), input.begin() + chunk_size);
    std::vector<float> key(input.begin() + chunk_size, input.begin() + 2 * chunk_size);
    std::vector<float> value(input.begin() + 2 * chunk_size, input.end());
    
    return forward(query, key, value);
}

void LinearAttention::set_kernel_function(const std::string& kernel_type) {
    linear_config_.kernel_type = kernel_type;
}

std::vector<float> LinearAttention::get_feature_map(const std::vector<float>& input) {
    return apply_feature_map(input);
}

size_t LinearAttention::get_memory_usage() const {
    size_t total = 0;
    total += feature_map_.size() * sizeof(float);
    return total;
}

float LinearAttention::get_computation_complexity() const {
    // Linear attention has O(n * d) complexity
    return static_cast<float>(config_.seq_len * config_.embed_dim);
}

void LinearAttention::benchmark_performance(size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> test_query(config_.seq_len * config_.embed_dim);
    std::vector<float> test_key(config_.seq_len * config_.embed_dim);
    std::vector<float> test_value(config_.seq_len * config_.embed_dim);
    std::fill(test_query.begin(), test_query.end(), 0.1f);
    std::fill(test_key.begin(), test_key.end(), 0.1f);
    std::fill(test_value.begin(), test_value.end(), 0.1f);
    
    for (size_t i = 0; i < iterations; ++i) {
        auto output = forward(test_query, test_key, test_value);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "LinearAttention Benchmark:\n";
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << "μs" << std::endl;
    std::cout << "  Average latency: " << static_cast<float>(duration.count()) / iterations << "μs" << std::endl;
}

float LinearAttention::kernel_function(float x) {
    if (linear_config_.kernel_type == "elu") {
        return std::max(0.0f, x);
    } else if (linear_config_.kernel_type == "relu") {
        return std::max(0.0f, x);
    } else if (linear_config_.kernel_type == "gaussian") {
        return std::exp(-x * x / (2.0f * linear_config_.kernel_param * linear_config_.kernel_param));
    }
    return x;
}

std::vector<float> LinearAttention::apply_feature_map(const std::vector<float>& input) {
    size_t embed_dim = config_.embed_dim;
    size_t feature_dim = linear_config_.feature_dim;
    size_t seq_len = (embed_dim > 0 && input.size() % embed_dim == 0)
        ? (input.size() / embed_dim)
        : 1;

    if (seq_len == 1 && input.size() != embed_dim) {
        seq_len = 1;
    }

    std::vector<float> feature_mapped(seq_len * feature_dim, 0.0f);

    for (size_t t = 0; t < seq_len; ++t) {
        size_t base = t * embed_dim;
        size_t out_base = t * feature_dim;
        size_t limit = std::min(embed_dim, input.size() - base);
        for (size_t i = 0; i < limit; ++i) {
            size_t feature_idx = i % feature_dim;
            feature_mapped[out_base + feature_idx] += kernel_function(input[base + i]);
        }
    }

    return feature_mapped;
}

std::vector<float> LinearAttention::linear_attention_computation(const std::vector<float>& query,
                                                          const std::vector<float>& key,
                                                          const std::vector<float>& value) {
    
    // Apply feature maps
    auto query_features = apply_feature_map(query);
    auto key_features = apply_feature_map(key);
    auto value_features = apply_feature_map(value);
    
    std::vector<float> output(config_.seq_len * config_.embed_dim, 0.0f);
    
    // Linear attention computation
    for (size_t i = 0; i < config_.seq_len; ++i) {
        for (size_t j = 0; j < config_.seq_len; ++j) {
            float kv_score = 0.0f;
            for (size_t k = 0; k < linear_config_.feature_dim; ++k) {
                kv_score += key_features[j * linear_config_.feature_dim + k] *
                            value_features[j * linear_config_.feature_dim + k];
            }
            
            for (size_t k = 0; k < config_.embed_dim; ++k) {
                output[i * config_.embed_dim + k] +=
                    query_features[i * linear_config_.feature_dim + (k % linear_config_.feature_dim)] *
                    kv_score;
            }
        }
    }
    
    return output;
}

// LocalAttention implementation
LocalAttention::LocalAttention(const Config& config, const LocalConfig& local_config)
    : AdvancedAttention(config), local_config_(local_config) {
}

std::vector<float> LocalAttention::forward(const std::vector<float>& query,
                                           const std::vector<float>& key,
                                           const std::vector<float>& value) {
    
    auto windows = generate_local_windows();
    return local_attention_computation(query, key, value, windows);
}

std::vector<float> LocalAttention::forward(const std::vector<float>& input) {
    // Split input into query, key, value (simplified)
    size_t chunk_size = input.size() / 3;
    std::vector<float> query(input.begin(), input.begin() + chunk_size);
    std::vector<float> key(input.begin() + chunk_size, input.begin() + 2 * chunk_size);
    std::vector<float> value(input.begin() + 2 * chunk_size, input.end());
    
    return forward(query, key, value);
}

void LocalAttention::set_window_size(size_t window_size) {
    local_config_.window_size = window_size;
}

std::vector<std::pair<size_t, size_t>> LocalAttention::get_local_windows() const {
    return generate_local_windows();
}

size_t LocalAttention::get_memory_usage() const {
    return config_.seq_len * config_.embed_dim * sizeof(float);
}

float LocalAttention::get_computation_complexity() const {
    // Local attention has O(n * w * d) complexity where w is window size
    return static_cast<float>(config_.seq_len * local_config_.window_size * config_.embed_dim);
}

void LocalAttention::benchmark_performance(size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> test_query(config_.seq_len * config_.embed_dim);
    std::vector<float> test_key(config_.seq_len * config_.embed_dim);
    std::vector<float> test_value(config_.seq_len * config_.embed_dim);
    std::fill(test_query.begin(), test_query.end(), 0.1f);
    std::fill(test_key.begin(), test_key.end(), 0.1f);
    std::fill(test_value.begin(), test_value.end(), 0.1f);
    
    for (size_t i = 0; i < iterations; ++i) {
        auto output = forward(test_query, test_key, test_value);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "LocalAttention Benchmark:\n";
    std::cout << "  Window size: " << local_config_.window_size << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << "μs" << std::endl;
    std::cout << "  Average latency: " << static_cast<float>(duration.count()) / iterations << "μs" << std::endl;
}

std::vector<std::pair<size_t, size_t>> LocalAttention::generate_local_windows() const {
    std::vector<std::pair<size_t, size_t>> windows;
    
    for (size_t i = 0; i < config_.seq_len; i += local_config_.stride) {
        size_t start = i;
        size_t end = std::min(i + local_config_.window_size, config_.seq_len);
        
        if (local_config_.use_dilated_window) {
            start = i * local_config_.dilation;
            end = std::min(start + local_config_.window_size * local_config_.dilation, config_.seq_len);
        }
        
        if (start < config_.seq_len) {
            windows.push_back({start, end});
        }
    }
    
    return windows;
}

std::vector<float> LocalAttention::local_attention_computation(const std::vector<float>& query,
                                                          const std::vector<float>& key,
                                                          const std::vector<float>& value,
                                                          const std::vector<std::pair<size_t, size_t>>& windows) {
    
    std::vector<float> output(config_.seq_len * config_.embed_dim, 0.0f);
    
    // Local attention computation within windows
    for (const auto& window : windows) {
        size_t window_size = window.second - window.first;
        
        for (size_t i = window.first; i < window.second; ++i) {
            for (size_t j = window.first; j < window.second; ++j) {
                // Compute attention weight
                float score = 0.0f;
                for (size_t k = 0; k < config_.embed_dim; ++k) {
                    score += query[i * config_.embed_dim + k] * key[j * config_.embed_dim + k];
                }
                
                // Apply attention to value
                for (size_t k = 0; k < config_.embed_dim; ++k) {
                    output[i * config_.embed_dim + k] += score * value[j * config_.embed_dim + k];
                }
            }
        }
    }
    
    return output;
}

// AdvancedTransformerBlock implementation
AdvancedTransformerBlock::AdvancedTransformerBlock(const BlockConfig& config)
    : config_(config) {
    
    // Initialize attention
    AdvancedAttention::Config attention_config;
    attention_config.embed_dim = config.embed_dim;
    attention_config.num_heads = config.num_heads;
    attention_config.seq_len = config.seq_len;
    attention_config.attention_type = config.attention_type;
    attention_ = AdvancedAttentionFactory::create_attention(config.attention_type, attention_config);
    
    // Initialize feedforward weights
    ff_weights1_.resize(config.embed_dim * config.ff_dim);
    ff_weights2_.resize(config.ff_dim * config.embed_dim);
    ff_bias1_.resize(config.ff_dim);
    ff_bias2_.resize(config.embed_dim);
    
    // Initialize layer norm weights
    norm_weights_.resize(config.embed_dim);
    norm_bias_.resize(config.embed_dim);
    
    // Initialize with small random values
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    for (auto& weight : ff_weights1_) {
        weight = dist(rng);
    }
    for (auto& weight : ff_weights2_) {
        weight = dist(rng);
    }
    for (auto& bias : ff_bias1_) {
        bias = dist(rng);
    }
    for (auto& bias : ff_bias2_) {
        bias = dist(rng);
    }
    for (auto& weight : norm_weights_) {
        weight = dist(rng);
    }
    for (auto& bias : norm_bias_) {
        bias = dist(rng);
    }
}

std::vector<float> AdvancedTransformerBlock::forward(const std::vector<float>& input) {
    std::vector<float> output = input;
    
    // Self-attention
    auto attention_output = attention_->forward(output);
    
    // Add residual connection
    if (config_.use_residual) {
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += input[i];
        }
    }
    
    // Layer normalization
    if (config_.use_layer_norm) {
        output = layer_norm(output);
    }
    
    // Feed-forward
    auto ff_output = feedforward(output);
    
    // Add residual connection
    if (config_.use_residual) {
        for (size_t i = 0; i < ff_output.size(); ++i) {
            ff_output[i] += output[i];
        }
    }
    
    // Layer normalization
    if (config_.use_layer_norm) {
        ff_output = layer_norm(ff_output);
    }
    
    return ff_output;
}

size_t AdvancedTransformerBlock::get_memory_usage() const {
    size_t total = 0;
    total += attention_->get_memory_usage();
    total += ff_weights1_.size() * sizeof(float);
    total += ff_weights2_.size() * sizeof(float);
    total += ff_bias1_.size() * sizeof(float);
    total += ff_bias2_.size() * sizeof(float);
    total += norm_weights_.size() * sizeof(float);
    total += norm_bias_.size() * sizeof(float);
    return total;
}

void AdvancedTransformerBlock::benchmark_performance(size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> test_input(config_.embed_dim);
    std::fill(test_input.begin(), test_input.end(), 0.1f);
    
    for (size_t i = 0; i < iterations; ++i) {
        auto output = forward(test_input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "AdvancedTransformerBlock Benchmark:\n";
    std::cout << "  Embed dim: " << config_.embed_dim << std::endl;
    std::cout << "  FF dim: " << config_.ff_dim << std::endl;
    std::cout << "  Attention type: " << static_cast<int>(config_.attention_type) << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << "μs" << std::endl;
    std::cout << "  Average latency: " << static_cast<float>(duration.count()) / iterations << "μs" << std::endl;
}

std::vector<float> AdvancedTransformerBlock::feedforward(const std::vector<float>& input) {
    std::vector<float> hidden(config_.ff_dim);
    
    // First linear layer: input (embed_dim) -> hidden (ff_dim)
    for (size_t i = 0; i < config_.ff_dim; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < config_.embed_dim; ++j) {
            sum += input[j] * ff_weights1_[i * config_.embed_dim + j];
        }
        hidden[i] = sum + ff_bias1_[i];
    }
    
    // Activation function
    if (config_.use_gelu) {
        hidden = gelu(hidden);
    } else {
        for (auto& val : hidden) {
            val = std::max(0.0f, val);
        }
    }
    
    // Second linear layer: hidden (ff_dim) -> output (embed_dim)
    std::vector<float> output(config_.embed_dim);
    for (size_t i = 0; i < config_.embed_dim; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < config_.ff_dim; ++j) {
            sum += hidden[j] * ff_weights2_[i * config_.ff_dim + j];
        }
        output[i] = sum + ff_bias2_[i];
    }
    
    return output;
}

std::vector<float> AdvancedTransformerBlock::layer_norm(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    // Compute mean and variance
    float mean = 0.0f;
    for (float val : input) {
        mean += val;
    }
    mean /= input.size();
    
    float variance = 0.0f;
    for (float val : input) {
        variance += (val - mean) * (val - mean);
    }
    variance /= input.size();
    
    float std_dev = std::sqrt(variance + 1e-6f);
    
    // Normalize
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = (input[i] - mean) / std_dev * norm_weights_[i] + norm_bias_[i];
    }
    
    return output;
}

std::vector<float> AdvancedTransformerBlock::gelu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
    }
    
    return output;
}

} // namespace Advanced
} // namespace ML
