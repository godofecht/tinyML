//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 7: Advanced Attention & Transformers
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef ADVANCED_ATTENTION_H
#define ADVANCED_ATTENTION_H

#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <cmath>
#include <random>
#include <chrono>

#include "XSIMDOperations.h"
#include "QuantizedOperations.h"

namespace ML {
namespace Advanced {

// Multi-modal attention configurations
enum class ModalityType {
    TEXT,
    VISION,
    AUDIO,
    VIDEO,
    SENSOR,
    TABULAR
};

// Attention types
enum class AttentionType {
    STANDARD,
    MULTI_HEAD,
    SPARSE,
    LINEAR,
    KERNEL_BASED,
    LOCAL,
    GLOBAL,
    HIERARCHICAL
};

// Advanced attention base class
class AdvancedAttention {
public:
    struct Config {
        size_t embed_dim = 512;
        size_t num_heads = 8;
        size_t seq_len = 1024;
        AttentionType attention_type = AttentionType::STANDARD;
        bool use_quantization = false;
        float dropout_rate = 0.1f;
        bool use_causal_masking = false;
        size_t local_window_size = 128;
        float sparsity_ratio = 0.5f;
    };
    
    AdvancedAttention(const Config& config);
    virtual ~AdvancedAttention() = default;
    
    // Core attention methods
    virtual std::vector<float> forward(const std::vector<float>& query,
                                     const std::vector<float>& key,
                                     const std::vector<float>& value) = 0;
    
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    
    // Performance and memory
    virtual size_t get_memory_usage() const = 0;
    virtual float get_computation_complexity() const = 0;
    virtual void benchmark_performance(size_t iterations = 100) = 0;
    
    // Configuration
    Config get_config() const { return config_; }
    void set_config(const Config& config) { config_ = config; }
    
protected:
    Config config_;
    std::mt19937 rng_;
    
    // Utility methods
    std::vector<float> softmax(const std::vector<float>& logits);
    std::vector<float> apply_dropout(const std::vector<float>& input);
    std::vector<float> causal_mask(const std::vector<float>& input);
};

// Multi-Modal Attention
class MultiModalAttention : public AdvancedAttention {
public:
    struct ModalityConfig {
        ModalityType type;
        size_t embed_dim;
        size_t seq_len;
        float weight;
        bool use_projection;
    };
    
    MultiModalAttention(const Config& config, 
                        const std::vector<ModalityConfig>& modality_configs);
    
    std::vector<float> forward(const std::vector<float>& query,
                             const std::vector<float>& key,
                             const std::vector<float>& value) override;
    
    std::vector<float> forward(const std::vector<float>& input) override;
    
    // Multi-modal specific methods
    void add_modality(const ModalityConfig& modality);
    void remove_modality(ModalityType type);
    void set_modality_weight(ModalityType type, float weight);
    
    size_t get_memory_usage() const override;
    float get_computation_complexity() const override;
    void benchmark_performance(size_t iterations = 100) override;
    
private:
    std::map<ModalityType, ModalityConfig> modality_configs_;
    std::vector<std::unique_ptr<ML::XSIMD::XSIMDVector>> modality_embeddings_;
    std::vector<float> modality_weights_;
    
    std::vector<float> fuse_modalities(const std::vector<std::vector<float>>& modality_inputs);
    std::vector<float> cross_modal_attention(const std::vector<float>& modality_a,
                                           const std::vector<float>& modality_b);
};

// Hierarchical Attention
class HierarchicalAttention : public AdvancedAttention {
public:
    struct LevelConfig {
        size_t level;
        size_t embed_dim;
        size_t num_heads;
        size_t window_size;
        float compression_ratio;
    };
    
    HierarchicalAttention(const Config& config, 
                          const std::vector<LevelConfig>& levels);
    
    std::vector<float> forward(const std::vector<float>& query,
                             const std::vector<float>& key,
                             const std::vector<float>& value) override;
    
    std::vector<float> forward(const std::vector<float>& input) override;
    
    // Hierarchical specific methods
    void add_level(const LevelConfig& level);
    void remove_level(size_t level);
    std::vector<float> get_level_output(size_t level) const;
    
    size_t get_memory_usage() const override;
    float get_computation_complexity() const override;
    void benchmark_performance(size_t iterations = 100) override;
    
private:
    std::vector<LevelConfig> levels_;
    std::vector<std::vector<float>> level_outputs_;
    std::vector<std::unique_ptr<ML::XSIMD::XSIMDVector>> level_weights_;
    
    std::vector<float> process_level(const std::vector<float>& input, 
                                    const LevelConfig& level_config);
    std::vector<float> aggregate_levels(const std::vector<std::vector<float>>& level_outputs);
};

// Sparse Attention (Longformer-style)
class SparseAttention : public AdvancedAttention {
public:
    struct SparseConfig {
        size_t local_window_size = 128;
        size_t global_tokens = 16;
        float sparsity_ratio = 0.1f;
        bool use_random_pattern = false;
        bool use_block_pattern = false;
        size_t block_size = 16;
    };
    
    SparseAttention(const Config& config, const SparseConfig& sparse_config);
    
    std::vector<float> forward(const std::vector<float>& query,
                             const std::vector<float>& key,
                             const std::vector<float>& value) override;
    
    std::vector<float> forward(const std::vector<float>& input) override;
    
    // Sparse specific methods
    void set_global_tokens(const std::vector<size_t>& global_indices);
    void update_sparsity_pattern();
    std::vector<std::pair<size_t, size_t>> get_attention_pattern() const;
    
    size_t get_memory_usage() const override;
    float get_computation_complexity() const override;
    void benchmark_performance(size_t iterations = 100) override;
    
private:
    SparseConfig sparse_config_;
    std::vector<size_t> global_tokens_;
    std::vector<std::vector<bool>> attention_mask_;
    
    std::vector<std::vector<bool>> generate_sparse_mask();
    std::vector<float> sparse_attention_computation(const std::vector<float>& query,
                                                   const std::vector<float>& key,
                                                   const std::vector<float>& value,
                                                   const std::vector<std::vector<bool>>& mask);
};

// Linear Attention (Performer-style)
class LinearAttention : public AdvancedAttention {
public:
    struct LinearConfig {
        std::string kernel_type = "elu"; // "elu", "relu", "gaussian"
        float kernel_param = 1.0f;
        bool use_feature_map = true;
        size_t feature_dim = 64;
        bool normalize_features = true;
    };
    
    LinearAttention(const Config& config, const LinearConfig& linear_config);
    
    std::vector<float> forward(const std::vector<float>& query,
                             const std::vector<float>& key,
                             const std::vector<float>& value) override;
    
    std::vector<float> forward(const std::vector<float>& input) override;
    
    // Linear specific methods
    void set_kernel_function(const std::string& kernel_type);
    std::vector<float> get_feature_map(const std::vector<float>& input);
    
    size_t get_memory_usage() const override;
    float get_computation_complexity() const override;
    void benchmark_performance(size_t iterations = 100) override;
    
private:
    LinearConfig linear_config_;
    std::vector<float> feature_map_;
    
    float kernel_function(float x);
    std::vector<float> apply_feature_map(const std::vector<float>& input);
    std::vector<float> linear_attention_computation(const std::vector<float>& query,
                                                   const std::vector<float>& key,
                                                   const std::vector<float>& value);
};

// Local Attention (Longformer local window)
class LocalAttention : public AdvancedAttention {
public:
    struct LocalConfig {
        size_t window_size = 128;
        bool use_causal_window = false;
        bool use_strided_window = false;
        size_t stride = 64;
        bool use_dilated_window = false;
        size_t dilation = 2;
    };
    
    LocalAttention(const Config& config, const LocalConfig& local_config);
    
    std::vector<float> forward(const std::vector<float>& query,
                             const std::vector<float>& key,
                             const std::vector<float>& value) override;
    
    std::vector<float> forward(const std::vector<float>& input) override;
    
    // Local specific methods
    void set_window_size(size_t window_size);
    std::vector<std::pair<size_t, size_t>> get_local_windows() const;
    
    size_t get_memory_usage() const override;
    float get_computation_complexity() const override;
    void benchmark_performance(size_t iterations = 100) override;
    
private:
    LocalConfig local_config_;
    
    std::vector<std::pair<size_t, size_t>> generate_local_windows() const;
    std::vector<float> local_attention_computation(const std::vector<float>& query,
                                                  const std::vector<float>& key,
                                                  const std::vector<float>& value,
                                                  const std::vector<std::pair<size_t, size_t>>& windows);
};

// Transformer Block with Advanced Attention
class AdvancedTransformerBlock {
public:
    struct BlockConfig {
        size_t embed_dim = 512;
        size_t ff_dim = 2048;
        size_t num_heads = 8;
        AttentionType attention_type = AttentionType::STANDARD;
        float dropout_rate = 0.1f;
        bool use_layer_norm = true;
        bool use_residual = true;
        bool use_gelu = true;
        size_t seq_len = 1024;
    };
    
    AdvancedTransformerBlock(const BlockConfig& config);
    ~AdvancedTransformerBlock() = default;
    
    std::vector<float> forward(const std::vector<float>& input);
    
    // Configuration
    BlockConfig get_config() const { return config_; }
    void set_config(const BlockConfig& config) { config_ = config; }
    
    // Performance
    size_t get_memory_usage() const;
    void benchmark_performance(size_t iterations = 100);
    
private:
    BlockConfig config_;
    std::unique_ptr<AdvancedAttention> attention_;
    std::vector<float> ff_weights1_, ff_weights2_;
    std::vector<float> ff_bias1_, ff_bias2_;
    std::vector<float> norm_weights_, norm_bias_;
    
    std::vector<float> feedforward(const std::vector<float>& input);
    std::vector<float> layer_norm(const std::vector<float>& input);
    std::vector<float> gelu(const std::vector<float>& input);
};

// Factory for creating advanced attention mechanisms
class AdvancedAttentionFactory {
public:
    static std::unique_ptr<AdvancedAttention> create_attention(
        AttentionType type, const AdvancedAttention::Config& config);
    
    static std::unique_ptr<AdvancedAttention> create_multi_modal_attention(
        const AdvancedAttention::Config& config,
        const std::vector<MultiModalAttention::ModalityConfig>& modality_configs);
    
    static std::unique_ptr<AdvancedAttention> create_hierarchical_attention(
        const AdvancedAttention::Config& config,
        const std::vector<HierarchicalAttention::LevelConfig>& levels);
    
    static std::unique_ptr<AdvancedAttention> create_sparse_attention(
        const AdvancedAttention::Config& config,
        const SparseAttention::SparseConfig& sparse_config);
    
    static std::unique_ptr<AdvancedAttention> create_linear_attention(
        const AdvancedAttention::Config& config,
        const LinearAttention::LinearConfig& linear_config);
    
    static std::unique_ptr<AdvancedAttention> create_local_attention(
        const AdvancedAttention::Config& config,
        const LocalAttention::LocalConfig& local_config);
    
    static std::unique_ptr<AdvancedTransformerBlock> create_transformer_block(
        const AdvancedTransformerBlock::BlockConfig& config);
};

// Performance benchmarks
class AdvancedAttentionBenchmarks {
public:
    struct BenchmarkResult {
        std::string attention_type;
        size_t seq_len;
        size_t embed_dim;
        float latency_ms;
        float throughput_tokens_per_sec;
        size_t memory_usage_mb;
        float accuracy_score;
    };
    
    static std::vector<BenchmarkResult> benchmark_all_attention_types(
        size_t seq_len = 1024, size_t embed_dim = 512);
    
    static BenchmarkResult benchmark_specific_attention(
        std::unique_ptr<AdvancedAttention> attention,
        const std::string& name,
        size_t iterations = 100);
    
    static void print_benchmark_results(const std::vector<BenchmarkResult>& results);
    static void save_benchmark_results(const std::vector<BenchmarkResult>& results,
                                     const std::string& filename);
};

// Helper function for attention type to string conversion
std::string attention_type_to_string(AttentionType type);

} // namespace Advanced
} // namespace ML

#endif // ADVANCED_ATTENTION_H
