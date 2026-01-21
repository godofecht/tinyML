//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 5: Quantized Operations - Advanced Optimizations
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef QUANTIZED_OPERATIONS_H
#define QUANTIZED_OPERATIONS_H

#include <vector>
#include <cstdint>
#include <memory>

namespace ML {
namespace Quantized {

// Quantization parameters
struct QuantParams {
    float scale;
    int32_t zero_point;
    int8_t min_val;
    int8_t max_val;
    
    QuantParams(float s = 1.0f, int32_t zp = 0, int8_t min = -128, int8_t max = 127)
        : scale(s), zero_point(zp), min_val(min), max_val(max) {}
};

// 8-bit quantized vector operations
class QuantizedVectorOps {
public:
    // Quantization utilities
    static QuantParams calibrate_quantization(const std::vector<float>& weights);
    static std::vector<int8_t> quantize(const std::vector<float>& input, const QuantParams& params);
    static std::vector<float> dequantize(const std::vector<int8_t>& input, const QuantParams& params);
    
    // Vector operations
    static std::vector<int8_t> vector_add_quantized(
        const std::vector<int8_t>& a, const std::vector<int8_t>& b, 
        const QuantParams& params);
    
    static std::vector<int8_t> vector_mul_quantized(
        const std::vector<int8_t>& a, const std::vector<int8_t>& b,
        const QuantParams& params);
    
    // Matrix-vector multiplication (quantized)
    static std::vector<int8_t> matrix_vector_multiply_quantized(
        const std::vector<int8_t>& weights, const std::vector<int8_t>& input,
        size_t output_size, size_t input_size, const QuantParams& params);
    
    // Activation functions (quantized)
    static std::vector<int8_t> relu_quantized(const std::vector<int8_t>& input, const QuantParams& params);
    static std::vector<int8_t> tanh_quantized(const std::vector<int8_t>& input, const QuantParams& params);
    
    // Performance benchmarks
    static void benchmark_quantized_vs_float(size_t vector_size, size_t iterations);
};

// 4-bit quantized operations (experimental)
class UltraQuantizedOps {
public:
    // 4-bit packing utilities
    static std::vector<uint8_t> pack_4bit(const std::vector<int8_t>& input);
    static std::vector<int8_t> unpack_4bit(const std::vector<uint8_t>& packed);
    
    // 4-bit operations
    static std::vector<uint8_t> vector_add_4bit(
        const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
        const QuantParams& params);
    
    static std::vector<uint8_t> matrix_vector_multiply_4bit(
        const std::vector<uint8_t>& weights, const std::vector<uint8_t>& input,
        size_t output_size, size_t input_size, const QuantParams& params);
};

// Quantized attention mechanism
class QuantizedAttention {
public:
    struct Config {
        size_t embed_dim = 256;
        size_t num_heads = 8;
        size_t head_dim = 32;
        bool use_4bit = false;
        bool sparse_attention = false;
        float sparsity_ratio = 0.5f;
    };
    
    QuantizedAttention(const Config& config);
    ~QuantizedAttention() = default;
    
    // Forward pass with quantization
    std::vector<int8_t> forward(const std::vector<int8_t>& input);
    
    // Memory usage optimization
    size_t get_memory_usage() const;
    void optimize_memory_layout();
    
    // Sparse attention support
    void enable_sparse_attention(float sparsity_ratio);
    void disable_sparse_attention();
    
private:
    Config config_;
    std::vector<QuantParams> quant_params_;
    std::vector<int8_t> q_weights_, k_weights_, v_weights_, o_weights_;
    std::vector<int8_t> attention_buffer_;
    
    bool sparse_enabled_;
    std::vector<bool> attention_mask_;
    
    void initialize_quantized_weights();
    std::vector<int8_t> compute_qkv_quantized(const std::vector<int8_t>& input);
    std::vector<int8_t> scaled_dot_product_quantized(
        const std::vector<int8_t>& q, const std::vector<int8_t>& k, const std::vector<int8_t>& v);
};

// Kernel fusion operations
class FusedOps {
public:
    // Fused attention + feedforward
    static std::vector<int8_t> fused_attention_ff(
        const std::vector<int8_t>& input,
        const std::vector<int8_t>& attn_weights,
        const std::vector<int8_t>& ff_weights,
        const QuantParams& params);
    
    // Fused layer normalization + activation
    static std::vector<int8_t> fused_norm_activation(
        const std::vector<int8_t>& input,
        const std::vector<int8_t>& norm_weights,
        const std::vector<int8_t>& activation_weights,
        const QuantParams& params);
    
    // Performance comparison
    static void benchmark_fused_vs_sequential(size_t batch_size, size_t seq_len, size_t embed_dim);
};

// Memory optimization utilities
class MemoryOptimizer {
public:
    // Memory pool for quantized tensors
    class QuantizedMemoryPool {
    public:
        QuantizedMemoryPool(size_t pool_size_mb);
        ~QuantizedMemoryPool();
        
        int8_t* allocate(size_t num_elements);
        void deallocate(int8_t* ptr);
        void clear();
        
        size_t get_allocated_bytes() const { return allocated_bytes_; }
        size_t get_peak_usage() const { return peak_usage_; }
        
    private:
        std::vector<uint8_t> memory_pool_;
        size_t pool_size_bytes_;
        size_t allocated_bytes_;
        size_t peak_usage_;
        std::vector<bool> allocation_map_;
    };
    
    // Memory layout optimization
    static void optimize_tensor_layout(std::vector<int8_t>& tensor, size_t access_pattern);
    static void cache_friendly_layout(std::vector<std::vector<int8_t>>& tensors);
    
    // Memory usage analysis
    static size_t estimate_memory_usage(size_t model_size, bool quantized_8bit, bool quantized_4bit);
    static void print_memory_breakdown(size_t model_size, bool use_sparse);
};

} // namespace Quantized
} // namespace ML

#endif // QUANTIZED_OPERATIONS_H
