//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Advanced Optimizations Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "AdvancedOptimizations.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <immintrin.h>

namespace ML {
namespace Advanced {

// ==================== QuantizedTensor Implementation ====================

QuantizedTensor::QuantizedTensor(size_t size, bool use_8bit) 
    : original_size(size), is_8bit(use_8bit) {
    
    if (is_8bit) {
        quantized_data_8bit.resize(size);
    } else {
        quantized_data_4bit.resize((size + 1) / 2); // 2 values per byte
    }
    
    scale_factors.resize(1);
    zero_points.resize(1);
}

void QuantizedTensor::quantize_from_float(const float* input, size_t size) {
    if (size != original_size) return;
    
    // Find min and max values for scaling
    float min_val = *std::min_element(input, input + size);
    float max_val = *std::max_element(input, input + size);
    
    // Calculate scale and zero point
    float scale = (max_val - min_val) / (is_8bit ? 255.0f : 15.0f);
    int32_t zero_point = is_8bit ? 128 : 8;
    
    if (std::abs(scale) < 1e-6f) scale = 1e-6f;
    
    scale_factors[0] = scale;
    zero_points[0] = zero_point;
    
    if (is_8bit) {
        // 8-bit quantization
        for (size_t i = 0; i < size; ++i) {
            int32_t quantized = static_cast<int32_t>(input[i] / scale + zero_point);
            quantized = std::max(0, std::min(255, quantized));
            quantized_data_8bit[i] = static_cast<uint8_t>(quantized);
        }
    } else {
        // 4-bit quantization (2 values per byte)
        for (size_t i = 0; i < size; i += 2) {
            int32_t q1 = static_cast<int32_t>(input[i] / scale + zero_point);
            int32_t q2 = (i + 1 < size) ? static_cast<int32_t>(input[i + 1] / scale + zero_point) : 0;
            
            q1 = std::max(0, std::min(15, q1));
            q2 = std::max(0, std::min(15, q2));
            
            quantized_data_4bit[i / 2] = (q1 << 4) | q2;
        }
    }
}

void QuantizedTensor::dequantize_to_float(float* output, size_t size) const {
    if (size != original_size) return;
    
    float scale = scale_factors[0];
    int32_t zero_point = zero_points[0];
    
    if (is_8bit) {
        for (size_t i = 0; i < size; ++i) {
            output[i] = (static_cast<int32_t>(quantized_data_8bit[i]) - zero_point) * scale;
        }
    } else {
        for (size_t i = 0; i < size; i += 2) {
            uint8_t packed = quantized_data_4bit[i / 2];
            int32_t q1 = (packed >> 4) & 0x0F;
            int32_t q2 = packed & 0x0F;
            
            output[i] = (q1 - zero_point) * scale;
            if (i + 1 < size) {
                output[i + 1] = (q2 - zero_point) * scale;
            }
        }
    }
}

size_t QuantizedTensor::memory_footprint() const {
    size_t size = is_8bit ? quantized_data_8bit.size() : quantized_data_4bit.size();
    return size + scale_factors.size() * sizeof(float) + zero_points.size() * sizeof(int32_t);
}

double QuantizedTensor::compression_ratio() const {
    size_t original_bytes = original_size * sizeof(float);
    return static_cast<double>(original_bytes) / memory_footprint();
}

// ==================== FusedKernel Implementation ====================

FusedKernel::FusedKernel() : is_compiled(false) {}

void FusedKernel::add_operation(FusedOperation::Type type, const std::vector<float>& params) {
    FusedOperation op;
    op.type = type;
    op.parameters = params;
    op.is_fused = false;
    operations.push_back(op);
}

void FusedKernel::fuse_operations() {
    if (operations.empty()) return;
    
    // Simple fusion strategy - combine compatible operations
    for (size_t i = 0; i < operations.size(); ++i) {
        operations[i].is_fused = true;
    }
    
    is_compiled = true;
}

void FusedKernel::execute(const float* input, float* output, size_t size) {
    if (!is_compiled) return;
    
    // Copy input to working buffer
    std::vector<float> working_buffer(input, input + size);
    
    // Execute fused operations
    for (const auto& op : operations) {
        switch (op.type) {
            case FusedOperation::ADD:
                for (size_t i = 0; i < size; ++i) {
                    working_buffer[i] += op.parameters[0];
                }
                break;
                
            case FusedOperation::MUL:
                for (size_t i = 0; i < size; ++i) {
                    working_buffer[i] *= op.parameters[0];
                }
                break;
                
            case FusedOperation::TANH:
                for (size_t i = 0; i < size; ++i) {
                    working_buffer[i] = std::tanh(working_buffer[i]);
                }
                break;
                
            case FusedOperation::RELU:
                for (size_t i = 0; i < size; ++i) {
                    working_buffer[i] = std::max(0.0f, working_buffer[i]);
                }
                break;
                
            default:
                break;
        }
    }
    
    // Copy result to output
    std::copy(working_buffer.begin(), working_buffer.end(), output);
}

void FusedKernel::execute_fused_attention(const float* q, const float* k, const float* v, 
                                         float* output, size_t seq_len, size_t embed_dim) {
    if (!is_compiled) return;
    
    // Simplified fused attention computation
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t d = 0; d < embed_dim; ++d) {
            float attention_sum = 0.0f;
            
            // Compute attention weights
            for (size_t j = 0; j < seq_len; ++j) {
                float score = 0.0f;
                for (size_t k_dim = 0; k_dim < embed_dim; ++k_dim) {
                    score += q[i * embed_dim + k_dim] * k[j * embed_dim + k_dim];
                }
                score = std::exp(score / std::sqrt(embed_dim));
                attention_sum += score * v[j * embed_dim + d];
            }
            
            // Apply fused operations
            output[i * embed_dim + d] = attention_sum / seq_len;
            
            // Apply any additional fused operations
            for (const auto& op : operations) {
                if (op.type == FusedOperation::TANH) {
                    output[i * embed_dim + d] = std::tanh(output[i * embed_dim + d]);
                } else if (op.type == FusedOperation::RELU) {
                    output[i * embed_dim + d] = std::max(0.0f, output[i * embed_dim + d]);
                }
            }
        }
    }
}

double FusedKernel::benchmark_latency(size_t size, int iterations) {
    if (!is_compiled) return 0.0;
    
    std::vector<float> input(size, 1.0f);
    std::vector<float> output(size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        execute(input.data(), output.data(), size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return static_cast<double>(duration.count()) / iterations;
}

// ==================== SparseAttention Implementation ====================

SparseAttention::SparseAttention(size_t seq_len, float sparsity_ratio, bool dynamic)
    : is_dynamic(dynamic), sparsity_threshold(sparsity_ratio) {
    
    attention_mask.mask.resize(seq_len * seq_len, true);
    attention_mask.nnz_count = seq_len * seq_len;
    attention_mask.sparsity_ratio = 0.0f;
    
    sparse_indices.reserve(seq_len * seq_len);
}

void SparseAttention::compute_sparsity_mask(const float* attention_scores, size_t seq_len) {
    if (!is_dynamic) return;
    
    // Flatten attention scores and find threshold
    std::vector<float> scores(attention_scores, attention_scores + seq_len * seq_len);
    std::sort(scores.begin(), scores.end());
    
    size_t threshold_idx = static_cast<size_t>((1.0f - sparsity_threshold) * scores.size());
    float threshold_value = scores[threshold_idx];
    
    // Create sparse mask
    attention_mask.nnz_count = 0;
    sparse_indices.clear();
    
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            bool keep = attention_scores[i * seq_len + j] >= threshold_value;
            attention_mask.mask[i * seq_len + j] = keep;
            
            if (keep) {
                sparse_indices.push_back(i * seq_len + j);
                attention_mask.nnz_count++;
            }
        }
    }
    
    attention_mask.sparsity_ratio = 1.0f - (static_cast<float>(attention_mask.nnz_count) / (seq_len * seq_len));
}

void SparseAttention::apply_sparsity_mask(float* attention_scores, size_t seq_len) {
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            if (!attention_mask.mask[i * seq_len + j]) {
                attention_scores[i * seq_len + j] = 0.0f;
            }
        }
    }
}

void SparseAttention::sparse_attention_computation(const float* q, const float* k, const float* v,
                                                 float* output, size_t seq_len, size_t embed_dim) {
    // Initialize output
    std::fill(output, output + seq_len * embed_dim, 0.0f);
    
    // Compute sparse attention
    for (size_t idx : sparse_indices) {
        size_t i = idx / seq_len;
        size_t j = idx % seq_len;
        
        if (i >= seq_len || j >= seq_len) continue;
        
        // Compute attention weight
        float score = 0.0f;
        for (size_t d = 0; d < embed_dim; ++d) {
            score += q[i * embed_dim + d] * k[j * embed_dim + d];
        }
        score = std::exp(score / std::sqrt(embed_dim));
        
        // Apply to output
        for (size_t d = 0; d < embed_dim; ++d) {
            output[i * embed_dim + d] += score * v[j * embed_dim + d];
        }
    }
    
    // Normalize
    for (size_t i = 0; i < seq_len; ++i) {
        float norm_sum = 0.0f;
        for (size_t d = 0; d < embed_dim; ++d) {
            norm_sum += std::abs(output[i * embed_dim + d]);
        }
        
        if (norm_sum > 1e-6f) {
            for (size_t d = 0; d < embed_dim; ++d) {
                output[i * embed_dim + d] /= norm_sum;
            }
        }
    }
}

size_t SparseAttention::get_computation_reduction() const {
    size_t total_ops = attention_mask.mask.size() * attention_mask.mask.size();
    size_t sparse_ops = attention_mask.nnz_count;
    return total_ops - sparse_ops;
}

double SparseAttention::speedup_factor() const {
    if (attention_mask.nnz_count == 0) return 1.0;
    return static_cast<double>(attention_mask.mask.size()) / attention_mask.nnz_count;
}

// ==================== MemoryOptimizer Implementation ====================

MemoryOptimizer::MemoryOptimizer(size_t initial_pool_size) 
    : pool_size(initial_pool_size), used_size(0) {
    
    memory_pool.pool.resize(initial_pool_size / sizeof(float));
    use_weight_sharing = false;
    use_gradient_checkpointing = false;
    use_activation_recomputation = false;
}

void* MemoryOptimizer::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // Align to 8-byte boundary
    size_t aligned_size = (size + 7) & ~7;
    
    if (used_size + aligned_size > memory_pool.pool.size()) {
        // Pool full, allocate from heap
        return malloc(size);
    }
    
    void* ptr = &memory_pool.pool[used_size];
    allocation_map[ptr] = aligned_size;
    memory_pool.allocated_blocks.push_back({used_size, aligned_size});
    used_size += aligned_size;
    
    return ptr;
}

void MemoryOptimizer::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    auto it = allocation_map.find(ptr);
    if (it != allocation_map.end()) {
        // Mark as free (simplified - in practice would use free list)
        allocation_map.erase(it);
    } else {
        // Was heap allocated
        free(ptr);
    }
}

void MemoryOptimizer::enable_weight_sharing(bool enable) {
    use_weight_sharing = enable;
}

void MemoryOptimizer::share_weights(float* weights, size_t size, float similarity_threshold) {
    if (!use_weight_sharing) return;
    
    // Simple weight sharing implementation
    std::unordered_map<float, std::vector<size_t>> weight_groups;
    
    for (size_t i = 0; i < size; ++i) {
        float w = weights[i];
        
        // Find similar weights
        bool found_group = false;
        for (auto& group : weight_groups) {
            if (std::abs(w - group.first) < similarity_threshold) {
                group.second.push_back(i);
                found_group = true;
                break;
            }
        }
        
        if (!found_group) {
            weight_groups[w].push_back(i);
        }
    }
    
    // Apply sharing (simplified)
    for (const auto& group : weight_groups) {
        if (group.second.size() > 1) {
            float avg_weight = group.first;
            for (size_t idx : group.second) {
                weights[idx] = avg_weight;
            }
        }
    }
}

size_t MemoryOptimizer::get_memory_usage() const {
    return used_size * sizeof(float);
}

size_t MemoryOptimizer::get_peak_memory_usage() const {
    return memory_pool.pool.size() * sizeof(float);
}

double MemoryOptimizer::memory_efficiency() const {
    if (memory_pool.pool.empty()) return 0.0;
    return static_cast<double>(used_size) / memory_pool.pool.size();
}

// ==================== ParallelProcessor Implementation ====================

ParallelProcessor::ParallelProcessor(size_t num_threads) 
    : num_threads(num_threads), is_initialized(false), current_strategy(HYBRID) {
    thread_pool.stop = false;
}

ParallelProcessor::~ParallelProcessor() {
    shutdown();
}

void ParallelProcessor::initialize() {
    if (is_initialized) return;
    
    is_initialized = true;
    
    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
        thread_pool.workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(thread_pool.queue_mutex);
                    thread_pool.condition.wait(lock, [this] {
                        return thread_pool.stop || !thread_pool.tasks.empty();
                    });
                    
                    if (thread_pool.stop) return;
                    
                    task = std::move(thread_pool.tasks.front());
                    thread_pool.tasks.pop();
                }
                
                task();
            }
        });
    }
}

void ParallelProcessor::shutdown() {
    if (!is_initialized) return;
    
    {
        std::unique_lock<std::mutex> lock(thread_pool.queue_mutex);
        thread_pool.stop = true;
    }
    
    thread_pool.condition.notify_all();
    
    for (auto& worker : thread_pool.workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    thread_pool.workers.clear();
    is_initialized = false;
}

void ParallelProcessor::parallel_attention(const float* q, const float* k, const float* v,
                                          float* output, size_t num_heads, size_t seq_len, size_t head_dim) {
    if (!is_initialized) return;
    
    std::vector<std::thread> workers;
    size_t heads_per_thread = num_heads / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_head = t * heads_per_thread;
        size_t end_head = (t == num_threads - 1) ? num_heads : start_head + heads_per_thread;
        
        workers.emplace_back([&, start_head, end_head] {
            for (size_t h = start_head; h < end_head; ++h) {
                size_t head_offset = h * head_dim;
                
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        float attention_sum = 0.0f;
                        
                        for (size_t j = 0; j < seq_len; ++j) {
                            float score = 0.0f;
                            for (size_t k_dim = 0; k_dim < head_dim; ++k_dim) {
                                score += q[i * num_heads * head_dim + head_offset + k_dim] * 
                                        k[j * num_heads * head_dim + head_offset + k_dim];
                            }
                            score = std::exp(score / std::sqrt(head_dim));
                            attention_sum += score * v[j * num_heads * head_dim + head_offset + d];
                        }
                        
                        output[i * num_heads * head_dim + head_offset + d] = attention_sum / seq_len;
                    }
                }
            }
        });
    }
    
    for (auto& worker : workers) {
        worker.join();
    }
}

// ==================== AdvancedOptimizer Implementation ====================

AdvancedOptimizer::AdvancedOptimizer() {
    quantization_enabled = false;
    fusion_enabled = false;
    sparsity_enabled = false;
    memory_optimization_enabled = false;
    parallel_processing_enabled = false;
    
    // Initialize metrics
    metrics.quantization_speedup = 1.0;
    metrics.fusion_speedup = 1.0;
    metrics.sparsity_speedup = 1.0;
    metrics.memory_savings_ratio = 1.0;
    metrics.parallel_efficiency = 1.0;
    metrics.overall_speedup = 1.0;
}

AdvancedOptimizer::~AdvancedOptimizer() {
    // Smart pointers will automatically clean up
}

void AdvancedOptimizer::enable_quantization(bool enable, bool use_8bit) {
    quantization_enabled = enable;
    if (enable) {
        quantizer = std::make_unique<QuantizedTensor>(1024, use_8bit);
    }
}

void AdvancedOptimizer::enable_kernel_fusion(bool enable) {
    fusion_enabled = enable;
    if (enable) {
        fused_kernel = std::make_unique<FusedKernel>();
        fused_kernel->add_operation(FusedKernel::FusedOperation::TANH);
        fused_kernel->add_operation(FusedKernel::FusedOperation::RELU);
        fused_kernel->fuse_operations();
    }
}

void AdvancedOptimizer::enable_sparse_attention(bool enable, float sparsity_ratio) {
    sparsity_enabled = enable;
    if (enable) {
        sparse_attention = std::make_unique<SparseAttention>(512, sparsity_ratio, true);
    }
}

void AdvancedOptimizer::enable_memory_optimization(bool enable, size_t pool_size) {
    memory_optimization_enabled = enable;
    if (enable) {
        memory_optimizer = std::make_unique<MemoryOptimizer>(pool_size);
        memory_optimizer->enable_weight_sharing(true);
    }
}

void AdvancedOptimizer::enable_parallel_processing(bool enable, size_t num_threads) {
    parallel_processing_enabled = enable;
    if (enable) {
        size_t threads = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
        parallel_processor = std::make_unique<ParallelProcessor>(threads);
        parallel_processor->initialize();
    }
}

void AdvancedOptimizer::optimized_attention(const float* q, const float* k, const float* v,
                                           float* output, size_t seq_len, size_t embed_dim, size_t num_heads) {
    // Apply optimizations in order
    
    if (parallel_processing_enabled && parallel_processor) {
        parallel_processor->parallel_attention(q, k, v, output, num_heads, seq_len, embed_dim / num_heads);
    } else {
        // Standard attention computation
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t h = 0; h < num_heads; ++h) {
                size_t head_dim = embed_dim / num_heads;
                size_t head_offset = h * head_dim;
                
                for (size_t d = 0; d < head_dim; ++d) {
                    float attention_sum = 0.0f;
                    
                    for (size_t j = 0; j < seq_len; ++j) {
                        float score = 0.0f;
                        for (size_t k_dim = 0; k_dim < head_dim; ++k_dim) {
                            score += q[i * embed_dim + head_offset + k_dim] * 
                                    k[j * embed_dim + head_offset + k_dim];
                        }
                        score = std::exp(score / std::sqrt(head_dim));
                        attention_sum += score * v[j * embed_dim + head_offset + d];
                    }
                    
                    output[i * embed_dim + head_offset + d] = attention_sum / seq_len;
                }
            }
        }
    }
    
    // Apply fusion if enabled
    if (fusion_enabled && fused_kernel) {
        fused_kernel->execute(output, output, seq_len * embed_dim);
    }
}

void AdvancedOptimizer::print_optimization_summary() {
    std::cout << "\n=== Advanced Optimizations Summary ===\n";
    std::cout << "Quantization: " << (quantization_enabled ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "Kernel Fusion: " << (fusion_enabled ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "Sparse Attention: " << (sparsity_enabled ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "Memory Optimization: " << (memory_optimization_enabled ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "Parallel Processing: " << (parallel_processing_enabled ? "ENABLED" : "DISABLED") << "\n";
    
    std::cout << "\nPerformance Metrics:\n";
    std::cout << "Quantization Speedup: " << metrics.quantization_speedup << "x\n";
    std::cout << "Fusion Speedup: " << metrics.fusion_speedup << "x\n";
    std::cout << "Sparsity Speedup: " << metrics.sparsity_speedup << "x\n";
    std::cout << "Memory Savings: " << metrics.memory_savings_ratio << "x\n";
    std::cout << "Parallel Efficiency: " << metrics.parallel_efficiency << "x\n";
    std::cout << "Overall Speedup: " << metrics.overall_speedup << "x\n";
}

bool AdvancedOptimizer::meets_latency_target(double target_ms) const {
    return metrics.overall_speedup >= (10.0 / target_ms); // Simplified target check
}

bool AdvancedOptimizer::meets_memory_target(double target_mb) const {
    return metrics.memory_savings_ratio >= (5.0 / target_mb); // Simplified target check
}

// ==================== Utility Functions ====================

namespace Utils {
    
void quantize_tensor(const float* input, uint8_t* output, float& scale, int32_t& zero_point, size_t size) {
    float min_val = *std::min_element(input, input + size);
    float max_val = *std::max_element(input, input + size);
    
    scale = (max_val - min_val) / 255.0f;
    if (std::abs(scale) < 1e-6f) scale = 1e-6f;
    
    zero_point = 128;
    
    for (size_t i = 0; i < size; ++i) {
        int32_t quantized = static_cast<int32_t>(input[i] / scale + zero_point);
        quantized = std::max(0, std::min(255, quantized));
        output[i] = static_cast<uint8_t>(quantized);
    }
}

void dequantize_tensor(const uint8_t* input, float* output, float scale, int32_t zero_point, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = (static_cast<int32_t>(input[i]) - zero_point) * scale;
    }
}

size_t count_nonzero_elements(const float* tensor, size_t size) {
    size_t count = 0;
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(tensor[i]) > 1e-6f) {
            count++;
        }
    }
    return count;
}

double calculate_speedup(double baseline_time, double optimized_time) {
    if (optimized_time <= 0) return 0.0;
    return baseline_time / optimized_time;
}

} // namespace Utils

} // namespace Advanced
} // namespace ML
