//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 6: Production API - Clean Public Interface
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef PRODUCTION_API_H
#define PRODUCTION_API_H

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <queue>

#include "XSIMDOperations.h"
#include "LightweightAttention.h"
#include "DynamicNeuralNetwork.h"
#include "QuantizedOperations.h"

namespace ML {
namespace Production {

// Forward declarations
class StreamingTransformer;
class EdgeOptimizer;
class HardwareAccelerator;

/**
 * @brief Production-ready streaming transformer interface
 * 
 * This class provides a clean, production-ready API for real-time
 * transformer inference with support for streaming, optimization,
 * and hardware acceleration.
 */
class StreamingTransformer {
public:
    struct Config {
        // Model configuration
        size_t vocab_size = 1000;
        size_t embed_dim = 256;
        size_t num_heads = 8;
        size_t num_layers = 4;
        size_t max_sequence_length = 512;
        
        // Performance configuration
        float target_latency_ms = 5.0f;
        size_t max_memory_mb = 10;
        bool enable_quantization = false;
        bool enable_sparse_attention = false;
        float sparsity_ratio = 0.5f;
        
        // Hardware configuration
        bool use_gpu_acceleration = false;
        bool use_multi_threading = true;
        size_t num_threads = std::thread::hardware_concurrency();
        
        // Streaming configuration
        bool enable_streaming = true;
        size_t chunk_size = 32;
        float adaptation_rate = 0.01f;
    };
    
    struct Metrics {
        float average_latency_ms = 0.0f;
        float throughput_tokens_per_sec = 0.0f;
        size_t memory_usage_mb = 0;
        float cpu_utilization = 0.0f;
        size_t total_tokens_processed = 0;
        bool is_adapting = false;
    };
    
    StreamingTransformer(const Config& config);
    ~StreamingTransformer();
    
    // Core inference methods
    std::vector<float> process(const std::vector<float>& input);
    std::vector<std::vector<float>> process_batch(const std::vector<std::vector<float>>& batch);
    
    // Streaming interface
    void start_stream();
    void stop_stream();
    void push_chunk(const std::vector<float>& chunk);
    std::vector<float> get_output();
    bool has_output() const;
    
    // Optimization methods
    void optimize_for_latency();
    void optimize_for_memory();
    void optimize_for_throughput();
    void enable_quantization();
    void disable_quantization();
    
    // Adaptive learning
    void update_online(const std::vector<float>& input, const std::vector<float>& target);
    void set_adaptation_rate(float rate);
    
    // Hardware acceleration
    void enable_gpu_acceleration();
    void disable_gpu_acceleration();
    void set_num_threads(size_t num_threads);
    
    // Monitoring and metrics
    Metrics get_metrics() const;
    void reset_metrics();
    void print_performance_report() const;
    
    // Configuration management
    void update_config(const Config& new_config);
    Config get_config() const;
    
    // Error handling
    std::string get_last_error() const;
    bool is_healthy() const;
    
private:
    Config config_;
    mutable std::mutex config_mutex_;
    
    // Internal components
    std::unique_ptr<ML::RealTime::LightweightAttention> attention_;
    std::unique_ptr<ML::Dynamic::DynamicNeuralNetwork> network_;
    std::unique_ptr<ML::Quantized::QuantizedAttention> quantized_attention_;
    std::unique_ptr<EdgeOptimizer> optimizer_;
    std::unique_ptr<HardwareAccelerator> accelerator_;
    
    // Streaming state
    std::queue<std::vector<float>> input_queue_;
    std::queue<std::vector<float>> output_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread processing_thread_;
    std::atomic<bool> streaming_active_{false};
    
    // Metrics and monitoring
    mutable std::mutex metrics_mutex_;
    Metrics metrics_;
    std::vector<float> latency_history_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Error handling
    mutable std::mutex error_mutex_;
    std::string last_error_;
    std::atomic<bool> healthy_{true};
    
    // Internal methods
    void processing_loop();
    std::vector<float> forward_pass(const std::vector<float>& input);
    void update_metrics(float latency_ms);
    void set_error(const std::string& error);
    
    // Optimization helpers
    void apply_latency_optimizations();
    void apply_memory_optimizations();
    void apply_throughput_optimizations();
};

/**
 * @brief Edge device optimization utilities
 */
class EdgeOptimizer {
public:
    struct DeviceProfile {
        std::string device_type; // "mobile", "edge", "server"
        size_t cpu_cores;
        size_t memory_mb;
        bool has_gpu;
        bool has_neon;
        bool has_avx2;
        float power_budget_watts;
    };
    
    EdgeOptimizer(const DeviceProfile& profile);
    ~EdgeOptimizer() = default;
    
    // Device-specific optimizations
    void optimize_for_device();
    void optimize_for_power_budget(float max_watts);
    void optimize_for_thermal_constraints(float max_temp_celsius);
    
    // Model compression
    void compress_model(float compression_ratio);
    void prune_model(float sparsity_ratio);
    void quantize_model(int bits);
    
    // Performance tuning
    void tune_for_latency(float target_ms);
    void tune_for_memory(size_t max_mb);
    void tune_for_power(float max_watts);
    
    // Analysis and reporting
    DeviceProfile get_device_profile() const;
    void print_optimization_report() const;
    std::vector<std::string> get_optimization_suggestions() const;
    
private:
    DeviceProfile profile_;
    std::vector<std::string> applied_optimizations_;
    
    void detect_hardware_capabilities();
    void apply_cpu_optimizations();
    void apply_memory_optimizations();
    void apply_power_optimizations();
};

/**
 * @brief Hardware acceleration interface
 */
class HardwareAccelerator {
public:
    enum class AcceleratorType {
        CPU,
        GPU_METAL,
        GPU_CUDA,
        GPU_OPENCL,
        NEURAL_ENGINE,
        DSP
    };
    
    HardwareAccelerator();
    ~HardwareAccelerator();
    
    // Accelerator detection and management
    std::vector<AcceleratorType> detect_available_accelerators();
    bool enable_accelerator(AcceleratorType type);
    void disable_accelerator(AcceleratorType type);
    AcceleratorType get_active_accelerator() const;
    
    // Accelerated operations
    std::vector<float> accelerated_matmul(
        const std::vector<float>& a, const std::vector<float>& b, 
        size_t m, size_t n, size_t k);
    
    std::vector<float> accelerated_attention(
        const std::vector<float>& q, const std::vector<float>& k, 
        const std::vector<float>& v, size_t seq_len, size_t head_dim);
    
    // Performance monitoring
    float get_accelerator_utilization() const;
    size_t get_accelerator_memory_usage() const;
    std::string get_accelerator_info() const;
    
private:
    AcceleratorType active_accelerator_;
    std::map<AcceleratorType, bool> available_accelerators_;
    
    void initialize_accelerators();
    void setup_metal_accelerator();
    void setup_cuda_accelerator();
    void setup_opencl_accelerator();
};

/**
 * @brief Factory class for creating optimized transformer instances
 */
class TransformerFactory {
public:
    // Pre-configured factory methods
    static std::unique_ptr<StreamingTransformer> create_mobile_transformer(
        size_t vocab_size = 1000, float target_latency_ms = 10.0f);
    
    static std::unique_ptr<StreamingTransformer> create_edge_transformer(
        size_t vocab_size = 1000, float target_latency_ms = 5.0f);
    
    static std::unique_ptr<StreamingTransformer> create_server_transformer(
        size_t vocab_size = 1000, float target_latency_ms = 1.0f);
    
    // Custom configuration
    static std::unique_ptr<StreamingTransformer> create_custom_transformer(
        const StreamingTransformer::Config& config);
    
    // Device-specific optimization
    static std::unique_ptr<StreamingTransformer> create_optimized_for_device(
        const std::string& device_type, size_t vocab_size = 1000);
    
    // Benchmarking and profiling
    static std::vector<std::string> benchmark_all_configurations();
    static std::string recommend_configuration(const std::string& use_case);

    static EdgeOptimizer::DeviceProfile detect_device_profile();

private:
    TransformerFactory() = default;
    static StreamingTransformer::Config create_config_for_profile(
        const EdgeOptimizer::DeviceProfile& profile, size_t vocab_size);
};

/**
 * @brief Utilities for deployment and integration
 */
class DeploymentUtils {
public:
    // Model serialization
    static bool save_model(const StreamingTransformer& transformer, const std::string& filepath);
    static std::unique_ptr<StreamingTransformer> load_model(const std::string& filepath);
    
    // Configuration export/import
    static bool save_config(const StreamingTransformer::Config& config, const std::string& filepath);
    static StreamingTransformer::Config load_config(const std::string& filepath);
    
    // Performance profiling
    static void profile_model(const StreamingTransformer& transformer, size_t iterations = 1000);
    static std::string generate_performance_report(const StreamingTransformer& transformer);
    
    // Validation and testing
    static bool validate_model(const StreamingTransformer& transformer);
    static std::vector<std::string> run_integration_tests(const StreamingTransformer& transformer);
    
    // Deployment preparation
    static void prepare_for_deployment(StreamingTransformer& transformer);
    static std::vector<std::string> check_deployment_readiness(const StreamingTransformer& transformer);
    
private:
    DeploymentUtils() = default;
};

/**
 * @brief Error handling and logging utilities
 */
class ErrorHandling {
public:
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL
    };
    
    static void set_log_level(LogLevel level);
    static void log(LogLevel level, const std::string& message);
    static void log_debug(const std::string& message);
    static void log_info(const std::string& message);
    static void log_warning(const std::string& message);
    static void log_error(const std::string& message);
    static void log_fatal(const std::string& message);
    
    static std::string get_last_error();
    static void clear_errors();
    static std::vector<std::string> get_error_history();
    
private:
    static LogLevel current_log_level_;
    static std::mutex log_mutex_;
    static std::vector<std::string> error_history_;
};

} // namespace Production
} // namespace ML

#endif // PRODUCTION_API_H
