//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef TINYML_API_H
#define TINYML_API_H

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <chrono>

namespace TinyML {

// Forward declarations
class RealTimeTransformer;
class DynamicNeuralNetwork;
class LightweightAttention;

// Configuration structures
struct ModelConfig {
    size_t embed_dim = 256;
    size_t num_heads = 8;
    size_t num_layers = 2;
    size_t sequence_length = 512;
    float dropout_rate = 0.1f;
    bool use_quantization = false;
    bool use_sparse_attention = false;
    std::string device = "cpu"; // "cpu", "gpu", "metal"
};

struct OptimizationConfig {
    bool optimize_for_latency = true;
    bool optimize_for_memory = false;
    bool enable_parallel_processing = false;
    size_t num_threads = 1;
    float target_latency_ms = 10.0f;
    size_t max_memory_mb = 10;
};

struct DeploymentConfig {
    std::string target_platform = "edge"; // "edge", "mobile", "server"
    bool enable_hardware_acceleration = false;
    bool enable_federated_learning = false;
    std::string model_format = "native"; // "native", "onnx", "tflite"
};

// Results and metrics
struct InferenceResult {
    std::vector<float> output;
    double latency_ms;
    size_t memory_used_mb;
    bool success;
    std::string error_message;
};

struct PerformanceMetrics {
    double avg_latency_ms;
    double throughput_tokens_per_sec;
    size_t memory_footprint_mb;
    double cpu_utilization_percent;
    size_t total_inferences;
    std::chrono::time_point<std::chrono::steady_clock> last_updated;
};

// Main API class
class TinyMLAPI {
public:
    TinyMLAPI();
    ~TinyMLAPI();

    // Model management
    bool create_model(const ModelConfig& config);
    bool load_model(const std::string& model_path);
    bool save_model(const std::string& model_path);
    void destroy_model();

    // Inference interface
    InferenceResult predict(const std::vector<float>& input);
    InferenceResult predict_batch(const std::vector<std::vector<float>>& inputs);
    
    // Streaming interface for real-time processing
    bool start_streaming();
    InferenceResult process_stream_chunk(const std::vector<float>& chunk);
    void end_streaming();

    // Optimization and tuning
    bool optimize_model(const OptimizationConfig& config);
    bool tune_for_hardware();
    PerformanceMetrics get_performance_metrics() const;

    // Deployment and export
    bool deploy_model(const DeploymentConfig& config);
    bool export_model(const std::string& format, const std::string& output_path);
    std::vector<std::string> get_supported_formats() const;

    // Federated learning
    bool enable_federated_learning(const std::string& server_url);
    bool update_model_federated(const std::vector<float>& local_gradients);
    bool sync_with_server();

    // Hardware acceleration
    bool enable_gpu_acceleration();
    bool enable_metal_acceleration();
    bool is_hardware_acceleration_available() const;

    // Monitoring and diagnostics
    std::string get_model_info() const;
    std::string get_system_info() const;
    bool run_self_test();
    void reset_performance_counters();

    // Configuration
    void set_model_config(const ModelConfig& config);
    ModelConfig get_model_config() const;
    void set_optimization_config(const OptimizationConfig& config);
    OptimizationConfig get_optimization_config() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Factory functions
std::unique_ptr<TinyMLAPI> create_tinyml_api();
std::unique_ptr<TinyMLAPI> create_tinyml_api_for_edge();
std::unique_ptr<TinyMLAPI> create_tinyml_api_for_mobile();
std::unique_ptr<TinyMLAPI> create_tinyml_api_for_server();

// Utility functions
namespace Utils {
    std::vector<ModelConfig> get_predefined_configs();
    bool validate_config(const ModelConfig& config);
    std::string config_to_string(const ModelConfig& config);
    ModelConfig config_from_string(const std::string& config_str);
    
    // Performance profiling
    class PerformanceProfiler {
    public:
        void start_profiling();
        void end_profiling();
        PerformanceMetrics get_metrics() const;
        void reset();
    private:
        std::chrono::time_point<std::chrono::steady_clock> start_time_;
        PerformanceMetrics metrics_;
    };
}

// Error handling
enum class ErrorCode {
    SUCCESS = 0,
    MODEL_NOT_LOADED,
    INVALID_INPUT,
    CONFIGURATION_ERROR,
    HARDWARE_ERROR,
    MEMORY_ERROR,
    NETWORK_ERROR,
    UNKNOWN_ERROR
};

class TinyMLException : public std::exception {
public:
    TinyMLException(ErrorCode code, const std::string& message);
    const char* what() const noexcept override;
    ErrorCode get_error_code() const noexcept;
    
private:
    ErrorCode code_;
    std::string message_;
};

// Callback types for async operations
using InferenceCallback = std::function<void(const InferenceResult&)>;
using TrainingCallback = std::function<void(float loss, int epoch)>;

// Async API
class AsyncTinyMLAPI : public TinyMLAPI {
public:
    void predict_async(const std::vector<float>& input, InferenceCallback callback);
    void predict_batch_async(const std::vector<std::vector<float>>& inputs, InferenceCallback callback);
    
    // Async training for federated learning
    void train_async(const std::vector<std::vector<float>>& training_data, 
                    const std::vector<std::vector<float>>& targets,
                    TrainingCallback callback);
};

// Constants and limits
namespace Constants {
    constexpr size_t MAX_SEQUENCE_LENGTH = 2048;
    constexpr size_t MAX_EMBED_DIM = 1024;
    constexpr size_t MAX_NUM_HEADS = 32;
    constexpr size_t MAX_NUM_LAYERS = 12;
    constexpr double MAX_LATENCY_MS = 1000.0;
    constexpr size_t MAX_MEMORY_MB = 1024;
}

} // namespace TinyML

#endif // TINYML_API_H
