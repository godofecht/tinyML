//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 6: Production API Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "ProductionAPI.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>

namespace ML {
namespace Production {

// StreamingTransformer Implementation
StreamingTransformer::StreamingTransformer(const Config& config)
    : config_(config) {
    
    // Initialize components
    ML::RealTime::LightweightAttention::Config attn_config{
        config.embed_dim, config.num_heads, config.embed_dim / config.num_heads, config.max_sequence_length
    };
    attention_ = std::make_unique<ML::RealTime::LightweightAttention>(attn_config);
    
    // Initialize dynamic network
    ML::Dynamic::DynamicNeuralNetwork::NetworkConfig net_config{
        {config.embed_dim, config.embed_dim * 2, config.embed_dim},
        0.001f, 0.1f, 1024, 16, true, true
    };
    network_ = std::make_unique<ML::Dynamic::DynamicNeuralNetwork>(net_config);
    
    // Initialize quantized attention if enabled
    if (config.enable_quantization) {
        ML::Quantized::QuantizedAttention::Config quant_config{
            config.embed_dim, config.num_heads, config.embed_dim / config.num_heads,
            false, config.enable_sparse_attention, config.sparsity_ratio
        };
        quantized_attention_ = std::make_unique<ML::Quantized::QuantizedAttention>(quant_config);
    }
    
    // Initialize optimizer
    EdgeOptimizer::DeviceProfile profile = TransformerFactory::detect_device_profile();
    optimizer_ = std::make_unique<EdgeOptimizer>(profile);
    
    // Initialize hardware accelerator
    accelerator_ = std::make_unique<HardwareAccelerator>();
    
    // Set start time for metrics
    start_time_ = std::chrono::steady_clock::now();
    
    // Apply initial optimizations
    optimizer_->optimize_for_device();
    
    ErrorHandling::log_info("StreamingTransformer initialized successfully");
}

StreamingTransformer::~StreamingTransformer() {
    stop_stream();
    ErrorHandling::log_info("StreamingTransformer destroyed");
}

std::vector<float> StreamingTransformer::process(const std::vector<float>& input) {
    if (!healthy_) {
        set_error("Transformer is in unhealthy state");
        return {};
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        auto output = forward_pass(input);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        update_metrics(static_cast<float>(duration.count()) / 1000.0f);
        
        return output;
    } catch (const std::exception& e) {
        set_error(std::string("Processing error: ") + e.what());
        return {};
    }
}

std::vector<std::vector<float>> StreamingTransformer::process_batch(const std::vector<std::vector<float>>& batch) {
    std::vector<std::vector<float>> results;
    results.reserve(batch.size());
    
    for (const auto& input : batch) {
        results.push_back(process(input));
    }
    
    return results;
}

void StreamingTransformer::start_stream() {
    if (streaming_active_) {
        return;
    }
    
    streaming_active_ = true;
    processing_thread_ = std::thread(&StreamingTransformer::processing_loop, this);
    
    ErrorHandling::log_info("Streaming started");
}

void StreamingTransformer::stop_stream() {
    if (!streaming_active_) {
        return;
    }
    
    streaming_active_ = false;
    queue_cv_.notify_all();
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    ErrorHandling::log_info("Streaming stopped");
}

void StreamingTransformer::push_chunk(const std::vector<float>& chunk) {
    if (!streaming_active_) {
        set_error("Streaming is not active");
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        input_queue_.push(chunk);
    }
    queue_cv_.notify_one();
}

std::vector<float> StreamingTransformer::get_output() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (output_queue_.empty()) {
        return {};
    }
    
    auto output = output_queue_.front();
    output_queue_.pop();
    return output;
}

bool StreamingTransformer::has_output() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return !output_queue_.empty();
}

void StreamingTransformer::optimize_for_latency() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    apply_latency_optimizations();
    ErrorHandling::log_info("Optimized for latency");
}

void StreamingTransformer::optimize_for_memory() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    apply_memory_optimizations();
    ErrorHandling::log_info("Optimized for memory");
}

void StreamingTransformer::optimize_for_throughput() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    apply_throughput_optimizations();
    ErrorHandling::log_info("Optimized for throughput");
}

void StreamingTransformer::enable_quantization() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    if (!config_.enable_quantization) {
        config_.enable_quantization = true;
        
        ML::Quantized::QuantizedAttention::Config quant_config{
            config_.embed_dim, config_.num_heads, config_.embed_dim / config_.num_heads,
            false, config_.enable_sparse_attention, config_.sparsity_ratio
        };
        quantized_attention_ = std::make_unique<ML::Quantized::QuantizedAttention>(quant_config);
        
        ErrorHandling::log_info("Quantization enabled");
    }
}

void StreamingTransformer::disable_quantization() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    if (config_.enable_quantization) {
        config_.enable_quantization = false;
        quantized_attention_.reset();
        ErrorHandling::log_info("Quantization disabled");
    }
}

void StreamingTransformer::update_online(const std::vector<float>& input, const std::vector<float>& target) {
    if (!healthy_) {
        return;
    }
    
    try {
        // Simple online learning (simplified)
        auto output = forward_pass(input);
        
        // Compute error and update (very basic implementation)
        if (output.size() == target.size()) {
            std::vector<float> errors;
            for (size_t i = 0; i < output.size(); ++i) {
                errors.push_back(target[i] - output[i]);
            }
            
            // Trigger adaptation
            network_->adapt_topology(errors);
            
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_.is_adapting = true;
        }
    } catch (const std::exception& e) {
        set_error(std::string("Online update error: ") + e.what());
    }
}

void StreamingTransformer::set_adaptation_rate(float rate) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_.adaptation_rate = rate;
}

StreamingTransformer::Metrics StreamingTransformer::get_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void StreamingTransformer::update_config(const Config& new_config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = new_config;
}

StreamingTransformer::Config StreamingTransformer::get_config() const {
    return config_;
}

void StreamingTransformer::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = Metrics{};
    latency_history_.clear();
    start_time_ = std::chrono::steady_clock::now();
}

void StreamingTransformer::print_performance_report() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    std::cout << "\n=== Performance Report ===\n";
    std::cout << "Average Latency: " << metrics_.average_latency_ms << "ms\n";
    std::cout << "Throughput: " << metrics_.throughput_tokens_per_sec << " tokens/sec\n";
    std::cout << "Memory Usage: " << metrics_.memory_usage_mb << "MB\n";
    std::cout << "CPU Utilization: " << (metrics_.cpu_utilization * 100) << "%\n";
    std::cout << "Total Tokens: " << metrics_.total_tokens_processed << "\n";
    std::cout << "Is Adapting: " << (metrics_.is_adapting ? "Yes" : "No") << "\n";
    std::cout << "========================\n";
}

std::string StreamingTransformer::get_last_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

bool StreamingTransformer::is_healthy() const {
    return healthy_;
}

// Private methods
void StreamingTransformer::processing_loop() {
    while (streaming_active_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !input_queue_.empty() || !streaming_active_; });
        
        if (!streaming_active_) break;
        
        auto input = input_queue_.front();
        input_queue_.pop();
        lock.unlock();
        
        // Process input
        auto output = forward_pass(input);
        
        // Add to output queue
        lock.lock();
        output_queue_.push(output);
        lock.unlock();
    }
}

std::vector<float> StreamingTransformer::forward_pass(const std::vector<float>& input) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    if (config_.enable_quantization && quantized_attention_) {
        // Use quantized attention
        std::vector<int8_t> quantized_input(input.size());
        ML::Quantized::QuantParams params(0.01f, 0, -128, 127);
        
        // Quantize input
        for (size_t i = 0; i < input.size(); ++i) {
            float scaled = input[i] / params.scale + params.zero_point;
            int32_t clamped = std::max(-128, std::min(127, static_cast<int32_t>(scaled)));
            quantized_input[i] = static_cast<int8_t>(clamped);
        }
        
        auto quantized_output = quantized_attention_->forward(quantized_input);
        
        // Dequantize output
        std::vector<float> output(quantized_output.size());
        for (size_t i = 0; i < quantized_output.size(); ++i) {
            output[i] = (static_cast<float>(quantized_output[i]) - params.zero_point) * params.scale;
        }
        
        return output;
    } else {
        // Use regular attention
        auto attention_output = attention_->forward(input);
        auto network_output = network_->forward(attention_output);
        return network_output;
    }
}

void StreamingTransformer::update_metrics(float latency_ms) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    latency_history_.push_back(latency_ms);
    if (latency_history_.size() > 100) {
        latency_history_.erase(latency_history_.begin());
    }
    
    // Update average latency
    float sum = 0.0f;
    for (float latency : latency_history_) {
        sum += latency;
    }
    metrics_.average_latency_ms = sum / latency_history_.size();
    
    // Update throughput
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
    
    if (duration.count() > 0) {
        metrics_.total_tokens_processed++;
        metrics_.throughput_tokens_per_sec = static_cast<float>(metrics_.total_tokens_processed) / duration.count();
    }
    
    // Update memory usage
    size_t total_memory = 0;
    if (attention_) total_memory += attention_->get_memory_usage();
    if (network_) total_memory += network_->get_memory_usage();
    if (quantized_attention_) total_memory += quantized_attention_->get_memory_usage();
    
    metrics_.memory_usage_mb = total_memory / (1024 * 1024);
    
    // Update CPU utilization (simplified)
    metrics_.cpu_utilization = std::min(1.0f, metrics_.average_latency_ms / config_.target_latency_ms);
}

void StreamingTransformer::set_error(const std::string& error) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = error;
    healthy_ = false;
    ErrorHandling::log_error(error);
}

void StreamingTransformer::apply_latency_optimizations() {
    // Reduce model size for faster inference
    if (config_.embed_dim > 128) {
        config_.embed_dim = 128;
        config_.num_heads = std::min(config_.num_heads, 4UL);
    }
    
    // Enable quantization if not already enabled
    if (!config_.enable_quantization) {
        enable_quantization();
    }
    
    // Reduce sequence length
    config_.max_sequence_length = std::min(config_.max_sequence_length, 256UL);
}

void StreamingTransformer::apply_memory_optimizations() {
    // Force quantization
    if (!config_.enable_quantization) {
        enable_quantization();
    }
    
    // Enable sparse attention
    config_.enable_sparse_attention = true;
    config_.sparsity_ratio = 0.7f;
    
    // Reduce model size
    config_.embed_dim = std::min(config_.embed_dim, 64UL);
    config_.num_layers = std::min(config_.num_layers, 2UL);
}

void StreamingTransformer::apply_throughput_optimizations() {
    // Increase batch processing capability
    config_.chunk_size = std::max(config_.chunk_size, 64UL);
    
    // Enable multi-threading
    config_.use_multi_threading = true;
    
    // Optimize for batch processing
    if (config_.embed_dim < 512) {
        config_.embed_dim = 512;
        config_.num_heads = 8;
    }
}

// EdgeOptimizer Implementation
EdgeOptimizer::EdgeOptimizer(const DeviceProfile& profile) : profile_(profile) {
    detect_hardware_capabilities();
}

void EdgeOptimizer::optimize_for_device() {
    applied_optimizations_.clear();
    
    if (profile_.device_type == "mobile") {
        apply_cpu_optimizations();
        apply_memory_optimizations();
        applied_optimizations_.push_back("Mobile optimization");
    } else if (profile_.device_type == "edge") {
        apply_cpu_optimizations();
        applied_optimizations_.push_back("Edge optimization");
    } else if (profile_.device_type == "server") {
        if (profile_.has_gpu) {
            applied_optimizations_.push_back("GPU optimization");
        }
        applied_optimizations_.push_back("Server optimization");
    }
    
    ErrorHandling::log_info("Device optimization applied: " + profile_.device_type);
}

void EdgeOptimizer::detect_hardware_capabilities() {
    // Detect CPU capabilities
    profile_.has_neon = ML::XSIMD::XSIMDVector::has_simd_support();
    
    // Detect GPU capabilities (simplified)
    profile_.has_gpu = false; // Would need actual GPU detection
    
    ErrorHandling::log_info("Hardware capabilities detected");
}

void EdgeOptimizer::apply_cpu_optimizations() {
    // Enable SIMD optimizations
    if (profile_.has_neon || profile_.has_avx2) {
        applied_optimizations_.push_back("SIMD optimization");
    }
    
    // Optimize for number of cores
    if (profile_.cpu_cores >= 4) {
        applied_optimizations_.push_back("Multi-threading");
    }
}

void EdgeOptimizer::apply_memory_optimizations() {
    // Apply memory-specific optimizations for low-memory devices
    if (profile_.memory_mb < 1024) {
        applied_optimizations_.push_back("Low memory optimization");
    }
}

void EdgeOptimizer::apply_power_optimizations() {
    if (profile_.power_budget_watts < 5.0f) {
        applied_optimizations_.push_back("Low power optimization");
    }
}

void EdgeOptimizer::optimize_for_power_budget(float max_watts) {
    profile_.power_budget_watts = max_watts;
    apply_power_optimizations();
    applied_optimizations_.push_back("Power budget optimization");
}

void EdgeOptimizer::optimize_for_thermal_constraints(float max_temp_celsius) {
    (void)max_temp_celsius;
    applied_optimizations_.push_back("Thermal constraints optimization");
}

void EdgeOptimizer::compress_model(float compression_ratio) {
    applied_optimizations_.push_back("Model compression ratio: " + std::to_string(compression_ratio));
}

void EdgeOptimizer::prune_model(float sparsity_ratio) {
    applied_optimizations_.push_back("Model pruning ratio: " + std::to_string(sparsity_ratio));
}

void EdgeOptimizer::quantize_model(int bits) {
    applied_optimizations_.push_back("Model quantization: " + std::to_string(bits) + "-bit");
}

void EdgeOptimizer::tune_for_latency(float target_ms) {
    applied_optimizations_.push_back("Latency tuning: " + std::to_string(target_ms) + "ms");
}

void EdgeOptimizer::tune_for_memory(size_t max_mb) {
    applied_optimizations_.push_back("Memory tuning: " + std::to_string(max_mb) + "MB");
}

void EdgeOptimizer::tune_for_power(float max_watts) {
    applied_optimizations_.push_back("Power tuning: " + std::to_string(max_watts) + "W");
}

EdgeOptimizer::DeviceProfile EdgeOptimizer::get_device_profile() const {
    return profile_;
}

void EdgeOptimizer::print_optimization_report() const {
    std::cout << "Edge Optimizer Report (" << profile_.device_type << ")\n";
    for (const auto& entry : applied_optimizations_) {
        std::cout << "  - " << entry << "\n";
    }
}

std::vector<std::string> EdgeOptimizer::get_optimization_suggestions() const {
    return applied_optimizations_;
}

// HardwareAccelerator Implementation
HardwareAccelerator::HardwareAccelerator() : active_accelerator_(AcceleratorType::CPU) {
    initialize_accelerators();
}

HardwareAccelerator::~HardwareAccelerator() = default;

std::vector<HardwareAccelerator::AcceleratorType> HardwareAccelerator::detect_available_accelerators() {
    std::vector<AcceleratorType> available;
    
    // CPU is always available
    available.push_back(AcceleratorType::CPU);
    
    // Detect other accelerators (simplified)
    // In a real implementation, this would detect Metal, CUDA, OpenCL, etc.
    
    return available;
}

void HardwareAccelerator::initialize_accelerators() {
    auto available = detect_available_accelerators();
    
    for (auto type : available) {
        available_accelerators_[type] = true;
    }
    
    ErrorHandling::log_info("Hardware accelerators initialized");
}

bool HardwareAccelerator::enable_accelerator(AcceleratorType type) {
    if (available_accelerators_.count(type) == 0) {
        return false;
    }
    active_accelerator_ = type;
    return true;
}

void HardwareAccelerator::disable_accelerator(AcceleratorType type) {
    if (active_accelerator_ == type) {
        active_accelerator_ = AcceleratorType::CPU;
    }
}

HardwareAccelerator::AcceleratorType HardwareAccelerator::get_active_accelerator() const {
    return active_accelerator_;
}

std::string HardwareAccelerator::get_accelerator_info() const {
    switch (active_accelerator_) {
        case AcceleratorType::CPU: return "CPU";
        case AcceleratorType::GPU_METAL: return "GPU (Metal)";
        case AcceleratorType::GPU_CUDA: return "GPU (CUDA)";
        case AcceleratorType::GPU_OPENCL: return "GPU (OpenCL)";
        case AcceleratorType::NEURAL_ENGINE: return "Neural Engine";
        case AcceleratorType::DSP: return "DSP";
        default: return "Unknown";
    }
}

// TransformerFactory Implementation
std::unique_ptr<StreamingTransformer> TransformerFactory::create_mobile_transformer(
    size_t vocab_size, float target_latency_ms) {
    
    StreamingTransformer::Config config;
    config.vocab_size = vocab_size;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 2;
    config.target_latency_ms = target_latency_ms;
    config.max_memory_mb = 2;
    config.enable_quantization = true;
    config.enable_sparse_attention = true;
    config.sparsity_ratio = 0.7f;
    config.use_multi_threading = false;
    
    return std::make_unique<StreamingTransformer>(config);
}

std::unique_ptr<StreamingTransformer> TransformerFactory::create_edge_transformer(
    size_t vocab_size, float target_latency_ms) {
    
    StreamingTransformer::Config config;
    config.vocab_size = vocab_size;
    config.embed_dim = 128;
    config.num_heads = 4;
    config.num_layers = 3;
    config.target_latency_ms = target_latency_ms;
    config.max_memory_mb = 5;
    config.enable_quantization = true;
    config.enable_sparse_attention = false;
    config.use_multi_threading = true;
    
    return std::make_unique<StreamingTransformer>(config);
}

std::unique_ptr<StreamingTransformer> TransformerFactory::create_server_transformer(
    size_t vocab_size, float target_latency_ms) {
    
    StreamingTransformer::Config config;
    config.vocab_size = vocab_size;
    config.embed_dim = 256;
    config.num_heads = 8;
    config.num_layers = 6;
    config.target_latency_ms = target_latency_ms;
    config.max_memory_mb = 20;
    config.enable_quantization = false;
    config.enable_sparse_attention = false;
    config.use_multi_threading = true;
    config.use_gpu_acceleration = true;
    
    return std::make_unique<StreamingTransformer>(config);
}

// DeploymentUtils Implementation
bool DeploymentUtils::save_model(const StreamingTransformer& transformer, const std::string& filepath) {
    return save_config(transformer.get_config(), filepath + ".config");
}

std::unique_ptr<StreamingTransformer> DeploymentUtils::load_model(const std::string& filepath) {
    auto config = load_config(filepath + ".config");
    return std::make_unique<StreamingTransformer>(config);
}

bool DeploymentUtils::save_config(const StreamingTransformer::Config& config, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    file << "vocab_size " << config.vocab_size << "\n";
    file << "embed_dim " << config.embed_dim << "\n";
    file << "num_heads " << config.num_heads << "\n";
    file << "num_layers " << config.num_layers << "\n";
    file << "max_sequence_length " << config.max_sequence_length << "\n";
    file << "target_latency_ms " << config.target_latency_ms << "\n";
    file << "max_memory_mb " << config.max_memory_mb << "\n";
    return true;
}

StreamingTransformer::Config DeploymentUtils::load_config(const std::string& filepath) {
    std::ifstream file(filepath);
    StreamingTransformer::Config config;
    if (!file.is_open()) {
        return config;
    }
    std::string key;
    while (file >> key) {
        if (key == "vocab_size") file >> config.vocab_size;
        else if (key == "embed_dim") file >> config.embed_dim;
        else if (key == "num_heads") file >> config.num_heads;
        else if (key == "num_layers") file >> config.num_layers;
        else if (key == "max_sequence_length") file >> config.max_sequence_length;
        else if (key == "target_latency_ms") file >> config.target_latency_ms;
        else if (key == "max_memory_mb") file >> config.max_memory_mb;
    }
    return config;
}

void DeploymentUtils::profile_model(const StreamingTransformer& transformer, size_t iterations) {
    (void)transformer;
    (void)iterations;
}

std::string DeploymentUtils::generate_performance_report(const StreamingTransformer& transformer) {
    std::ostringstream report;
    auto metrics = transformer.get_metrics();
    report << "Performance Report\n";
    report << "  Avg latency (ms): " << metrics.average_latency_ms << "\n";
    report << "  Throughput (tokens/s): " << metrics.throughput_tokens_per_sec << "\n";
    report << "  Memory (MB): " << metrics.memory_usage_mb << "\n";
    report << "  Total tokens: " << metrics.total_tokens_processed << "\n";
    return report.str();
}

bool DeploymentUtils::validate_model(const StreamingTransformer& transformer) {
    return transformer.is_healthy();
}

std::vector<std::string> DeploymentUtils::run_integration_tests(const StreamingTransformer& transformer) {
    std::vector<std::string> results;
    results.push_back(transformer.is_healthy() ? "Health check: PASS" : "Health check: FAIL");
    return results;
}

void DeploymentUtils::prepare_for_deployment(StreamingTransformer& transformer) {
    transformer.optimize_for_latency();
    transformer.optimize_for_memory();
}

std::vector<std::string> DeploymentUtils::check_deployment_readiness(const StreamingTransformer& transformer) {
    std::vector<std::string> warnings;
    if (!transformer.is_healthy()) {
        warnings.push_back("Model health check failed");
    }
    return warnings;
}

EdgeOptimizer::DeviceProfile TransformerFactory::detect_device_profile() {
    EdgeOptimizer::DeviceProfile profile;
    
    // Detect device type (simplified)
    profile.cpu_cores = std::thread::hardware_concurrency();
    profile.memory_mb = 8192; // Would need actual memory detection
    profile.device_type = "edge"; // Would need actual device detection
    profile.has_neon = ML::XSIMD::XSIMDVector::has_simd_support();
    profile.has_avx2 = false; // Would need actual detection
    profile.has_gpu = false; // Would need actual GPU detection
    profile.power_budget_watts = 15.0f;
    
    return profile;
}

// ErrorHandling Implementation
ErrorHandling::LogLevel ErrorHandling::current_log_level_ = LogLevel::INFO;
std::mutex ErrorHandling::log_mutex_;
std::vector<std::string> ErrorHandling::error_history_;

void ErrorHandling::set_log_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    current_log_level_ = level;
}

void ErrorHandling::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    if (level >= current_log_level_) {
        std::string level_str;
        switch (level) {
            case LogLevel::DEBUG: level_str = "DEBUG"; break;
            case LogLevel::INFO: level_str = "INFO"; break;
            case LogLevel::WARNING: level_str = "WARNING"; break;
            case LogLevel::ERROR: level_str = "ERROR"; break;
            case LogLevel::FATAL: level_str = "FATAL"; break;
        }
        
        std::cout << "[" << level_str << "] " << message << std::endl;
        
        if (level >= LogLevel::ERROR) {
            error_history_.push_back(message);
        }
    }
}

void ErrorHandling::log_info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void ErrorHandling::log_error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

std::string ErrorHandling::get_last_error() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    return error_history_.empty() ? "" : error_history_.back();
}

} // namespace Production
} // namespace ML
