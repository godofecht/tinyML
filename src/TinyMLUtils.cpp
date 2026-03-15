//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "TinyMLAPI.h"
#include <sstream>
#include <algorithm>
#include <chrono>

namespace TinyML {
namespace Utils {

std::vector<ModelConfig> get_predefined_configs() {
    std::vector<ModelConfig> configs;
    
    // Mobile configuration
    ModelConfig mobile;
    mobile.embed_dim = 64;
    mobile.num_heads = 2;
    mobile.num_layers = 1;
    mobile.sequence_length = 128;
    mobile.dropout_rate = 0.1f;
    mobile.device = "cpu";
    configs.push_back(mobile);
    
    // Edge configuration
    ModelConfig edge;
    edge.embed_dim = 128;
    edge.num_heads = 4;
    edge.num_layers = 2;
    edge.sequence_length = 256;
    edge.dropout_rate = 0.1f;
    edge.device = "cpu";
    configs.push_back(edge);
    
    // Server configuration
    ModelConfig server;
    server.embed_dim = 512;
    server.num_heads = 16;
    server.num_layers = 6;
    server.sequence_length = 512;
    server.dropout_rate = 0.1f;
    server.device = "gpu";
    configs.push_back(server);
    
    // Large configuration
    ModelConfig large;
    large.embed_dim = 1024;
    large.num_heads = 32;
    large.num_layers = 12;
    large.sequence_length = 1024;
    large.dropout_rate = 0.1f;
    large.device = "gpu";
    configs.push_back(large);
    
    return configs;
}

bool validate_config(const ModelConfig& config) {
    // Check basic constraints
    if (config.embed_dim == 0 || config.embed_dim > Constants::MAX_EMBED_DIM) {
        return false;
    }
    
    if (config.num_heads == 0 || config.num_heads > Constants::MAX_NUM_HEADS) {
        return false;
    }
    
    if (config.num_layers == 0 || config.num_layers > Constants::MAX_NUM_LAYERS) {
        return false;
    }
    
    if (config.sequence_length == 0 || config.sequence_length > Constants::MAX_SEQUENCE_LENGTH) {
        return false;
    }
    
    // Check that embed_dim is divisible by num_heads
    if (config.embed_dim % config.num_heads != 0) {
        return false;
    }
    
    // Check dropout rate
    if (config.dropout_rate < 0.0f || config.dropout_rate >= 1.0f) {
        return false;
    }
    
    // Check device string
    if (config.device != "cpu" && config.device != "gpu" && config.device != "metal") {
        return false;
    }
    
    return true;
}

std::string config_to_string(const ModelConfig& config) {
    std::ostringstream oss;
    oss << "ModelConfig{";
    oss << "embed_dim=" << config.embed_dim;
    oss << ", num_heads=" << config.num_heads;
    oss << ", num_layers=" << config.num_layers;
    oss << ", sequence_length=" << config.sequence_length;
    oss << ", dropout_rate=" << config.dropout_rate;
    oss << ", use_quantization=" << (config.use_quantization ? "true" : "false");
    oss << ", use_sparse_attention=" << (config.use_sparse_attention ? "true" : "false");
    oss << ", device=\"" << config.device << "\"";
    oss << "}";
    return oss.str();
}

ModelConfig config_from_string(const std::string& config_str) {
    ModelConfig config;
    
    // Simple parsing (in a real implementation, this would be more robust)
    std::istringstream iss(config_str);
    std::string token;
    
    while (std::getline(iss, token, ',')) {
        size_t pos = token.find('=');
        if (pos != std::string::npos) {
            std::string key = token.substr(0, pos);
            std::string value = token.substr(pos + 1);
            
            // Remove any remaining braces or quotes
            key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
            value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
            value.erase(std::remove(value.begin(), value.end(), '{'), value.end());
            value.erase(std::remove(value.begin(), value.end(), '}'), value.end());
            value.erase(std::remove(value.begin(), value.end(), '\"'), value.end());
            
            if (key == "embed_dim") {
                config.embed_dim = std::stoul(value);
            } else if (key == "num_heads") {
                config.num_heads = std::stoul(value);
            } else if (key == "num_layers") {
                config.num_layers = std::stoul(value);
            } else if (key == "sequence_length") {
                config.sequence_length = std::stoul(value);
            } else if (key == "dropout_rate") {
                config.dropout_rate = std::stof(value);
            } else if (key == "use_quantization") {
                config.use_quantization = (value == "true");
            } else if (key == "use_sparse_attention") {
                config.use_sparse_attention = (value == "true");
            } else if (key == "device") {
                config.device = value;
            }
        }
    }
    
    return config;
}

// PerformanceProfiler implementation
void PerformanceProfiler::start_profiling() {
    start_time_ = std::chrono::steady_clock::now();
    reset();
}

void PerformanceProfiler::end_profiling() {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time_);
    
    metrics_.avg_latency_ms = duration.count();
    metrics_.total_inferences = 1;
    metrics_.throughput_tokens_per_sec = 1000.0 / metrics_.avg_latency_ms;
    metrics_.last_updated = end_time;
}

PerformanceMetrics PerformanceProfiler::get_metrics() const {
    return metrics_;
}

void PerformanceProfiler::reset() {
    metrics_ = PerformanceMetrics{};
    metrics_.last_updated = std::chrono::steady_clock::now();
}

} // namespace Utils
} // namespace TinyML
