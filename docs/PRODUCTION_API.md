# TinyML Production API Documentation

## Overview

The TinyML Production API provides a clean, standardized interface for deploying real-time transformer models on edge devices, mobile platforms, and servers. This API abstracts away the complexity of model management, optimization, and hardware acceleration while maintaining high performance.

## Quick Start

```cpp
#include "TinyMLAPI.h"

// Create API instance
auto api = TinyML::create_tinyml_api_for_edge();

// Create model
TinyML::ModelConfig config;
config.embed_dim = 128;
config.num_heads = 4;
config.num_layers = 2;
api->create_model(config);

// Run inference
std::vector<float> input(128, 0.1f);
auto result = api->predict(input);

if (result.success) {
    std::cout << "Inference successful! Latency: " << result.latency_ms << "ms\n";
}
```

## API Reference

### Core Classes

#### `TinyMLAPI`
Main API class for model management and inference.

#### `ModelConfig`
Configuration structure for model parameters.

#### `OptimizationConfig`
Configuration for performance optimization.

#### `DeploymentConfig`
Configuration for deployment settings.

#### `InferenceResult`
Result structure containing output and performance metrics.

### Factory Functions

- `create_tinyml_api()` - General purpose API instance
- `create_tinyml_api_for_edge()` - Optimized for edge devices
- `create_tinyml_api_for_mobile()` - Optimized for mobile devices  
- `create_tinyml_api_for_server()` - Optimized for server deployment

### Key Methods

#### Model Management
- `create_model(config)` - Create a new model
- `load_model(path)` - Load model from file
- `save_model(path)` - Save model to file
- `destroy_model()` - Destroy current model

#### Inference
- `predict(input)` - Single inference
- `predict_batch(inputs)` - Batch inference
- `start_streaming()` - Start streaming mode
- `process_stream_chunk(chunk)` - Process streaming chunk
- `end_streaming()` - End streaming mode

#### Optimization
- `optimize_model(config)` - Optimize model performance
- `tune_for_hardware()` - Auto-tune for current hardware
- `get_performance_metrics()` - Get performance statistics

#### Deployment
- `deploy_model(config)` - Deploy for specific platform
- `export_model(format, path)` - Export to different formats
- `enable_hardware_acceleration()` - Enable GPU/Metal acceleration

## Configuration

### Model Configuration

```cpp
TinyML::ModelConfig config;
config.embed_dim = 256;           // Embedding dimension
config.num_heads = 8;             // Number of attention heads
config.num_layers = 4;            // Number of transformer layers
config.sequence_length = 512;     // Maximum sequence length
config.dropout_rate = 0.1f;       // Dropout rate
config.use_quantization = false;  // Enable quantization
config.use_sparse_attention = false; // Enable sparse attention
config.device = "cpu";            // Target device ("cpu", "gpu", "metal")
```

### Optimization Configuration

```cpp
TinyML::OptimizationConfig opt_config;
opt_config.optimize_for_latency = true;     // Optimize for speed
opt_config.optimize_for_memory = false;     // Optimize for memory
opt_config.enable_parallel_processing = true; // Enable multi-threading
opt_config.num_threads = 4;                // Number of threads
opt_config.target_latency_ms = 10.0f;       // Target latency
opt_config.max_memory_mb = 10;              // Maximum memory usage
```

### Deployment Configuration

```cpp
TinyML::DeploymentConfig deploy_config;
deploy_config.target_platform = "edge";     // "edge", "mobile", "server"
deploy_config.enable_hardware_acceleration = false; // GPU/Metal acceleration
deploy_config.enable_federated_learning = false;   // Federated learning
deploy_config.model_format = "native";     // Export format
```

## Examples

### Basic Inference Example

```cpp
#include "TinyMLAPI.h"

int main() {
    // Create API for edge device
    auto api = TinyML::create_tinyml_api_for_edge();
    
    // Create test input
    std::vector<float> input(128);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) / input.size();
    }
    
    // Run inference
    auto result = api->predict(input);
    
    if (result.success) {
        std::cout << "Success! Output size: " << result.output.size() << "\n";
        std::cout << "Latency: " << result.latency_ms << " ms\n";
        
        // Process output...
        for (size_t i = 0; i < std::min(result.output.size(), size_t(10)); ++i) {
            std::cout << "output[" << i << "] = " << result.output[i] << "\n";
        }
    } else {
        std::cout << "Error: " << result.error_message << "\n";
    }
    
    return 0;
}
```

### Streaming Processing Example

```cpp
#include "TinyMLAPI.h"

int main() {
    auto api = TinyML::create_tinyml_api_for_edge();
    
    // Start streaming mode
    if (!api->start_streaming()) {
        std::cout << "Failed to start streaming\n";
        return 1;
    }
    
    // Process multiple chunks (simulating real-time data)
    for (int chunk_id = 0; chunk_id < 10; ++chunk_id) {
        // Create chunk data
        std::vector<float> chunk(128);
        for (size_t i = 0; i < chunk.size(); ++i) {
            chunk[i] = std::sin(chunk_id + i * 0.1f);
        }
        
        // Process chunk
        auto result = api->process_stream_chunk(chunk);
        
        if (result.success) {
            std::cout << "Chunk " << chunk_id << " processed in " 
                     << result.latency_ms << " ms\n";
        } else {
            std::cout << "Error processing chunk: " << result.error_message << "\n";
        }
    }
    
    // End streaming
    api->end_streaming();
    
    // Get final performance metrics
    auto metrics = api->get_performance_metrics();
    std::cout << "Total inferences: " << metrics.total_inferences << "\n";
    std::cout << "Average latency: " << metrics.avg_latency_ms << " ms\n";
    std::cout << "Throughput: " << metrics.throughput_tokens_per_sec << " tokens/sec\n";
    
    return 0;
}
```

### Batch Processing Example

```cpp
#include "TinyMLAPI.h"

int main() {
    auto api = TinyML::create_tinyml_api_for_server();
    
    // Create batch of inputs
    std::vector<std::vector<float>> batch;
    const int batch_size = 8;
    const int input_size = 512;
    
    for (int i = 0; i < batch_size; ++i) {
        std::vector<float> input(input_size);
        for (int j = 0; j < input_size; ++j) {
            input[j] = std::sin(i * 0.1f + j * 0.01f);
        }
        batch.push_back(input);
    }
    
    // Process batch
    auto start = std::chrono::high_resolution_clock::now();
    auto result = api->predict_batch(batch);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (result.success) {
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        std::cout << "Batch processed in " << duration.count() << " ms\n";
        std::cout << "Per-sample latency: " << result.latency_ms << " ms\n";
        std::cout << "Output size: " << result.output.size() << "\n";
    } else {
        std::cout << "Batch processing failed: " << result.error_message << "\n";
    }
    
    return 0;
}
```

### Model Optimization Example

```cpp
#include "TinyMLAPI.h"

int main() {
    auto api = TinyML::create_tinyml_api();
    
    // Create model
    TinyML::ModelConfig config;
    config.embed_dim = 256;
    config.num_heads = 8;
    config.num_layers = 4;
    api->create_model(config);
    
    // Optimize for latency
    TinyML::OptimizationConfig opt_config;
    opt_config.optimize_for_latency = true;
    opt_config.enable_parallel_processing = true;
    opt_config.num_threads = std::thread::hardware_concurrency();
    opt_config.target_latency_ms = 5.0f; // Target 5ms
    
    if (api->optimize_model(opt_config)) {
        std::cout << "Model optimized for latency\n";
    }
    
    // Test performance
    std::vector<float> input(256, 0.1f);
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        api->predict(input);
    }
    
    // Benchmark
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        api->predict(input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double, std::milli>(end - start);
    
    std::cout << "Average latency: " << total_time.count() / iterations << " ms\n";
    
    // Check if target met
    auto metrics = api->get_performance_metrics();
    if (metrics.avg_latency_ms <= opt_config.target_latency_ms) {
        std::cout << "✅ Target latency achieved!\n";
    } else {
        std::cout << "❌ Target latency not met\n";
    }
    
    return 0;
}
```

### Hardware Acceleration Example

```cpp
#include "TinyMLAPI.h"

int main() {
    auto api = TinyML::create_tinyml_api();
    
    // Check hardware acceleration availability
    if (api->is_hardware_acceleration_available()) {
        std::cout << "Hardware acceleration is available\n";
        
        // Try to enable Metal (macOS) or GPU (other platforms)
        if (api->enable_metal_acceleration()) {
            std::cout << "Metal acceleration enabled\n";
        } else if (api->enable_gpu_acceleration()) {
            std::cout << "GPU acceleration enabled\n";
        } else {
            std::cout << "Failed to enable hardware acceleration\n";
        }
    } else {
        std::cout << "Hardware acceleration not available\n";
    }
    
    // Auto-tune for hardware
    if (api->tune_for_hardware()) {
        std::cout << "Model tuned for current hardware\n";
    }
    
    // Get system information
    std::cout << "\nSystem Information:\n";
    std::cout << api->get_system_info() << "\n";
    
    // Create and test model
    TinyML::ModelConfig config;
    config.embed_dim = 512;
    config.num_heads = 16;
    config.num_layers = 6;
    api->create_model(config);
    
    // Run performance test
    std::vector<float> input(512, 0.1f);
    auto result = api->predict(input);
    
    if (result.success) {
        std::cout << "Test inference successful\n";
        std::cout << "Latency: " << result.latency_ms << " ms\n";
    }
    
    return 0;
}
```

### Model Export Example

```cpp
#include "TinyMLAPI.h"

int main() {
    auto api = TinyML::create_tinyml_api_for_edge();
    
    // Get supported export formats
    auto formats = api->get_supported_formats();
    std::cout << "Supported export formats:\n";
    for (const auto& format : formats) {
        std::cout << "  - " << format << "\n";
    }
    
    // Export to different formats
    std::vector<std::string> export_formats = {"native", "onnx", "tflite"};
    
    for (const auto& format : export_formats) {
        std::string filename = "model." + format;
        if (format == "native") filename = "model.bin";
        else if (format == "onnx") filename = "model.onnx";
        else if (format == "tflite") filename = "model.tflite";
        
        if (api->export_model(format, filename)) {
            std::cout << "✅ Exported to " << format << " format: " << filename << "\n";
        } else {
            std::cout << "❌ Failed to export to " << format << " format\n";
        }
    }
    
    // Test loading exported model
    api->destroy_model();
    if (api->load_model("model.bin")) {
        std::cout << "✅ Successfully loaded exported model\n";
        
        // Test inference with loaded model
        std::vector<float> input(128, 0.1f);
        auto result = api->predict(input);
        
        if (result.success) {
            std::cout << "✅ Loaded model works correctly\n";
        }
    }
    
    return 0;
}
```

## Performance Optimization

### Memory Optimization

```cpp
// Optimize for memory-constrained environments
TinyML::OptimizationConfig mem_opt;
mem_opt.optimize_for_memory = true;
mem_opt.max_memory_mb = 5;  // Limit to 5MB
api->optimize_model(mem_opt);
```

### Latency Optimization

```cpp
// Optimize for low latency
TinyML::OptimizationConfig latency_opt;
latency_opt.optimize_for_latency = true;
latency_opt.enable_parallel_processing = true;
latency_opt.num_threads = 8;
latency_opt.target_latency_ms = 1.0f;  // Target 1ms
api->optimize_model(latency_opt);
```

### Platform-Specific Deployment

```cpp
// Mobile deployment
TinyML::DeploymentConfig mobile_deploy;
mobile_deploy.target_platform = "mobile";
mobile_deploy.enable_hardware_acceleration = false;  // Usually false on mobile
api->deploy_model(mobile_deploy);

// Edge deployment
TinyML::DeploymentConfig edge_deploy;
edge_deploy.target_platform = "edge";
edge_deploy.enable_hardware_acceleration = true;   // Try to use GPU/Metal
api->deploy_model(edge_deploy);

// Server deployment
TinyML::DeploymentConfig server_deploy;
server_deploy.target_platform = "server";
server_deploy.enable_hardware_acceleration = true;  // Use GPU
server_deploy.enable_federated_learning = true;    // Enable federated learning
api->deploy_model(server_deploy);
```

## Error Handling

The API uses exceptions for error handling:

```cpp
try {
    auto api = TinyML::create_tinyml_api();
    
    TinyML::ModelConfig config;
    config.embed_dim = 128;
    config.num_heads = 4;
    api->create_model(config);
    
    auto result = api->predict(input);
    
} catch (const TinyML::TinyMLException& e) {
    std::cout << "TinyML Error: " << e.what() << "\n";
    std::cout << "Error Code: " << static_cast<int>(e.get_error_code()) << "\n";
} catch (const std::exception& e) {
    std::cout << "Standard Error: " << e.what() << "\n";
}
```

## Best Practices

1. **Use Factory Functions**: Start with `create_tinyml_api_for_edge()`, `create_tinyml_api_for_mobile()`, or `create_tinyml_api_for_server()` for pre-optimized configurations.

2. **Optimize After Creation**: Create the model first, then apply optimizations based on your target platform.

3. **Monitor Performance**: Use `get_performance_metrics()` to track latency and throughput.

4. **Handle Errors Gracefully**: Always check `result.success` and handle error messages.

5. **Use Streaming for Real-Time**: For continuous data processing, use the streaming interface.

6. **Export for Deployment**: Use the export functionality to save models in different formats for different platforms.

## Performance Targets

- **Edge Devices**: <10ms latency, <10MB memory
- **Mobile Devices**: <20ms latency, <5MB memory  
- **Server Deployment**: <1ms latency, <100MB memory

## Troubleshooting

### Common Issues

1. **Model Creation Fails**: Check that `embed_dim` is divisible by `num_heads`
2. **High Latency**: Enable hardware acceleration or optimize for latency
3. **Memory Issues**: Optimize for memory or reduce model size
4. **Export Fails**: Ensure the model is created before exporting

### Debug Information

```cpp
// Get detailed model and system information
std::cout << api->get_model_info() << "\n";
std::cout << api->get_system_info() << "\n";

// Run self-test to verify model integrity
if (api->run_self_test()) {
    std::cout << "Model self-test passed\n";
} else {
    std::cout << "Model self-test failed\n";
}
```

## Integration Examples

### Audio Processing

```cpp
// Real-time audio enhancement
auto api = TinyML::create_tinyml_api_for_edge();
api->start_streaming();

while (recording_audio) {
    std::vector<float> audio_chunk = get_audio_chunk();
    auto result = api->process_stream_chunk(audio_chunk);
    
    if (result.success) {
        play_enhanced_audio(result.output);
    }
}

api->end_streaming();
```

### Time Series Prediction

```cpp
// IoT sensor data prediction
auto api = TinyML::create_tinyml_api_for_mobile();

std::vector<float> sensor_data = get_recent_sensor_readings();
auto prediction = api->predict(sensor_data);

if (prediction.success) {
    float predicted_value = prediction.output[0];
    take_action_based_on_prediction(predicted_value);
}
```

### Natural Language Processing

```cpp
// On-device text processing
auto api = TinyML::create_tinyml_api_for_edge();

std::vector<float> text_embedding = embed_text(input_text);
auto classification = api->predict(text_embedding);

if (classification.success) {
    std::string category = interpret_classification_output(classification.output);
    display_category_to_user(category);
}
```

## Advanced Features

### Federated Learning

```cpp
// Enable federated learning
TinyML::DeploymentConfig federated_config;
federated_config.enable_federated_learning = true;
federated_config.target_platform = "edge";
api->deploy_model(federated_config);

// Update model with local gradients
std::vector<float> local_gradients = compute_local_gradients();
if (api->update_model_federated(local_gradients)) {
    std::cout << "Model updated with federated learning\n";
}
```

### Custom Optimization

```cpp
// Create custom optimization configuration
TinyML::OptimizationConfig custom_opt;
custom_opt.optimize_for_latency = true;
custom_opt.optimize_for_memory = true;  // Balance both
custom_opt.enable_parallel_processing = true;
custom_opt.num_threads = 4;
custom_opt.target_latency_ms = 5.0f;
custom_opt.max_memory_mb = 8;

api->optimize_model(custom_opt);
```

This documentation provides a comprehensive guide for using the TinyML Production API in various deployment scenarios.
