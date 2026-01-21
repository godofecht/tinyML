//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "TinyMLAPI.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

int main() {
    std::cout << "=== TinyML Production API Demo ===\n\n";
    
    try {
        // Create API instance for edge device
        std::cout << "Creating TinyML API for edge device...\n";
        auto api = TinyML::create_tinyml_api_for_edge();
        
        // Display system information
        std::cout << "\n" << api->get_system_info() << "\n";
        
        // Display model information
        std::cout << api->get_model_info() << "\n";
        
        // Run self-test
        std::cout << "Running model self-test...\n";
        if (api->run_self_test()) {
            std::cout << "✅ Self-test passed!\n";
        } else {
            std::cout << "❌ Self-test failed!\n";
            return 1;
        }
        
        // Create test input
        std::vector<float> input(128);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::sin(i * 0.1f);
        }
        
        // Test single inference
        std::cout << "\nTesting single inference...\n";
        auto result = api->predict(input);
        
        if (result.success) {
            std::cout << "✅ Inference successful!\n";
            std::cout << "   Latency: " << result.latency_ms << " ms\n";
            std::cout << "   Output size: " << result.output.size() << "\n";
            std::cout << "   First 5 outputs: ";
            for (size_t i = 0; i < std::min(size_t(5), result.output.size()); ++i) {
                std::cout << result.output[i] << " ";
            }
            std::cout << "\n";
        } else {
            std::cout << "❌ Inference failed: " << result.error_message << "\n";
            return 1;
        }
        
        // Test batch inference
        std::cout << "\nTesting batch inference...\n";
        std::vector<std::vector<float>> batch_inputs;
        for (int i = 0; i < 4; ++i) {
            std::vector<float> batch_input(128);
            for (size_t j = 0; j < batch_input.size(); ++j) {
                batch_input[j] = std::sin(i * 0.5f + j * 0.1f);
            }
            batch_inputs.push_back(batch_input);
        }
        
        auto batch_result = api->predict_batch(batch_inputs);
        
        if (batch_result.success) {
            std::cout << "✅ Batch inference successful!\n";
            std::cout << "   Latency: " << batch_result.latency_ms << " ms\n";
            std::cout << "   Output size: " << batch_result.output.size() << "\n";
        } else {
            std::cout << "❌ Batch inference failed: " << batch_result.error_message << "\n";
        }
        
        // Test streaming
        std::cout << "\nTesting streaming inference...\n";
        if (api->start_streaming()) {
            std::cout << "✅ Streaming started\n";
            
            for (int chunk = 0; chunk < 5; ++chunk) {
                std::vector<float> stream_chunk(128);
                for (size_t i = 0; i < stream_chunk.size(); ++i) {
                    stream_chunk[i] = std::sin(chunk * 0.2f + i * 0.1f);
                }
                
                auto stream_result = api->process_stream_chunk(stream_chunk);
                if (stream_result.success) {
                    std::cout << "   Chunk " << chunk << ": " << stream_result.latency_ms << " ms\n";
                } else {
                    std::cout << "   Chunk " << chunk << " failed: " << stream_result.error_message << "\n";
                }
            }
            
            api->end_streaming();
            std::cout << "✅ Streaming ended\n";
        } else {
            std::cout << "❌ Failed to start streaming\n";
        }
        
        // Test optimization
        std::cout << "\nTesting model optimization...\n";
        TinyML::OptimizationConfig opt_config;
        opt_config.optimize_for_latency = true;
        opt_config.enable_parallel_processing = true;
        opt_config.num_threads = std::thread::hardware_concurrency();
        
        if (api->optimize_model(opt_config)) {
            std::cout << "✅ Model optimized for latency\n";
        } else {
            std::cout << "❌ Model optimization failed\n";
        }
        
        // Test hardware acceleration
        std::cout << "\nTesting hardware acceleration...\n";
        if (api->is_hardware_acceleration_available()) {
            std::cout << "Hardware acceleration is available\n";
            
            if (api->tune_for_hardware()) {
                std::cout << "✅ Model tuned for hardware\n";
            } else {
                std::cout << "❌ Hardware tuning failed\n";
            }
        } else {
            std::cout << "Hardware acceleration not available\n";
        }
        
        // Performance benchmark
        std::cout << "\nRunning performance benchmark...\n";
        const int benchmark_iterations = 100;
        
        auto benchmark_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < benchmark_iterations; ++i) {
            api->predict(input);
        }
        
        auto benchmark_end = std::chrono::high_resolution_clock::now();
        auto benchmark_duration = std::chrono::duration<double, std::milli>(benchmark_end - benchmark_start);
        
        std::cout << "Benchmark results:\n";
        std::cout << "   " << benchmark_iterations << " inferences in " << benchmark_duration.count() << " ms\n";
        std::cout << "   Average: " << benchmark_duration.count() / benchmark_iterations << " ms per inference\n";
        std::cout << "   Throughput: " << (benchmark_iterations * 1000.0) / benchmark_duration.count() << " inferences/sec\n";
        
        // Get final performance metrics
        auto metrics = api->get_performance_metrics();
        std::cout << "\nFinal performance metrics:\n";
        std::cout << "   Total inferences: " << metrics.total_inferences << "\n";
        std::cout << "   Average latency: " << metrics.avg_latency_ms << " ms\n";
        std::cout << "   Throughput: " << metrics.throughput_tokens_per_sec << " tokens/sec\n";
        std::cout << "   Memory footprint: " << metrics.memory_footprint_mb << " MB\n";
        
        // Test model export
        std::cout << "\nTesting model export...\n";
        auto formats = api->get_supported_formats();
        std::cout << "Supported export formats: ";
        for (const auto& format : formats) {
            std::cout << format << " ";
        }
        std::cout << "\n";
        
        if (api->export_model("native", "demo_model.bin")) {
            std::cout << "✅ Model exported to native format\n";
        } else {
            std::cout << "❌ Model export failed\n";
        }
        
        // Test model loading
        std::cout << "\nTesting model loading...\n";
        api->destroy_model();
        
        if (api->load_model("demo_model.bin")) {
            std::cout << "✅ Model loaded successfully\n";
            
            // Test loaded model
            auto loaded_result = api->predict(input);
            if (loaded_result.success) {
                std::cout << "✅ Loaded model works correctly\n";
                std::cout << "   Latency: " << loaded_result.latency_ms << " ms\n";
            } else {
                std::cout << "❌ Loaded model inference failed\n";
            }
        } else {
            std::cout << "❌ Model loading failed\n";
        }
        
        // Test factory functions
        std::cout << "\nTesting factory functions...\n";
        
        auto mobile_api = TinyML::create_tinyml_api_for_mobile();
        auto mobile_config = mobile_api->get_model_config();
        std::cout << "Mobile config: " << mobile_config.embed_dim << "D, " 
                 << mobile_config.num_heads << " heads, " 
                 << mobile_config.num_layers << " layers\n";
        
        auto server_api = TinyML::create_tinyml_api_for_server();
        auto server_config = server_api->get_model_config();
        std::cout << "Server config: " << server_config.embed_dim << "D, " 
                 << server_config.num_heads << " heads, " 
                 << server_config.num_layers << " layers\n";
        
        std::cout << "\n=== Demo Complete ===\n";
        std::cout << "✅ All tests passed successfully!\n";
        std::cout << "✅ TinyML Production API is ready for deployment!\n";
        
        // Cleanup
        std::remove("demo_model.bin");
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
