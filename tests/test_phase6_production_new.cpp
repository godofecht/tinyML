//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 6: Production API Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <thread>
#include <future>

#include "ProductionAPI.h"

class Phase6ProductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test data
        embed_dim = 256;
        test_input.resize(embed_dim);
        
        for (size_t i = 0; i < embed_dim; ++i) {
            test_input[i] = dis(gen);
        }
        
        // Create test batch
        test_batch.resize(10);
        for (size_t i = 0; i < 10; ++i) {
            test_batch[i].resize(embed_dim);
            for (size_t j = 0; j < embed_dim; ++j) {
                test_batch[i][j] = dis(gen);
            }
        }
    }
    
    size_t embed_dim;
    std::vector<float> test_input;
    std::vector<std::vector<float>> test_batch;
};

// Test basic streaming transformer
TEST_F(Phase6ProductionTest, BasicStreamingTransformer) {
    std::cout << "\n=== Basic Streaming Transformer Test ===\n";
    
    ML::Production::StreamingTransformer::Config config;
    config.embed_dim = embed_dim;
    config.num_heads = 8;
    config.num_layers = 2;
    config.target_latency_ms = 10.0f;
    config.max_memory_mb = 5;
    config.enable_quantization = false;
    config.enable_streaming = false;
    
    ML::Production::StreamingTransformer transformer(config);
    
    // Test basic processing
    auto output = transformer.process(test_input);
    
    // Verify output
    ASSERT_EQ(output.size(), embed_dim);
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in output";
    }
    
    // Check health
    EXPECT_TRUE(transformer.is_healthy()) << "Transformer should be healthy";
    
    std::cout << "Basic streaming transformer: PASS\n";
}

// Test streaming interface
TEST_F(Phase6ProductionTest, StreamingInterface) {
    std::cout << "\n=== Streaming Interface Test ===\n";
    
    ML::Production::StreamingTransformer::Config config;
    config.embed_dim = embed_dim;
    config.num_heads = 4;
    config.num_layers = 2;
    config.enable_streaming = true;
    config.chunk_size = 32;
    
    ML::Production::StreamingTransformer transformer(config);
    
    // Start streaming
    transformer.start_stream();
    
    // Push chunks
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < 5; ++i) {
        auto future = std::async(std::launch::async, [this, &transformer]() {
            transformer.push_chunk(test_input);
        });
        futures.push_back(std::move(future));
    }
    
    // Wait for all chunks to be processed
    for (auto& future : futures) {
        future.wait();
    }
    
    // Collect outputs
    std::vector<std::vector<float>> outputs;
    for (size_t i = 0; i < 5; ++i) {
        auto output = transformer.get_output();
        if (!output.empty()) {
            outputs.push_back(output);
        }
    }
    
    // Stop streaming
    transformer.stop_stream();
    
    // Verify outputs
    EXPECT_GE(outputs.size(), 3) << "Should have processed at least some chunks";
    
    for (const auto& output : outputs) {
        ASSERT_EQ(output.size(), embed_dim);
        
        for (float val : output) {
            ASSERT_FALSE(std::isnan(val));
            ASSERT_FALSE(std::isinf(val));
        }
    }
    
    std::cout << "Streaming interface: PASS\n";
}

// Test batch processing
TEST_F(Phase6ProductionTest, BatchProcessing) {
    std::cout << "\n=== Batch Processing Test ===\n";
    
    ML::Production::StreamingTransformer::Config config;
    config.embed_dim = embed_dim;
    config.num_heads = 4;
    config.num_layers = 2;
    config.use_multi_threading = true;
    
    ML::Production::StreamingTransformer transformer(config);
    
    // Test batch processing
    auto outputs = transformer.process_batch(test_batch);
    
    // Verify outputs
    ASSERT_EQ(outputs.size(), test_batch.size());
    
    for (size_t i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs[i].size(), embed_dim);
        
        for (float val : outputs[i]) {
            ASSERT_FALSE(std::isnan(val));
            ASSERT_FALSE(std::isinf(val));
        }
    }
    
    std::cout << "Batch processing: PASS\n";
}

// Test optimization methods
TEST_F(Phase6ProductionTest, OptimizationMethods) {
    std::cout << "\n=== Optimization Methods Test ===\n";
    
    ML::Production::StreamingTransformer::Config config;
    config.embed_dim = embed_dim;
    config.num_heads = 8;
    config.num_layers = 4;
    config.enable_quantization = false;
    
    ML::Production::StreamingTransformer transformer(config);
    
    // Test latency optimization
    transformer.optimize_for_latency();
    auto latency_output = transformer.process(test_input);
    ASSERT_EQ(latency_output.size(), embed_dim);
    
    // Test memory optimization
    transformer.optimize_for_memory();
    auto memory_output = transformer.process(test_input);
    ASSERT_EQ(memory_output.size(), embed_dim);
    
    // Test throughput optimization
    transformer.optimize_for_throughput();
    auto throughput_output = transformer.process(test_input);
    ASSERT_EQ(throughput_output.size(), embed_dim);
    
    // Test quantization
    transformer.enable_quantization();
    auto quantized_output = transformer.process(test_input);
    ASSERT_EQ(quantized_output.size(), embed_dim);
    
    transformer.disable_quantization();
    auto dequantized_output = transformer.process(test_input);
    ASSERT_EQ(dequantized_output.size(), embed_dim);
    
    std::cout << "Optimization methods: PASS\n";
}

// Test online learning
TEST_F(Phase6ProductionTest, OnlineLearning) {
    std::cout << "\n=== Online Learning Test ===\n";
    
    ML::Production::StreamingTransformer::Config config;
    config.embed_dim = embed_dim;
    config.num_heads = 4;
    config.num_layers = 2;
    config.adaptation_rate = 0.01f;
    
    ML::Production::StreamingTransformer transformer(config);
    
    // Create target data
    std::vector<float> target(embed_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < embed_dim; ++i) {
        target[i] = dis(gen);
    }
    
    // Test online update
    transformer.update_online(test_input, target);
    
    // Check metrics
    auto metrics = transformer.get_metrics();
    EXPECT_TRUE(metrics.is_adapting) << "Should be adapting after online update";
    
    // Test adaptation rate
    transformer.set_adaptation_rate(0.05f);
    
    std::cout << "Online learning: PASS\n";
}

// Test metrics and monitoring
TEST_F(Phase6ProductionTest, MetricsAndMonitoring) {
    std::cout << "\n=== Metrics and Monitoring Test ===\n";
    
    ML::Production::StreamingTransformer::Config config;
    config.embed_dim = embed_dim;
    config.num_heads = 4;
    config.num_layers = 2;
    config.target_latency_ms = 5.0f;
    
    ML::Production::StreamingTransformer transformer(config);
    
    // Reset metrics
    transformer.reset_metrics();
    
    // Process some data to generate metrics
    const int iterations = 10;
    for (int i = 0; i < iterations; ++i) {
        transformer.process(test_input);
    }
    
    // Get metrics
    auto metrics = transformer.get_metrics();
    
    // Verify metrics
    EXPECT_GT(metrics.total_tokens_processed, 0) << "Should have processed tokens";
    EXPECT_GT(metrics.average_latency_ms, 0.0f) << "Should have latency measurements";
    EXPECT_GT(metrics.throughput_tokens_per_sec, 0.0f) << "Should have throughput measurements";
    EXPECT_GT(metrics.memory_usage_mb, 0) << "Should have memory usage";
    
    // Print performance report
    transformer.print_performance_report();
    
    std::cout << "Metrics and monitoring: PASS\n";
}

// Test error handling
TEST_F(Phase6ProductionTest, ErrorHandling) {
    std::cout << "\n=== Error Handling Test ===\n";
    
    ML::Production::StreamingTransformer::Config config;
    config.embed_dim = embed_dim;
    config.num_heads = 4;
    config.num_layers = 2;
    
    ML::Production::StreamingTransformer transformer(config);
    
    // Test with invalid input size
    std::vector<float> invalid_input(embed_dim / 2, 0.5f);
    auto output = transformer.process(invalid_input);
    
    // Should handle error gracefully
    EXPECT_TRUE(output.empty()) << "Should return empty output for invalid input";
    EXPECT_FALSE(transformer.is_healthy()) << "Should be unhealthy after error";
    
    // Check error message
    std::string error = transformer.get_last_error();
    EXPECT_FALSE(error.empty()) << "Should have error message";
    
    std::cout << "Error message: " << error << std::endl;
    
    std::cout << "Error handling: PASS\n";
}

// Test factory methods
TEST_F(Phase6ProductionTest, FactoryMethods) {
    std::cout << "\n=== Factory Methods Test ===\n";
    
    // Test mobile transformer
    auto mobile_transformer = ML::Production::TransformerFactory::create_mobile_transformer(1000, 10.0f);
    ASSERT_NE(mobile_transformer, nullptr);
    
    auto mobile_output = mobile_transformer->process(test_input);
    ASSERT_EQ(mobile_output.size(), 64); // Mobile uses 64D embedding
    
    // Test edge transformer
    auto edge_transformer = ML::Production::TransformerFactory::create_edge_transformer(1000, 5.0f);
    ASSERT_NE(edge_transformer, nullptr);
    
    auto edge_output = edge_transformer->process(test_input);
    ASSERT_EQ(edge_output.size(), 128); // Edge uses 128D embedding
    
    // Test server transformer
    auto server_transformer = ML::Production::TransformerFactory::create_server_transformer(1000, 1.0f);
    ASSERT_NE(server_transformer, nullptr);
    
    auto server_output = server_transformer->process(test_input);
    ASSERT_EQ(server_output.size(), 256); // Server uses 256D embedding
    
    std::cout << "Mobile transformer (64D): PASS\n";
    std::cout << "Edge transformer (128D): PASS\n";
    std::cout << "Server transformer (256D): PASS\n";
    
    std::cout << "Factory methods: PASS\n";
}

// Test edge optimizer
TEST_F(Phase6ProductionTest, EdgeOptimizer) {
    std::cout << "\n=== Edge Optimizer Test ===\n";
    
    // Create device profile
    ML::Production::EdgeOptimizer::DeviceProfile profile;
    profile.device_type = "edge";
    profile.cpu_cores = std::thread::hardware_concurrency();
    profile.memory_mb = 4096;
    profile.has_gpu = false;
    profile.has_neon = ML::XSIMD::XSIMDVector::has_simd_support();
    profile.power_budget_watts = 15.0f;
    
    ML::Production::EdgeOptimizer optimizer(profile);
    
    // Test device optimization
    optimizer.optimize_for_device();
    
    // Test power optimization
    optimizer.optimize_for_power_budget(10.0f);
    
    // Test thermal optimization
    optimizer.optimize_for_thermal_constraints(70.0f);
    
    // Test model compression
    optimizer.compress_model(0.5f);
    optimizer.prune_model(0.3f);
    optimizer.quantize_model(8);
    
    // Test performance tuning
    optimizer.tune_for_latency(5.0f);
    optimizer.tune_for_memory(2048);
    optimizer.tune_for_power(10.0f);
    
    // Get device profile
    auto detected_profile = optimizer.get_device_profile();
    EXPECT_EQ(detected_profile.device_type, "edge");
    
    // Print optimization report
    optimizer.print_optimization_report();
    
    // Get optimization suggestions
    auto suggestions = optimizer.get_optimization_suggestions();
    EXPECT_FALSE(suggestions.empty()) << "Should have optimization suggestions";
    
    std::cout << "Optimization suggestions: " << suggestions.size() << std::endl;
    
    std::cout << "Edge optimizer: PASS\n";
}

// Test hardware accelerator
TEST_F(Phase6ProductionTest, HardwareAccelerator) {
    std::cout << "\n=== Hardware Accelerator Test ===\n";
    
    ML::Production::HardwareAccelerator accelerator;
    
    // Test accelerator detection
    auto available = accelerator.detect_available_accelerators();
    EXPECT_FALSE(available.empty()) << "Should have at least CPU accelerator";
    
    // Test enabling/disabling accelerators
    bool cpu_enabled = accelerator.enable_accelerator(ML::Production::HardwareAccelerator::AcceleratorType::CPU);
    EXPECT_TRUE(cpu_enabled) << "CPU accelerator should be enableable";
    
    auto active = accelerator.get_active_accelerator();
    EXPECT_EQ(active, ML::Production::HardwareAccelerator::AcceleratorType::CPU);
    
    // Test accelerator info
    std::string info = accelerator.get_accelerator_info();
    EXPECT_FALSE(info.empty()) << "Should have accelerator info";
    
    std::cout << "Available accelerators: " << available.size() << std::endl;
    std::cout << "Active accelerator: CPU" << std::endl;
    std::cout << "Accelerator info: " << info << std::endl;
    
    std::cout << "Hardware accelerator: PASS\n";
}

// Test performance benchmarks
TEST_F(Phase6ProductionTest, PerformanceBenchmarks) {
    std::cout << "\n=== Performance Benchmarks Test ===\n";
    
    // Test different configurations
    std::vector<std::pair<std::string, std::unique_ptr<ML::Production::StreamingTransformer>>> transformers;
    
    // Mobile configuration
    transformers.push_back({"Mobile", ML::Production::TransformerFactory::create_mobile_transformer()});
    
    // Edge configuration
    transformers.push_back({"Edge", ML::Production::TransformerFactory::create_edge_transformer()});
    
    // Server configuration
    transformers.push_back({"Server", ML::Production::TransformerFactory::create_server_transformer()});
    
    // Benchmark each configuration
    for (auto& [name, transformer] : transformers) {
        // Warm up
        for (int i = 0; i < 10; ++i) {
            transformer->process(test_input);
        }
        
        // Benchmark
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            transformer->process(test_input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_latency_us = static_cast<double>(duration.count()) / iterations;
        double avg_latency_ms = avg_latency_us / 1000.0;
        
        // Get metrics
        auto metrics = transformer->get_metrics();
        
        std::cout << name << " Transformer:\n";
        std::cout << "  Latency: " << std::fixed << std::setprecision(3) << avg_latency_ms << "ms\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << (1000.0 / avg_latency_ms) << " tokens/sec\n";
        std::cout << "  Memory: " << metrics.memory_usage_mb << "MB\n";
        
        // Performance assertions
        EXPECT_LT(avg_latency_ms, 20.0) << "Latency should be under 20ms";
        EXPECT_GT(1000.0 / avg_latency_ms, 50.0) << "Throughput should be reasonable";
        EXPECT_LT(metrics.memory_usage_mb, 50.0) << "Memory usage should be reasonable";
    }
    
    std::cout << "Performance benchmarks: PASS\n";
}

// Test deployment utilities
TEST_F(Phase6ProductionTest, DeploymentUtils) {
    std::cout << "\n=== Deployment Utilities Test ===\n";
    
    // Create a transformer for testing
    auto transformer = ML::Production::TransformerFactory::create_edge_transformer();
    
    // Test model validation
    bool is_valid = ML::Production::DeploymentUtils::validate_model(*transformer);
    EXPECT_TRUE(is_valid) << "Transformer should be valid";
    
    // Test deployment readiness check
    auto readiness_issues = ML::Production::DeploymentUtils::check_deployment_readiness(*transformer);
    
    std::cout << "Deployment readiness issues: " << readiness_issues.size() << std::endl;
    for (const auto& issue : readiness_issues) {
        std::cout << "  - " << issue << std::endl;
    }
    
    // Test performance profiling
    ML::Production::DeploymentUtils::profile_model(*transformer, 50);
    
    // Generate performance report
    std::string report = ML::Production::DeploymentUtils::generate_performance_report(*transformer);
    EXPECT_FALSE(report.empty()) << "Should generate performance report";
    
    std::cout << "Performance report generated (" << report.length() << " characters)\n";
    
    // Prepare for deployment
    ML::Production::DeploymentUtils::prepare_for_deployment(*transformer);
    
    std::cout << "Deployment utilities: PASS\n";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
