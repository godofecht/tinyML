//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "TinyMLAPI.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>
#include <thread>

using namespace TinyML;

class TinyMLAPITest : public ::testing::Test {
protected:
    void SetUp() override {
        api_ = create_tinyml_api();
    }

    void TearDown() override {
        api_->destroy_model();
    }

    std::unique_ptr<TinyMLAPI> api_;
};

TEST_F(TinyMLAPITest, CreateModel) {
    ModelConfig config;
    config.embed_dim = 128;
    config.num_heads = 4;
    config.num_layers = 2;
    config.sequence_length = 256;

    EXPECT_TRUE(api_->create_model(config));
    EXPECT_EQ(api_->get_model_config().embed_dim, 128);
    EXPECT_EQ(api_->get_model_config().num_heads, 4);
}

TEST_F(TinyMLAPITest, InvalidModelConfig) {
    ModelConfig invalid_config;
    invalid_config.embed_dim = 0; // Invalid
    invalid_config.num_heads = 4;
    invalid_config.num_layers = 2;

    EXPECT_FALSE(api_->create_model(invalid_config));
}

TEST_F(TinyMLAPITest, SaveAndLoadModel) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    const std::string model_path = "test_model.bin";
    EXPECT_TRUE(api_->save_model(model_path));
    
    // Destroy and reload
    api_->destroy_model();
    EXPECT_TRUE(api_->load_model(model_path));
    
    // Verify loaded config
    auto loaded_config = api_->get_model_config();
    EXPECT_EQ(loaded_config.embed_dim, 64);
    EXPECT_EQ(loaded_config.num_heads, 2);
    
    // Cleanup
    std::remove(model_path.c_str());
}

TEST_F(TinyMLAPITest, BasicInference) {
    ModelConfig config;
    config.embed_dim = 128;
    config.num_heads = 4;
    config.num_layers = 2;
    config.sequence_length = 256;

    EXPECT_TRUE(api_->create_model(config));
    
    // Create test input
    std::vector<float> input(128, 0.1f);
    
    auto result = api_->predict(input);
    
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.output.empty());
    EXPECT_EQ(result.output.size(), 128);
    EXPECT_GT(result.latency_ms, 0.0);
    
    // Check for valid output values
    for (float val : result.output) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(TinyMLAPITest, BatchInference) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    // Create batch inputs
    std::vector<std::vector<float>> batch_inputs;
    for (int i = 0; i < 4; ++i) {
        std::vector<float> input(64, 0.1f * i);
        batch_inputs.push_back(input);
    }
    
    auto result = api_->predict_batch(batch_inputs);
    
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.output.empty());
    EXPECT_GT(result.latency_ms, 0.0);
}

TEST_F(TinyMLAPITest, StreamingInference) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    EXPECT_TRUE(api_->start_streaming());
    
    // Process multiple chunks
    std::vector<float> chunk(64, 0.1f);
    for (int i = 0; i < 3; ++i) {
        auto result = api_->process_stream_chunk(chunk);
        EXPECT_TRUE(result.success);
        EXPECT_FALSE(result.output.empty());
    }
    
    api_->end_streaming();
}

TEST_F(TinyMLAPITest, ModelOptimization) {
    ModelConfig config;
    config.embed_dim = 128;
    config.num_heads = 4;
    config.num_layers = 2;
    config.sequence_length = 256;

    EXPECT_TRUE(api_->create_model(config));
    
    OptimizationConfig opt_config;
    opt_config.optimize_for_latency = true;
    opt_config.optimize_for_memory = false;
    opt_config.enable_parallel_processing = true;
    opt_config.num_threads = 4;
    
    EXPECT_TRUE(api_->optimize_model(opt_config));
}

TEST_F(TinyMLAPITest, HardwareTuning) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    // This should not crash even if hardware acceleration is not available
    EXPECT_TRUE(api_->tune_for_hardware());
}

TEST_F(TinyMLAPITest, PerformanceMetrics) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    // Reset counters
    api_->reset_performance_counters();
    
    auto initial_metrics = api_->get_performance_metrics();
    EXPECT_EQ(initial_metrics.total_inferences, 0);
    
    // Run some inferences
    std::vector<float> input(64, 0.1f);
    for (int i = 0; i < 5; ++i) {
        api_->predict(input);
    }
    
    auto updated_metrics = api_->get_performance_metrics();
    EXPECT_EQ(updated_metrics.total_inferences, 5);
    EXPECT_GT(updated_metrics.avg_latency_ms, 0.0);
    EXPECT_GT(updated_metrics.throughput_tokens_per_sec, 0.0);
}

TEST_F(TinyMLAPITest, ModelDeployment) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    DeploymentConfig deploy_config;
    deploy_config.target_platform = "edge";
    deploy_config.enable_hardware_acceleration = false;
    deploy_config.model_format = "native";
    
    EXPECT_TRUE(api_->deploy_model(deploy_config));
}

TEST_F(TinyMLAPITest, ModelExport) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    // Test native export
    EXPECT_TRUE(api_->export_model("native", "test_export.bin"));
    
    // Test ONNX export (should create placeholder file)
    EXPECT_TRUE(api_->export_model("onnx", "test_export.onnx"));
    
    // Test TensorFlow Lite export
    EXPECT_TRUE(api_->export_model("tflite", "test_export.tflite"));
    
    // Check supported formats
    auto formats = api_->get_supported_formats();
    EXPECT_FALSE(formats.empty());
    EXPECT_TRUE(std::find(formats.begin(), formats.end(), "native") != formats.end());
    
    // Cleanup
    std::remove("test_export.bin");
    std::remove("test_export.onnx");
    std::remove("test_export.tflite");
}

TEST_F(TinyMLAPITest, HardwareAcceleration) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    // Check availability (should not crash)
    bool available = api_->is_hardware_acceleration_available();
    
    // Try to enable (should not crash even if not available)
    bool gpu_enabled = api_->enable_gpu_acceleration();
    bool metal_enabled = api_->enable_metal_acceleration();
    
    // Results depend on the system, but should not crash
    EXPECT_TRUE(gpu_enabled || !gpu_enabled); // Always true, just checking no crash
    EXPECT_TRUE(metal_enabled || !metal_enabled);
}

TEST_F(TinyMLAPITest, SystemInformation) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    std::string model_info = api_->get_model_info();
    EXPECT_FALSE(model_info.empty());
    EXPECT_NE(model_info.find("Embedding Dimension"), std::string::npos);
    
    std::string system_info = api_->get_system_info();
    EXPECT_FALSE(system_info.empty());
    EXPECT_NE(system_info.find("CPU Cores"), std::string::npos);
}

TEST_F(TinyMLAPITest, SelfTest) {
    ModelConfig config;
    config.embed_dim = 64;
    config.num_heads = 2;
    config.num_layers = 1;
    config.sequence_length = 128;

    EXPECT_TRUE(api_->create_model(config));
    
    EXPECT_TRUE(api_->run_self_test());
}

TEST_F(TinyMLAPITest, FactoryFunctions) {
    // Test edge device factory
    auto edge_api = create_tinyml_api_for_edge();
    EXPECT_NE(edge_api, nullptr);
    
    auto edge_config = edge_api->get_model_config();
    EXPECT_EQ(edge_config.embed_dim, 128);
    EXPECT_EQ(edge_config.num_heads, 4);
    
    // Test mobile device factory
    auto mobile_api = create_tinyml_api_for_mobile();
    EXPECT_NE(mobile_api, nullptr);
    
    auto mobile_config = mobile_api->get_model_config();
    EXPECT_EQ(mobile_config.embed_dim, 64);
    EXPECT_EQ(mobile_config.num_heads, 2);
    
    // Test server device factory
    auto server_api = create_tinyml_api_for_server();
    EXPECT_NE(server_api, nullptr);
    
    auto server_config = server_api->get_model_config();
    EXPECT_EQ(server_config.embed_dim, 512);
    EXPECT_EQ(server_config.num_heads, 16);
}

class TinyMLUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TinyMLUtilsTest, PredefinedConfigs) {
    auto configs = Utils::get_predefined_configs();
    EXPECT_EQ(configs.size(), 4);
    
    // Check mobile config
    auto& mobile = configs[0];
    EXPECT_EQ(mobile.embed_dim, 64);
    EXPECT_EQ(mobile.num_heads, 2);
    EXPECT_EQ(mobile.num_layers, 1);
    
    // Check edge config
    auto& edge = configs[1];
    EXPECT_EQ(edge.embed_dim, 128);
    EXPECT_EQ(edge.num_heads, 4);
    EXPECT_EQ(edge.num_layers, 2);
    
    // Check server config
    auto& server = configs[2];
    EXPECT_EQ(server.embed_dim, 512);
    EXPECT_EQ(server.num_heads, 16);
    EXPECT_EQ(server.num_layers, 6);
}

TEST_F(TinyMLUtilsTest, ConfigValidation) {
    // Valid config
    ModelConfig valid_config;
    valid_config.embed_dim = 128;
    valid_config.num_heads = 8;
    valid_config.num_layers = 2;
    valid_config.sequence_length = 256;
    valid_config.dropout_rate = 0.1f;
    valid_config.device = "cpu";
    
    EXPECT_TRUE(Utils::validate_config(valid_config));
    
    // Invalid embed_dim
    ModelConfig invalid_embed = valid_config;
    invalid_embed.embed_dim = 0;
    EXPECT_FALSE(Utils::validate_config(invalid_embed));
    
    // Invalid num_heads
    ModelConfig invalid_heads = valid_config;
    invalid_heads.num_heads = 0;
    EXPECT_FALSE(Utils::validate_config(invalid_heads));
    
    // Embed_dim not divisible by num_heads
    ModelConfig invalid_divisible = valid_config;
    invalid_divisible.embed_dim = 100;
    invalid_divisible.num_heads = 3;
    EXPECT_FALSE(Utils::validate_config(invalid_divisible));
    
    // Invalid dropout rate
    ModelConfig invalid_dropout = valid_config;
    invalid_dropout.dropout_rate = 1.5f;
    EXPECT_FALSE(Utils::validate_config(invalid_dropout));
}

TEST_F(TinyMLUtilsTest, ConfigSerialization) {
    ModelConfig original;
    original.embed_dim = 256;
    original.num_heads = 8;
    original.num_layers = 4;
    original.sequence_length = 512;
    original.dropout_rate = 0.1f;
    original.use_quantization = true;
    original.device = "gpu";
    
    std::string config_str = Utils::config_to_string(original);
    EXPECT_FALSE(config_str.empty());
    EXPECT_NE(config_str.find("embed_dim=256"), std::string::npos);
    EXPECT_NE(config_str.find("num_heads=8"), std::string::npos);
    
    ModelConfig deserialized = Utils::config_from_string(config_str);
    EXPECT_EQ(deserialized.embed_dim, original.embed_dim);
    EXPECT_EQ(deserialized.num_heads, original.num_heads);
    EXPECT_EQ(deserialized.num_layers, original.num_layers);
    EXPECT_EQ(deserialized.sequence_length, original.sequence_length);
    EXPECT_FLOAT_EQ(deserialized.dropout_rate, original.dropout_rate);
    EXPECT_EQ(deserialized.use_quantization, original.use_quantization);
    EXPECT_EQ(deserialized.device, original.device);
}

TEST_F(TinyMLUtilsTest, PerformanceProfiler) {
    Utils::PerformanceProfiler profiler;
    
    profiler.start_profiling();
    
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    profiler.end_profiling();
    
    auto metrics = profiler.get_metrics();
    EXPECT_GT(metrics.avg_latency_ms, 10.0); // Should be at least 10ms
    EXPECT_EQ(metrics.total_inferences, 1);
    EXPECT_GT(metrics.throughput_tokens_per_sec, 0.0);
    
    profiler.reset();
    auto reset_metrics = profiler.get_metrics();
    EXPECT_EQ(reset_metrics.total_inferences, 0);
}

// Integration test for the complete workflow
class TinyMLIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        api_ = create_tinyml_api_for_edge();
    }

    void TearDown() override {
        api_->destroy_model();
    }

    std::unique_ptr<TinyMLAPI> api_;
};

TEST_F(TinyMLIntegrationTest, CompleteWorkflow) {
    // Model should already be created by factory function
    
    // 1. Run self-test
    EXPECT_TRUE(api_->run_self_test());
    
    // 2. Optimize for edge deployment
    DeploymentConfig deploy_config;
    deploy_config.target_platform = "edge";
    deploy_config.enable_hardware_acceleration = false;
    EXPECT_TRUE(api_->deploy_model(deploy_config));
    
    // 3. Process some data
    std::vector<float> input(128, 0.1f);
    auto result = api_->predict(input);
    EXPECT_TRUE(result.success);
    
    // 4. Check performance
    auto metrics = api_->get_performance_metrics();
    EXPECT_GT(metrics.total_inferences, 0);
    
    // 5. Export model
    EXPECT_TRUE(api_->export_model("native", "integration_test_model.bin"));
    
    // 6. Load and verify
    api_->destroy_model();
    EXPECT_TRUE(api_->load_model("integration_test_model.bin"));
    EXPECT_TRUE(api_->run_self_test());
    
    // Cleanup
    std::remove("integration_test_model.bin");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
