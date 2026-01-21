//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 7: Advanced Attention & Transformers Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

#include "AdvancedAttention.h"

class Phase7AdvancedAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Test data
        embed_dim = 256;
        seq_len = 512;
        test_input.resize(embed_dim);
        
        for (size_t i = 0; i < embed_dim; ++i) {
            test_input[i] = dis(gen);
        }
        
        // Create query, key, value matrices
        query.resize(seq_len * embed_dim);
        key.resize(seq_len * embed_dim);
        value.resize(seq_len * embed_dim);
        
        for (size_t i = 0; i < seq_len * embed_dim; ++i) {
            query[i] = dis(gen);
            key[i] = dis(gen);
            value[i] = dis(gen);
        }
    }
    
    size_t embed_dim;
    size_t seq_len;
    std::vector<float> test_input;
    std::vector<float> query, key, value;
};

// Test Multi-Modal Attention
TEST_F(Phase7AdvancedAttentionTest, MultiModalAttention) {
    std::cout << "\n=== Multi-Modal Attention Test ===\n";
    
    ML::Advanced::AdvancedAttention::Config config;
    config.embed_dim = embed_dim;
    config.seq_len = seq_len;
    config.attention_type = ML::Advanced::AttentionType::MULTI_HEAD;
    
    std::vector<ML::Advanced::MultiModalAttention::ModalityConfig> modality_configs = {
        {ML::Advanced::ModalityType::TEXT, 128, 256, 0.5f, true},
        {ML::Advanced::ModalityType::VISION, 64, 256, 0.3f, true},
        {ML::Advanced::ModalityType::AUDIO, 64, 256, 0.2f, true}
    };
    
    auto attention = ML::Advanced::AdvancedAttentionFactory::create_multi_modal_attention(
        config, modality_configs);
    
    // Test forward pass
    auto output = attention->forward(test_input);
    
    // Verify output
    ASSERT_GT(output.size(), 0) << "Output should not be empty";
    EXPECT_EQ(output.size(), 256) << "Output should match total embedding dimension";
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in output";
    }
    
    // Test memory usage
    size_t memory_usage = attention->get_memory_usage();
    std::cout << "Multi-modal attention memory usage: " << memory_usage << " bytes\n";
    EXPECT_LT(memory_usage, 1024 * 1024) << "Memory usage should be reasonable";
    
    // Test computational complexity
    float complexity = attention->get_computation_complexity();
    std::cout << "Computational complexity: " << complexity << std::endl;
    EXPECT_GT(complexity, 0.0f) << "Complexity should be positive";
    
    // Test modality management
    auto multi_modal_attention = dynamic_cast<ML::Advanced::MultiModalAttention*>(attention.get());
    if (multi_modal_attention) {
        multi_modal_attention->set_modality_weight(ML::Advanced::ModalityType::TEXT, 0.7f);
        
        // Add new modality
        ML::Advanced::MultiModalAttention::ModalityConfig sensor_config{
            ML::Advanced::ModalityType::SENSOR, 32, 256, 0.1f, true
        };
        multi_modal_attention->add_modality(sensor_config);
        
        auto output_with_sensor = attention->forward(test_input);
        EXPECT_GT(output_with_sensor.size(), output.size()) << "Output should be larger with new modality";
    }
    
    std::cout << "Multi-modal attention: PASS\n";
}

// Test Hierarchical Attention
TEST_F(Phase7AdvancedAttentionTest, HierarchicalAttention) {
    std::cout << "\n=== Hierarchical Attention Test ===\n";
    
    ML::Advanced::AdvancedAttention::Config config;
    config.embed_dim = embed_dim;
    config.seq_len = seq_len;
    config.attention_type = ML::Advanced::AttentionType::HIERARCHICAL;
    
    std::vector<ML::Advanced::HierarchicalAttention::LevelConfig> levels = {
        {0, 512, 8, 64, 0.5f},
        {1, 256, 4, 32, 0.5f},
        {2, 128, 2, 16, 0.5f}
    };
    
    auto attention = ML::Advanced::AdvancedAttentionFactory::create_hierarchical_attention(
        config, levels);
    
    // Test forward pass
    auto output = attention->forward(test_input);
    
    // Verify output
    ASSERT_GT(output.size(), 0) << "Output should not be empty";
    EXPECT_EQ(output.size(), 128) << "Output should match top level embedding dimension";
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in output";
    }
    
    // Test level management
    auto hierarchical_attention = dynamic_cast<ML::Advanced::HierarchicalAttention*>(attention.get());
    if (hierarchical_attention) {
        size_t initial_levels = 3;
        hierarchical_attention->add_level({3, 64, 1, 8, 0.5f});
        
        auto output_with_new_level = attention->forward(test_input);
        EXPECT_EQ(output_with_new_level.size(), 64) << "Output should match new top level dimension";
        
        // Test level output retrieval
        auto level_output = hierarchical_attention->get_level_output(1);
        EXPECT_EQ(level_output.size(), 256) << "Level 1 output should match its dimension";
    }
    
    // Test memory usage
    size_t memory_usage = attention->get_memory_usage();
    std::cout << "Hierarchical attention memory usage: " << memory_usage << " bytes\n";
    EXPECT_LT(memory_usage, 1024 * 1024) << "Memory usage should be reasonable";
    
    std::cout << "Hierarchical attention: PASS\n";
}

// Test Sparse Attention
TEST_F(Phase7AdvancedAttentionTest, SparseAttention) {
    std::cout << "\n=== Sparse Attention Test ===\n";
    
    ML::Advanced::AdvancedAttention::Config config;
    config.embed_dim = embed_dim;
    config.seq_len = seq_len;
    config.attention_type = ML::Advanced::AttentionType::SPARSE;
    config.sparsity_ratio = 0.1f;
    
    ML::Advanced::SparseAttention::SparseConfig sparse_config;
    sparse_config.local_window_size = 64;
    sparse_config.global_tokens = 16;
    sparse_config.sparsity_ratio = 0.1f;
    
    auto attention = ML::Advanced::AdvancedAttentionFactory::create_sparse_attention(
        config, sparse_config);
    
    // Test forward pass
    auto output = attention->forward(query, key, value);
    
    // Verify output
    ASSERT_GT(output.size(), 0) << "Output should not be empty";
    EXPECT_EQ(output.size(), seq_len * embed_dim) << "Output should match input dimensions";
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in output";
    }
    
    // Test sparse attention specific methods
    auto sparse_attention = dynamic_cast<ML::Advanced::SparseAttention*>(attention.get());
    if (sparse_attention) {
        // Set global tokens
        std::vector<size_t> global_indices = {0, 128, 256, 384};
        sparse_attention->set_global_tokens(global_indices);
        
        // Update sparsity pattern
        sparse_attention->update_sparsity_pattern();
        
        // Get attention pattern
        auto pattern = sparse_attention->get_attention_pattern();
        EXPECT_GT(pattern.size(), 0) << "Attention pattern should not be empty";
    }
    
    // Test memory usage
    size_t memory_usage = attention->get_memory_usage();
    std::cout << "Sparse attention memory usage: " << memory_usage << " bytes\n";
    EXPECT_LT(memory_usage, 2 * 1024 * 1024) << "Memory usage should be reasonable for sparse attention";
    
    std::cout << "Sparse attention: PASS\n";
}

// Test Linear Attention
TEST_F(Phase7AdvancedAttentionTest, LinearAttention) {
    std::cout << "\n=== Linear Attention Test ===\n";
    
    ML::Advanced::AdvancedAttention::Config config;
    config.embed_dim = embed_dim;
    config.seq_len = seq_len;
    config.attention_type = ML::Advanced::AttentionType::LINEAR;
    
    ML::Advanced::LinearAttention::LinearConfig linear_config;
    linear_config.kernel_type = "elu";
    linear_config.kernel_param = 1.0f;
    linear_config.feature_dim = 64;
    
    auto attention = ML::Advanced::AdvancedAttentionFactory::create_linear_attention(
        config, linear_config);
    
    // Test forward pass
    auto output = attention->forward(query, key, value);
    
    // Verify output
    ASSERT_GT(output.size(), 0) << "Output should not be empty";
    EXPECT_EQ(output.size(), seq_len * embed_dim) << "Output should match input dimensions";
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in output";
    }
    
    // Test linear attention specific methods
    auto linear_attention = dynamic_cast<ML::Advanced::LinearAttention*>(attention.get());
    if (linear_attention) {
        // Test different kernel functions
        linear_attention->set_kernel_function("relu");
        auto relu_output = attention->forward(query, key, value);
        EXPECT_GT(relu_output.size(), 0) << "ReLU kernel output should not be empty";
        
        linear_attention->set_kernel_function("gaussian");
        auto gaussian_output = attention->forward(query, key, value);
        EXPECT_GT(gaussian_output.size(), 0) << "Gaussian kernel output should not be empty";
        
        // Test feature map
        auto feature_map = linear_attention->get_feature_map(test_input);
        EXPECT_EQ(feature_map.size(), linear_config.feature_dim) << "Feature map should have correct dimension";
    }
    
    // Test memory usage
    size_t memory_usage = attention->get_memory_usage();
    std::cout << "Linear attention memory usage: " << memory_usage << " bytes\n";
    EXPECT_LT(memory_usage, 1024 * 1024) << "Memory usage should be reasonable";
    
    std::cout << "Linear attention: PASS\n";
}

// Test Local Attention
TEST_F(Phase7AdvancedAttentionTest, LocalAttention) {
    std::cout << "\n=== Local Attention Test ===\n";
    
    ML::Advanced::AdvancedAttention::Config config;
    config.embed_dim = embed_dim;
    config.seq_len = seq_len;
    config.attention_type = ML::Advanced::AttentionType::LOCAL;
    
    ML::Advanced::LocalAttention::LocalConfig local_config;
    local_config.window_size = 64;
    local_config.use_causal_window = false;
    local_config.stride = 32;
    
    auto attention = ML::Advanced::AdvancedAttentionFactory::create_local_attention(
        config, local_config);
    
    // Test forward pass
    auto output = attention->forward(query, key, value);
    
    // Verify output
    ASSERT_GT(output.size(), 0) << "Output should not be empty";
    EXPECT_EQ(output.size(), seq_len * embed_dim) << "Output should match input dimensions";
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in output";
    }
    
    // Test local attention specific methods
    auto local_attention = dynamic_cast<ML::Advanced::LocalAttention*>(attention.get());
    if (local_attention) {
        // Test window size changes
        local_attention->set_window_size(128);
        auto larger_window_output = attention->forward(query, key, value);
        EXPECT_GT(larger_window_output.size(), 0) << "Larger window output should not be empty";
        
        // Get local windows
        auto windows = local_attention->get_local_windows();
        EXPECT_GT(windows.size(), 0) << "Local windows should not be empty";
        
        // Verify window boundaries
        for (const auto& window : windows) {
            EXPECT_LT(window.first, window.second) << "Window start should be less than end";
            EXPECT_LE(window.second, seq_len) << "Window end should not exceed sequence length";
        }
    }
    
    // Test memory usage
    size_t memory_usage = attention->get_memory_usage();
    std::cout << "Local attention memory usage: " << memory_usage << " bytes\n";
    EXPECT_LT(memory_usage, 1024 * 1024) << "Memory usage should be reasonable";
    
    std::cout << "Local attention: PASS\n";
}

// Test Advanced Transformer Block
TEST_F(Phase7AdvancedAttentionTest, AdvancedTransformerBlock) {
    std::cout << "\n=== Advanced Transformer Block Test ===\n";
    
    ML::Advanced::AdvancedTransformerBlock::BlockConfig block_config;
    block_config.embed_dim = embed_dim;
    block_config.ff_dim = embed_dim * 4;
    block_config.num_heads = 8;
    block_config.attention_type = ML::Advanced::AttentionType::MULTI_HEAD;
    block_config.dropout_rate = 0.1f;
    block_config.use_layer_norm = true;
    block_config.use_residual = true;
    block_config.seq_len = seq_len;
    
    auto transformer_block = ML::Advanced::AdvancedAttentionFactory::create_transformer_block(block_config);
    
    // Test forward pass
    auto output = transformer_block->forward(test_input);
    
    // Verify output
    ASSERT_GT(output.size(), 0) << "Output should not be empty";
    EXPECT_EQ(output.size(), embed_dim) << "Output should match embedding dimension";
    
    // Verify no NaN or infinite values
    for (float val : output) {
        ASSERT_FALSE(std::isnan(val)) << "NaN detected in output";
        ASSERT_FALSE(std::isinf(val)) << "Inf detected in output";
    }
    
    // Test memory usage
    size_t memory_usage = transformer_block->get_memory_usage();
    std::cout << "Transformer block memory usage: " << memory_usage << " bytes\n";
    EXPECT_LT(memory_usage, 5 * 1024 * 1024) << "Memory usage should be reasonable";
    
    std::cout << "Advanced transformer block: PASS\n";
}

// Test Performance Benchmarks
TEST_F(Phase7AdvancedAttentionTest, PerformanceBenchmarks) {
    std::cout << "\n=== Performance Benchmarks Test ===\n";
#ifndef NDEBUG
    GTEST_SKIP() << "Performance targets require a Release build (-O3).";
#endif
    
    // Benchmark all attention types
    auto results = ML::Advanced::AdvancedAttentionBenchmarks::benchmark_all_attention_types(512, 256);
    
    // Print results
    ML::Advanced::AdvancedAttentionBenchmarks::print_benchmark_results(results);
    
    // Verify performance targets
    for (const auto& result : results) {
        EXPECT_LT(result.latency_ms, 50.0f) << "Latency should be under 50ms for all attention types";
        EXPECT_GT(result.throughput_tokens_per_sec, 25.0f) << "Throughput should be reasonable";
        EXPECT_LT(result.memory_usage_mb, 50.0f) << "Memory usage should be under 50MB";
    }
    
    // Save results
    ML::Advanced::AdvancedAttentionBenchmarks::save_benchmark_results(results, "advanced_attention_benchmarks.csv");
    
    std::cout << "Performance benchmarks: PASS\n";
}

// Test Factory Methods
TEST_F(Phase7AdvancedAttentionTest, FactoryMethods) {
    std::cout << "\n=== Factory Methods Test ===\n";
    
    // Test creating different attention types
    std::vector<ML::Advanced::AttentionType> types = {
        ML::Advanced::AttentionType::MULTI_HEAD,
        ML::Advanced::AttentionType::HIERARCHICAL,
        ML::Advanced::AttentionType::SPARSE,
        ML::Advanced::AttentionType::LINEAR,
        ML::Advanced::AttentionType::LOCAL
    };
    
    for (auto type : types) {
        ML::Advanced::AdvancedAttention::Config config;
        config.embed_dim = embed_dim;
        config.seq_len = seq_len;
        config.attention_type = type;
        
        try {
            auto attention = ML::Advanced::AdvancedAttentionFactory::create_attention(type, config);
            ASSERT_NE(attention, nullptr) << "Factory should create valid attention instance";
            
            auto output = attention->forward(test_input);
            EXPECT_GT(output.size(), 0) << "Created attention should produce valid output";
            
            std::cout << "Successfully created " << static_cast<int>(type) << " attention type\n";
        } catch (const std::exception& e) {
            FAIL() << "Factory failed to create attention type " << static_cast<int>(type) << ": " << e.what();
        }
    }
    
    // Test transformer block creation
    ML::Advanced::AdvancedTransformerBlock::BlockConfig block_config;
    block_config.embed_dim = embed_dim;
    block_config.ff_dim = embed_dim * 4;
    block_config.num_heads = 8;
    block_config.attention_type = ML::Advanced::AttentionType::MULTI_HEAD;
    
    auto transformer_block = ML::Advanced::AdvancedAttentionFactory::create_transformer_block(block_config);
    ASSERT_NE(transformer_block, nullptr) << "Factory should create valid transformer block";
    
    auto block_output = transformer_block->forward(test_input);
    EXPECT_GT(block_output.size(), 0) << "Created transformer block should produce valid output";
    
    std::cout << "Factory methods: PASS\n";
}

// Test Memory Efficiency
TEST_F(Phase7AdvancedAttentionTest, MemoryEfficiency) {
    std::cout << "\n=== Memory Efficiency Test ===\n";
    
    std::vector<std::pair<std::string, std::unique_ptr<ML::Advanced::AdvancedAttention>>> attentions;
    
    // Create different attention types
    ML::Advanced::AdvancedAttention::Config config;
    config.embed_dim = embed_dim;
    config.seq_len = seq_len;
    
    attentions.push_back({"MultiModal", 
        ML::Advanced::AdvancedAttentionFactory::create_attention(
            ML::Advanced::AttentionType::MULTI_HEAD, config)});
    
    attentions.push_back({"Hierarchical", 
        ML::Advanced::AdvancedAttentionFactory::create_attention(
            ML::Advanced::AttentionType::HIERARCHICAL, config)});
    
    attentions.push_back({"Sparse", 
        ML::Advanced::AdvancedAttentionFactory::create_attention(
            ML::Advanced::AttentionType::SPARSE, config)});
    
    attentions.push_back({"Linear", 
        ML::Advanced::AdvancedAttentionFactory::create_attention(
            ML::Advanced::AttentionType::LINEAR, config)});
    
    attentions.push_back({"Local", 
        ML::Advanced::AdvancedAttentionFactory::create_attention(
            ML::Advanced::AttentionType::LOCAL, config)});
    
    // Compare memory usage
    std::cout << "Memory usage comparison:\n";
    for (const auto& [name, attention] : attentions) {
        size_t memory_usage = attention->get_memory_usage();
        float memory_mb = static_cast<float>(memory_usage) / (1024 * 1024);
        std::cout << "  " << name << ": " << memory_mb << "MB\n";
        
        EXPECT_LT(memory_usage, 10 * 1024 * 1024) << "Memory usage should be under 10MB";
    }
    
    std::cout << "Memory efficiency: PASS\n";
}
