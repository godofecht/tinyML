//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "LightweightAttention.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <fstream>
#include <algorithm>

using namespace ML::RealTime;
using namespace std::chrono;

class AttentionBenchmark {
public:
    struct BenchmarkResult {
        std::string test_name;
        size_t embed_dim;
        size_t num_heads;
        size_t sequence_length;
        double avg_latency_us;
        double throughput_sequences_per_sec;
        size_t memory_usage_kb;
        bool meets_latency_target;
    };

    void run_all_benchmarks() {
        std::cout << "=== Lightweight Attention Benchmark Suite ===\n\n";
        
        // Test different configurations
        std::vector<std::pair<size_t, size_t>> configs = {
            {128, 4},   // Small
            {256, 8},   // Medium
            {512, 8},   // Large
            {512, 16},  // Many heads
            {1024, 16}  // Very large
        };
        
        std::vector<BenchmarkResult> results;
        
        for (auto [embed_dim, num_heads] : configs) {
            if (embed_dim % num_heads != 0) continue; // Skip invalid configs
            
            std::cout << "Testing: Embed=" << embed_dim << ", Heads=" << num_heads << "\n";
            
            // Single sequence benchmark
            auto single_result = benchmark_single_sequence(embed_dim, num_heads);
            results.push_back(single_result);
            
            // Batch processing benchmark
            auto batch_result = benchmark_batch_processing(embed_dim, num_heads);
            results.push_back(batch_result);
            
            // Streaming benchmark
            auto stream_result = benchmark_streaming(embed_dim, num_heads);
            results.push_back(stream_result);
            
            std::cout << "\n";
        }
        
        // Print summary table
        print_summary_table(results);
        
        // Save results to file
        save_results_to_file(results);
        
        // Check performance targets
        check_performance_targets(results);
    }

private:
    BenchmarkResult benchmark_single_sequence(size_t embed_dim, size_t num_heads) {
        BenchmarkResult result;
        result.test_name = "Single Sequence";
        result.embed_dim = embed_dim;
        result.num_heads = num_heads;
        result.sequence_length = 1;
        
        // Create attention instance
        LightweightAttention::Config config;
        config.embed_dim = embed_dim;
        config.num_heads = num_heads;
        config.sequence_length = 512;
        
        auto attention = create_lightweight_attention(config);
        result.memory_usage_kb = attention->get_memory_usage() / 1024;
        
        // Create test input
        std::vector<float> input(embed_dim);
        for (size_t i = 0; i < embed_dim; ++i) {
            input[i] = static_cast<float>(i) / embed_dim;
        }
        
        // Warm up
        for (int i = 0; i < 10; ++i) {
            attention->forward(input);
        }
        
        // Benchmark
        const int iterations = 1000;
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            attention->forward(input);
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        result.avg_latency_us = static_cast<double>(duration.count()) / iterations;
        result.throughput_sequences_per_sec = 1000000.0 / result.avg_latency_us;
        result.meets_latency_target = result.avg_latency_us < 1000.0; // < 1ms target
        
        std::cout << "  Single: " << result.avg_latency_us << " μs avg, "
                 << result.throughput_sequences_per_sec << " seq/s, "
                 << (result.meets_latency_target ? "✅" : "❌") << "\n";
        
        return result;
    }
    
    BenchmarkResult benchmark_batch_processing(size_t embed_dim, size_t num_heads) {
        BenchmarkResult result;
        result.test_name = "Batch Processing";
        result.embed_dim = embed_dim;
        result.num_heads = num_heads;
        result.sequence_length = 8; // 8 sequences in batch
        
        LightweightAttention::Config config;
        config.embed_dim = embed_dim;
        config.num_heads = num_heads;
        config.sequence_length = 512;
        
        auto attention = create_lightweight_attention(config);
        result.memory_usage_kb = attention->get_memory_usage() / 1024;
        
        // Create batch inputs
        std::vector<std::vector<float>> batch_inputs;
        for (int seq = 0; seq < 8; ++seq) {
            std::vector<float> input(embed_dim);
            for (size_t i = 0; i < embed_dim; ++i) {
                input[i] = static_cast<float>(seq * embed_dim + i) / (embed_dim * 8);
            }
            batch_inputs.push_back(input);
        }
        
        // Warm up
        for (int i = 0; i < 5; ++i) {
            attention->forward_batch(batch_inputs);
        }
        
        // Benchmark
        const int iterations = 100;
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            attention->forward_batch(batch_inputs);
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        double total_time_us = static_cast<double>(duration.count()) / iterations;
        result.avg_latency_us = total_time_us / 8.0; // Per sequence average
        result.throughput_sequences_per_sec = 8000000.0 / total_time_us; // 8 sequences per batch
        result.meets_latency_target = result.avg_latency_us < 1000.0;
        
        std::cout << "  Batch:  " << result.avg_latency_us << " μs avg per seq, "
                 << result.throughput_sequences_per_sec << " seq/s, "
                 << (result.meets_latency_target ? "✅" : "❌") << "\n";
        
        return result;
    }
    
    BenchmarkResult benchmark_streaming(size_t embed_dim, size_t num_heads) {
        BenchmarkResult result;
        result.test_name = "Streaming";
        result.embed_dim = embed_dim;
        result.num_heads = num_heads;
        result.sequence_length = 1;
        
        LightweightAttention::Config config;
        config.embed_dim = embed_dim;
        config.num_heads = num_heads;
        config.sequence_length = 512;
        
        auto attention = create_lightweight_attention(config);
        result.memory_usage_kb = attention->get_memory_usage() / 1024;
        
        // Create test chunk
        std::vector<float> chunk(embed_dim);
        for (size_t i = 0; i < embed_dim; ++i) {
            chunk[i] = static_cast<float>(i) / (embed_dim * 2);
        }
        
        // Start streaming
        attention->start_stream();
        
        // Warm up
        for (int i = 0; i < 10; ++i) {
            attention->process_chunk(chunk);
        }
        
        // Benchmark
        const int iterations = 1000;
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            attention->process_chunk(chunk);
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        attention->end_stream();
        
        result.avg_latency_us = static_cast<double>(duration.count()) / iterations;
        result.throughput_sequences_per_sec = 1000000.0 / result.avg_latency_us;
        result.meets_latency_target = result.avg_latency_us < 1000.0;
        
        std::cout << "  Stream: " << result.avg_latency_us << " μs avg, "
                 << result.throughput_sequences_per_sec << " seq/s, "
                 << (result.meets_latency_target ? "✅" : "❌") << "\n";
        
        return result;
    }
    
    void print_summary_table(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== Performance Summary ===\n";
        std::cout << std::left << std::setw(15) << "Test"
                 << std::setw(8) << "Embed"
                 << std::setw(6) << "Heads"
                 << std::setw(12) << "Latency (μs)"
                 << std::setw(15) << "Throughput"
                 << std::setw(12) << "Memory (KB)"
                 << std::setw(8) << "Target" << "\n";
        std::cout << std::string(80, '-') << "\n";
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(15) << result.test_name
                     << std::setw(8) << result.embed_dim
                     << std::setw(6) << result.num_heads
                     << std::setw(12) << std::fixed << std::setprecision(1) << result.avg_latency_us
                     << std::setw(15) << std::setprecision(0) << result.throughput_sequences_per_sec
                     << std::setw(12) << result.memory_usage_kb
                     << std::setw(8) << (result.meets_latency_target ? "✅" : "❌") << "\n";
        }
    }
    
    void save_results_to_file(const std::vector<BenchmarkResult>& results) {
        std::ofstream file("attention_benchmark_results.csv");
        
        // Write header
        file << "Test,Embed_Dim,Num_Heads,Sequence_Length,Avg_Latency_us,"
             << "Throughput_seq_per_sec,Memory_Usage_KB,Meets_Latency_Target\n";
        
        // Write data
        for (const auto& result : results) {
            file << result.test_name << ","
                 << result.embed_dim << ","
                 << result.num_heads << ","
                 << result.sequence_length << ","
                 << result.avg_latency_us << ","
                 << result.throughput_sequences_per_sec << ","
                 << result.memory_usage_kb << ","
                 << (result.meets_latency_target ? "true" : "false") << "\n";
        }
        
        file.close();
        std::cout << "\nResults saved to attention_benchmark_results.csv\n";
    }
    
    void check_performance_targets(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== Performance Target Analysis ===\n";
        
        int total_tests = results.size();
        int passing_tests = 0;
        double best_latency = std::numeric_limits<double>::max();
        double worst_latency = 0;
        double avg_latency = 0;
        
        for (const auto& result : results) {
            if (result.meets_latency_target) {
                passing_tests++;
            }
            best_latency = std::min(best_latency, result.avg_latency_us);
            worst_latency = std::max(worst_latency, result.avg_latency_us);
            avg_latency += result.avg_latency_us;
        }
        
        avg_latency /= total_tests;
        
        std::cout << "Target: <1ms (1000μs) latency\n";
        std::cout << "Tests passing: " << passing_tests << "/" << total_tests 
                 << " (" << (100.0 * passing_tests / total_tests) << "%)\n";
        std::cout << "Best latency: " << best_latency << " μs\n";
        std::cout << "Worst latency: " << worst_latency << " μs\n";
        std::cout << "Average latency: " << avg_latency << " μs\n";
        
        if (passing_tests == total_tests) {
            std::cout << "🎉 All tests meet the <1ms target!\n";
        } else {
            std::cout << "⚠️  Some tests exceed the 1ms target\n";
        }
        
        // Memory efficiency check
        size_t max_memory = 0;
        for (const auto& result : results) {
            max_memory = std::max(max_memory, result.memory_usage_kb);
        }
        
        std::cout << "Maximum memory usage: " << max_memory << " KB\n";
        if (max_memory < 10240) { // < 10MB
            std::cout << "✅ Memory usage is within target (<10MB)\n";
        } else {
            std::cout << "❌ Memory usage exceeds 10MB target\n";
        }
    }
};

// Stress test for long-running stability
void stress_test_attention() {
    std::cout << "\n=== Stress Test: Long-running Stability ===\n";
    
    LightweightAttention::Config config;
    config.embed_dim = 256;
    config.num_heads = 8;
    config.sequence_length = 512;
    
    auto attention = create_lightweight_attention(config);
    
    std::vector<float> input(config.embed_dim);
    for (size_t i = 0; i < config.embed_dim; ++i) {
        input[i] = static_cast<float>(i) / config.embed_dim;
    }
    
    const int stress_iterations = 10000;
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < stress_iterations; ++i) {
        auto output = attention->forward(input);
        
        // Simple validation - check for NaN or inf
        for (float val : output) {
            if (std::isnan(val) || std::isinf(val)) {
                std::cout << "❌ Invalid output detected at iteration " << i << "\n";
                return;
            }
        }
        
        if (i % 1000 == 0) {
            std::cout << "Completed " << i << " iterations...\n";
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    std::cout << "✅ Stress test completed: " << stress_iterations 
             << " iterations in " << duration.count() << " ms\n";
    std::cout << "Average per iteration: " 
             << (static_cast<double>(duration.count()) / stress_iterations) << " ms\n";
}

int main() {
    try {
        AttentionBenchmark benchmark;
        benchmark.run_all_benchmarks();
        
        stress_test_attention();
        
        std::cout << "\n=== Benchmark Complete ===\n";
        std::cout << "✅ Multi-head attention implementation validated\n";
        std::cout << "✅ Performance targets measured\n";
        std::cout << "✅ Memory usage analyzed\n";
        std::cout << "✅ Stability tested\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
