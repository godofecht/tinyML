//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Generative Models Performance Benchmarks for TinyML
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 21/01/2026
*****************************************************************************/

#include <gtest/gtest.h>
#include "GenerativeModels.h"
#include <chrono>
#include <vector>
#include <iomanip>
#include <sstream>

using namespace ML::Generative;

class GenerativeModelsBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random test data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Create test datasets of different sizes
        small_data_.resize(10);
        medium_data_.resize(100);
        large_data_.resize(1000);
        
        for (auto& val : small_data_) val = dis(gen);
        for (auto& val : medium_data_) val = dis(gen);
        for (auto& val : large_data_) val = dis(gen);
        
        condition_ = {1.0f, 0.5f, 0.0f};
    }
    
    template<typename Func>
    double benchmark_function(Func&& func, int iterations = 100) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return static_cast<double>(duration.count()) / iterations;
    }
    
    Tensor small_data_;
    Tensor medium_data_;
    Tensor large_data_;
    Tensor condition_;
};

// =============================================================================
// VAE BENCHMARKS
// =============================================================================

TEST_F(GenerativeModelsBenchmark, VAESmallDataBenchmark) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 4;
    config.hidden_dim = 32;
    
    VAE vae(config);
    
    double encode_time = benchmark_function([&]() {
        Tensor mu, logvar;
        vae.encode(small_data_, mu, logvar);
    }, 1000);
    
    double decode_time = benchmark_function([&]() {
        Tensor latent = {0.1f, 0.2f, 0.3f, 0.4f};
        Tensor output;
        vae.decode(latent, output);
    }, 1000);
    
    double forward_time = benchmark_function([&]() {
        Tensor output, mu, logvar;
        vae.forward(small_data_, output, mu, logvar);
    }, 1000);
    
    double train_time = benchmark_function([&]() {
        vae.train_step(small_data_);
    }, 500);
    
    std::cout << "\n=== VAE Small Data (10 dim) Benchmark ===" << std::endl;
    std::cout << "Encode: " << std::fixed << std::setprecision(3) << encode_time << " μs" << std::endl;
    std::cout << "Decode: " << decode_time << " μs" << std::endl;
    std::cout << "Forward: " << forward_time << " μs" << std::endl;
    std::cout << "Train: " << train_time << " μs" << std::endl;
    
    // Performance targets
    EXPECT_LT(encode_time, 100.0);    // < 100μs for encode
    EXPECT_LT(decode_time, 100.0);    // < 100μs for decode
    EXPECT_LT(forward_time, 200.0);  // < 200μs for forward
    EXPECT_LT(train_time, 500.0);     // < 500μs for training step
}

TEST_F(GenerativeModelsBenchmark, VAEMediumDataBenchmark) {
    VAE::Config config;
    config.input_dim = 100;
    config.latent_dim = 16;
    config.hidden_dim = 64;
    
    VAE vae(config);
    
    double encode_time = benchmark_function([&]() {
        Tensor mu, logvar;
        vae.encode(medium_data_, mu, logvar);
    }, 500);
    
    double decode_time = benchmark_function([&]() {
        Tensor latent(16, 0.1f);
        Tensor output;
        vae.decode(latent, output);
    }, 500);
    
    double forward_time = benchmark_function([&]() {
        Tensor output, mu, logvar;
        vae.forward(medium_data_, output, mu, logvar);
    }, 500);
    
    double train_time = benchmark_function([&]() {
        vae.train_step(medium_data_);
    }, 200);
    
    std::cout << "\n=== VAE Medium Data (100 dim) Benchmark ===" << std::endl;
    std::cout << "Encode: " << std::fixed << std::setprecision(3) << encode_time << " μs" << std::endl;
    std::cout << "Decode: " << decode_time << " μs" << std::endl;
    std::cout << "Forward: " << forward_time << " μs" << std::endl;
    std::cout << "Train: " << train_time << " μs" << std::endl;
    
    // Performance targets for medium data
    EXPECT_LT(encode_time, 500.0);    // < 500μs for encode
    EXPECT_LT(decode_time, 500.0);    // < 500μs for decode
    EXPECT_LT(forward_time, 1000.0); // < 1ms for forward
    EXPECT_LT(train_time, 2000.0);    // < 2ms for training step
}

TEST_F(GenerativeModelsBenchmark, VAELargeDataBenchmark) {
    VAE::Config config;
    config.input_dim = 1000;
    config.latent_dim = 32;
    config.hidden_dim = 128;
    
    VAE vae(config);
    
    double encode_time = benchmark_function([&]() {
        Tensor mu, logvar;
        vae.encode(large_data_, mu, logvar);
    }, 100);
    
    double decode_time = benchmark_function([&]() {
        Tensor latent(32, 0.1f);
        Tensor output;
        vae.decode(latent, output);
    }, 100);
    
    double forward_time = benchmark_function([&]() {
        Tensor output, mu, logvar;
        vae.forward(large_data_, output, mu, logvar);
    }, 100);
    
    double train_time = benchmark_function([&]() {
        vae.train_step(large_data_);
    }, 50);
    
    std::cout << "\n=== VAE Large Data (1000 dim) Benchmark ===" << std::endl;
    std::cout << "Encode: " << std::fixed << std::setprecision(3) << encode_time << " μs" << std::endl;
    std::cout << "Decode: " << decode_time << " μs" << std::endl;
    std::cout << "Forward: " << forward_time << " μs" << std::endl;
    std::cout << "Train: " << train_time << " μs" << std::endl;
    
    // Performance targets for large data
    EXPECT_LT(encode_time, 5000.0);   // < 5ms for encode
    EXPECT_LT(decode_time, 5000.0);   // < 5ms for decode
    EXPECT_LT(forward_time, 10000.0); // < 10ms for forward
    EXPECT_LT(train_time, 20000.0);   // < 20ms for training step
}

// =============================================================================
// GAN BENCHMARKS
// =============================================================================

TEST_F(GenerativeModelsBenchmark, GANSmallDataBenchmark) {
    GAN::Config config;
    config.latent_dim = 8;
    config.input_dim = 10;
    config.hidden_dim = 32;
    
    GAN gan(config);
    
    double gen_time = benchmark_function([&]() {
        Tensor noise(8, 0.1f);
        Tensor output;
        gan.generate(noise, {}, output);
    }, 1000);
    
    double disc_time = benchmark_function([&]() {
        gan.discriminate(small_data_, {});
    }, 1000);
    
    double train_time = benchmark_function([&]() {
        std::vector<Tensor> real_data = {small_data_};
        gan.train_step(real_data, {});
    }, 200);
    
    std::cout << "\n=== GAN Small Data (10 dim) Benchmark ===" << std::endl;
    std::cout << "Generate: " << std::fixed << std::setprecision(3) << gen_time << " μs" << std::endl;
    std::cout << "Discriminate: " << disc_time << " μs" << std::endl;
    std::cout << "Train: " << train_time << " μs" << std::endl;
    
    // Performance targets
    EXPECT_LT(gen_time, 100.0);     // < 100μs for generation
    EXPECT_LT(disc_time, 50.0);      // < 50μs for discrimination
    EXPECT_LT(train_time, 1000.0);   // < 1ms for training step
}

TEST_F(GenerativeModelsBenchmark, GANMediumDataBenchmark) {
    GAN::Config config;
    config.latent_dim = 16;
    config.input_dim = 100;
    config.hidden_dim = 64;
    
    GAN gan(config);
    
    double gen_time = benchmark_function([&]() {
        Tensor noise(16, 0.1f);
        Tensor output;
        gan.generate(noise, {}, output);
    }, 500);
    
    double disc_time = benchmark_function([&]() {
        gan.discriminate(medium_data_, {});
    }, 500);
    
    double train_time = benchmark_function([&]() {
        std::vector<Tensor> real_data = {medium_data_};
        gan.train_step(real_data, {});
    }, 100);
    
    std::cout << "\n=== GAN Medium Data (100 dim) Benchmark ===" << std::endl;
    std::cout << "Generate: " << std::fixed << std::setprecision(3) << gen_time << " μs" << std::endl;
    std::cout << "Discriminate: " << disc_time << " μs" << std::endl;
    std::cout << "Train: " << train_time << " μs" << std::endl;
    
    // Performance targets for medium data
    EXPECT_LT(gen_time, 500.0);      // < 500μs for generation
    EXPECT_LT(disc_time, 200.0);     // < 200μs for discrimination
    EXPECT_LT(train_time, 5000.0);   // < 5ms for training step
}

TEST_F(GenerativeModelsBenchmark, GANBatchGenerationBenchmark) {
    GAN::Config config;
    config.latent_dim = 16;
    config.input_dim = 100;
    config.hidden_dim = 64;
    
    GAN gan(config);
    
    double batch_gen_time = benchmark_function([&]() {
        std::vector<Tensor> outputs;
        gan.generate_batch(10, {}, outputs);
    }, 100);
    
    std::cout << "\n=== GAN Batch Generation (10 samples) Benchmark ===" << std::endl;
    std::cout << "Batch Generate: " << std::fixed << std::setprecision(3) << batch_gen_time << " μs" << std::endl;
    
    // Performance targets for batch generation
    EXPECT_LT(batch_gen_time, 2000.0); // < 2ms for 10 samples
}

// =============================================================================
// DIFFUSION MODEL BENCHMARKS
// =============================================================================

TEST_F(GenerativeModelsBenchmark, DiffusionSmallDataBenchmark) {
    DiffusionModel::Config config;
    config.input_dim = 10;
    config.hidden_dim = 32;
    config.timesteps = 50;  // Reduced for benchmarking
    
    DiffusionModel diffusion(config);
    
    double forward_time = benchmark_function([&]() {
        Tensor xt, epsilon;
        diffusion.forward_diffusion(small_data_, 25, xt, epsilon);
    }, 500);
    
    double predict_time = benchmark_function([&]() {
        Tensor xt = small_data_;
        Tensor epsilon_pred;
        diffusion.predict_noise(xt, 25, {}, epsilon_pred);
    }, 500);
    
    double train_time = benchmark_function([&]() {
        diffusion.train_step(small_data_, {});
    }, 200);
    
    std::cout << "\n=== Diffusion Small Data (10 dim, 50 steps) Benchmark ===" << std::endl;
    std::cout << "Forward Process: " << std::fixed << std::setprecision(3) << forward_time << " μs" << std::endl;
    std::cout << "Predict Noise: " << predict_time << " μs" << std::endl;
    std::cout << "Train: " << train_time << " μs" << std::endl;
    
    // Performance targets
    EXPECT_LT(forward_time, 100.0);   // < 100μs for forward process
    EXPECT_LT(predict_time, 200.0);    // < 200μs for noise prediction
    EXPECT_LT(train_time, 500.0);      // < 500μs for training step
}

TEST_F(GenerativeModelsBenchmark, DiffusionSamplingBenchmark) {
    DiffusionModel::Config config;
    config.input_dim = 10;
    config.hidden_dim = 32;
    config.timesteps = 20;  // Very reduced for sampling benchmark
    
    DiffusionModel diffusion(config);
    
    double sample_time = benchmark_function([&]() {
        Tensor output;
        diffusion.reverse_diffusion({}, output);
    }, 10);
    
    std::cout << "\n=== Diffusion Sampling (10 dim, 20 steps) Benchmark ===" << std::endl;
    std::cout << "Full Sample: " << std::fixed << std::setprecision(3) << sample_time << " μs" << std::endl;
    
    // Performance targets for sampling (should be under 50ms for target)
    EXPECT_LT(sample_time, 50000.0);  // < 50ms for full sampling
}

// =============================================================================
// MEMORY USAGE BENCHMARKS
// =============================================================================

TEST_F(GenerativeModelsBenchmark, MemoryUsageBenchmark) {
    // Test memory usage for different model sizes
    std::vector<size_t> input_dims = {10, 100, 1000};
    std::vector<size_t> latent_dims = {4, 16, 32};
    
    std::cout << "\n=== Memory Usage Benchmark ===" << std::endl;
    std::cout << std::setw(12) << "Input Dim" 
              << std::setw(12) << "Latent Dim" 
              << std::setw(15) << "VAE Memory (KB)" 
              << std::setw(15) << "GAN Memory (KB)"
              << std::setw(18) << "Diffusion Memory (KB)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (size_t i = 0; i < input_dims.size(); ++i) {
        // VAE memory estimation
        VAE::Config vae_config;
        vae_config.input_dim = input_dims[i];
        vae_config.latent_dim = latent_dims[i];
        vae_config.hidden_dim = std::min(input_dims[i] * 2, 256UL);
        
        size_t vae_memory = (vae_config.input_dim * vae_config.hidden_dim + 
                           vae_config.hidden_dim * vae_config.latent_dim * 2 +
                           vae_config.latent_dim * vae_config.hidden_dim * 2 +
                           vae_config.hidden_dim * vae_config.input_dim) * sizeof(float) / 1024;
        
        // GAN memory estimation
        GAN::Config gan_config;
        gan_config.latent_dim = latent_dims[i];
        gan_config.input_dim = input_dims[i];
        gan_config.hidden_dim = std::min(input_dims[i] * 2, 256UL);
        
        size_t gan_memory = (gan_config.latent_dim * gan_config.hidden_dim * 3 +
                           gan_config.input_dim * gan_config.hidden_dim * 3 +
                           gan_config.hidden_dim * 1) * sizeof(float) / 1024;
        
        // Diffusion memory estimation
        DiffusionModel::Config diff_config;
        diff_config.input_dim = input_dims[i];
        diff_config.hidden_dim = std::min(input_dims[i] * 2, 256UL);
        diff_config.timesteps = 100;
        
        size_t diff_memory = (diff_config.input_dim * diff_config.hidden_dim * 3 +
                            128 * diff_config.hidden_dim) * sizeof(float) / 1024;
        
        std::cout << std::setw(12) << input_dims[i]
                  << std::setw(12) << latent_dims[i]
                  << std::setw(15) << std::fixed << std::setprecision(1) << vae_memory
                  << std::setw(15) << gan_memory
                  << std::setw(18) << diff_memory << std::endl;
        
        // Memory targets (should be reasonable for edge devices)
        EXPECT_LT(vae_memory, 4096);   // < 4MB for VAE
        EXPECT_LT(gan_memory, 4096);   // < 4MB for GAN
        EXPECT_LT(diff_memory, 4096); // < 4MB for Diffusion
    }
}

// =============================================================================
// THROUGHPUT BENCHMARKS
// =============================================================================

TEST_F(GenerativeModelsBenchmark, ThroughputBenchmark) {
    VAE::Config config;
    config.input_dim = 100;
    config.latent_dim = 16;
    config.hidden_dim = 64;
    
    VAE vae(config);
    
    // Test different batch sizes
    std::vector<int> batch_sizes = {1, 10, 50, 100};
    
    std::cout << "\n=== VAE Throughput Benchmark ===" << std::endl;
    std::cout << std::setw(12) << "Batch Size" 
              << std::setw(15) << "Total Time (ms)" 
              << std::setw(15) << "Per Sample (μs)" 
              << std::setw(15) << "Samples/sec" << std::endl;
    std::cout << std::string(57, '-') << std::endl;
    
    for (int batch_size : batch_sizes) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < batch_size; ++i) {
            Tensor output, mu, logvar;
            vae.forward(medium_data_, output, mu, logvar);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double per_sample = static_cast<double>(total_time.count()) / batch_size;
        double samples_per_sec = 1000000.0 / per_sample;
        
        std::cout << std::setw(12) << batch_size
                  << std::setw(15) << std::fixed << std::setprecision(3) << total_time.count() / 1000.0
                  << std::setw(15) << per_sample
                  << std::setw(15) << std::setprecision(0) << samples_per_sec << std::endl;
        
        // Throughput targets
        EXPECT_GT(samples_per_sec, 100);  // > 100 samples per second
    }
}

// =============================================================================
// CONDITIONAL MODELS BENCHMARK
// =============================================================================

TEST_F(GenerativeModelsBenchmark, ConditionalModelsBenchmark) {
    // Conditional VAE
    VAE::Config vae_config;
    vae_config.input_dim = 100;
    vae_config.latent_dim = 16;
    vae_config.hidden_dim = 64;
    vae_config.conditional = true;
    vae_config.condition_dim = 3;
    
    VAE vae(vae_config);
    
    double vae_cond_train_time = benchmark_function([&]() {
        vae.train_step(medium_data_, condition_);
    }, 200);
    
    double vae_cond_gen_time = benchmark_function([&]() {
        Tensor output;
        vae.generate(condition_, output);
    }, 500);
    
    // Conditional GAN
    GAN::Config gan_config;
    gan_config.latent_dim = 16;
    gan_config.input_dim = 100;
    gan_config.hidden_dim = 64;
    gan_config.conditional = true;
    gan_config.condition_dim = 3;
    
    GAN gan(gan_config);
    
    double gan_cond_train_time = benchmark_function([&]() {
        std::vector<Tensor> real_data = {medium_data_};
        gan.train_step(real_data, condition_);
    }, 100);
    
    double gan_cond_gen_time = benchmark_function([&]() {
        Tensor noise(16, 0.1f);
        Tensor output;
        gan.generate(noise, condition_, output);
    }, 500);
    
    std::cout << "\n=== Conditional Models Benchmark ===" << std::endl;
    std::cout << "VAE Conditional Train: " << std::fixed << std::setprecision(3) << vae_cond_train_time << " μs" << std::endl;
    std::cout << "VAE Conditional Generate: " << vae_cond_gen_time << " μs" << std::endl;
    std::cout << "GAN Conditional Train: " << gan_cond_train_time << " μs" << std::endl;
    std::cout << "GAN Conditional Generate: " << gan_cond_gen_time << " μs" << std::endl;
    
    // Conditional models should not be significantly slower
    EXPECT_LT(vae_cond_train_time, 3000.0);  // < 3ms
    EXPECT_LT(vae_cond_gen_time, 1000.0);   // < 1ms
    EXPECT_LT(gan_cond_train_time, 10000.0); // < 10ms
    EXPECT_LT(gan_cond_gen_time, 1000.0);    // < 1ms
}

// =============================================================================
// COMPREHENSIVE PERFORMANCE SUMMARY
// =============================================================================

TEST_F(GenerativeModelsBenchmark, PerformanceSummary) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "GENERATIVE MODELS PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Test all models with standard configuration
    VAE::Config vae_config;
    vae_config.input_dim = 100;
    vae_config.latent_dim = 16;
    vae_config.hidden_dim = 64;
    
    GAN::Config gan_config;
    gan_config.latent_dim = 16;
    gan_config.input_dim = 100;
    gan_config.hidden_dim = 64;
    
    DiffusionModel::Config diff_config;
    diff_config.input_dim = 100;
    diff_config.hidden_dim = 64;
    diff_config.timesteps = 50;
    
    VAE vae(vae_config);
    GAN gan(gan_config);
    DiffusionModel diffusion(diff_config);
    
    // Benchmark each model
    double vae_time = benchmark_function([&]() {
        Tensor output, mu, logvar;
        vae.forward(medium_data_, output, mu, logvar);
    }, 500);
    
    double gan_gen_time = benchmark_function([&]() {
        Tensor noise(16, 0.1f);
        Tensor output;
        gan.generate(noise, {}, output);
    }, 500);
    
    double diff_train_time = benchmark_function([&]() {
        diffusion.train_step(medium_data_, {});
    }, 200);
    
    std::cout << "Model Configuration: 100D input, 16D latent, 64D hidden" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "VAE Forward Pass:     " << std::fixed << std::setprecision(3) << vae_time << " μs" << std::endl;
    std::cout << "GAN Generation:        " << gan_gen_time << " μs" << std::endl;
    std::cout << "Diffusion Training:    " << diff_train_time << " μs" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // Overall performance targets
    EXPECT_LT(vae_time, 2000.0);        // < 2ms for VAE
    EXPECT_LT(gan_gen_time, 1000.0);    // < 1ms for GAN generation
    EXPECT_LT(diff_train_time, 2000.0); // < 2ms for diffusion training
    
    std::cout << "\n✅ All performance targets met!" << std::endl;
    std::cout << "🚀 Generative models ready for real-time applications!" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
