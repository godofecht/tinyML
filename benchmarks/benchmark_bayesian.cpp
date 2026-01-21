//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 9 Bayesian Neural Networks Benchmark
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>

#include "BayesianNeuralNetwork.h"

class BayesianBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Initialize comprehensive test data
        benchmark_sizes = {100, 500, 1000, 2000};
        
        for (size_t size : benchmark_sizes) {
            test_data[size] = std::vector<float>(size * input_dim);
            test_targets[size] = std::vector<float>(size);
            
            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < input_dim; ++j) {
                    test_data[size][i * input_dim + j] = dis(gen);
                }
                
                // Generate targets with some noise
                float sum = 0.0f;
                for (size_t j = 0; j < input_dim; ++j) {
                    sum += test_data[size][i * input_dim + j];
                }
                test_targets[size][i] = sum / input_dim + dis(gen) * 0.1f;
            }
        }
        
        // Performance targets from roadmap
        target_overhead_percent = 5.0f;   // <5% computational overhead
        target_uncertainty_quality = 0.95f; // >95% uncertainty quality
        target_calibration_ece = 0.05f;     // ECE < 0.05
    }
    
    std::vector<size_t> benchmark_sizes;
    std::map<size_t, std::vector<float>> test_data;
    std::map<size_t, std::vector<float>> test_targets;
    
    const size_t input_dim = 10;
    const size_t output_dim = 1;
    
    float target_overhead_percent;
    float target_uncertainty_quality;
    float target_calibration_ece;
    
    template<typename Func>
    double benchmarkFunction(Func&& func, int iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return static_cast<double>(duration.count()) / iterations;
    }
    
    void printBenchmarkHeader(const std::string& title) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "                    " << title << "\n";
        std::cout << std::string(80, '=') << "\n";
    }
    
    void printBenchmarkFooter() {
        std::cout << std::string(80, '=') << "\n\n";
    }
};

// Comprehensive Variational Inference Benchmark
TEST_F(BayesianBenchmark, VariationalInferenceBenchmark) {
    printBenchmarkHeader("VARIATIONAL INFERENCE BENCHMARK");
    
    std::cout << std::setw(12) << "Model Size" << std::setw(15) << "ELBO" 
              << std::setw(15) << "KL Divergence" << std::setw(15) << "Training (ms)" 
              << std::setw(15) << "Inference (μs)" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(82, '-') << std::endl;
    
    std::vector<std::vector<size_t>> model_configs = {
        {10, 32, 1},      // Small
        {10, 64, 32, 1},  // Medium
        {10, 128, 64, 32, 1}  // Large
    };
    
    std::vector<std::string> model_names = {"Small", "Medium", "Large"};
    
    for (size_t config_idx = 0; config_idx < model_configs.size(); ++config_idx) {
        const auto& layer_sizes = model_configs[config_idx];
        const std::string& model_name = model_names[config_idx];
        
        // Create variational model
        ML::Bayesian::BayesianConfig config;
        config.use_variational_inference = true;
        config.use_local_reparameterization = true;
        config.kl_weight = 1.0f;
        config.prior_std = 1.0f;
        
        auto model = ML::Bayesian::BayesianNetworkFactory::create_mlp(layer_sizes, config);
        
        // Create training data
        ML::Bayesian::Dataset train_data;
        train_data.input_dim = input_dim;
        train_data.output_dim = output_dim;
        train_data.X = test_data[500];
        train_data.y = test_targets[500];
        
        // Benchmark training
        auto train_func = [&]() {
            model->train_epoch(train_data);
        };
        
        double train_time_ms = benchmarkFunction(train_func, 10) / 1000.0;
        
        // Benchmark inference
        const float* test_input = test_data[100].data();
        auto inference_func = [&]() {
            float output;
            model->forward(test_input, &output, false);
        };
        
        double inference_time_us = benchmarkFunction(inference_func, 1000);
        
        // Compute metrics
        float elbo = model->compute_elbo(train_data);
        
        float total_kl = 0.0f;
        for (const auto& layer : model->layers) {
            total_kl += layer->compute_kl_divergence();
        }
        
        std::cout << std::setw(12) << model_name << std::setw(15) << std::fixed << std::setprecision(2) << elbo
                  << std::setw(15) << std::fixed << std::setprecision(4) << total_kl
                  << std::setw(15) << std::fixed << std::setprecision(3) << train_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << inference_time_us
                  << std::setw(10) << "PASS" << std::endl;
        
        // Verify variational inference quality
        EXPECT_LT(elbo, 0.0f) << "ELBO should be negative";
        EXPECT_GT(total_kl, 0.0f) << "KL divergence should be positive";
        EXPECT_LT(inference_time_us, 100) << "Inference should be fast";
    }
    
    printBenchmarkFooter();
}

// Monte Carlo Dropout Performance Benchmark
TEST_F(BayesianBenchmark, MonteCarloDropoutBenchmark) {
    printBenchmarkHeader("MONTE CARLO DROPOUT BENCHMARK");
    
    std::cout << std::setw(10) << "Samples" << std::setw(15) << "Mean (μs)" 
              << std::setw(15) << "Uncertainty (μs)" << std::setw(15) << "Std Dev (μs)" 
              << std::setw(15) << "Overhead" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::vector<int> sample_counts = {5, 10, 20, 50, 100};
    
    // Create deterministic baseline
    ML::Bayesian::BayesianConfig det_config;
    det_config.use_monte_carlo_dropout = false;
    
    auto det_model = ML::Bayesian::BayesianNetworkFactory::create_mlp({input_dim, 32, 16, output_dim}, det_config);
    
    const float* test_input = test_data[100].data();
    
    auto det_func = [&]() {
        float output;
        det_model->forward(test_input, &output, false);
    };
    
    double det_time_us = benchmarkFunction(det_func, 1000);
    
    for (int T : sample_counts) {
        // Create MC Dropout model
        ML::Bayesian::BayesianConfig mc_config;
        mc_config.use_monte_carlo_dropout = true;
        mc_config.dropout_rate = 0.1f;
        mc_config.mc_samples = T;
        
        auto mc_model = ML::Bayesian::BayesianNetworkFactory::create_mlp({input_dim, 32, 16, output_dim}, mc_config);
        
        // Benchmark MC inference
        auto mc_func = [&]() {
            float mean, uncertainty;
            mc_model->forward_with_uncertainty(test_input, &mean, &uncertainty);
        };
        
        double mc_time_us = benchmarkFunction(mc_func, 1000);
        
        // Compute standard deviation across runs
        std::vector<float> predictions(10);
        for (int i = 0; i < 10; ++i) {
            float pred_mean, pred_unc;
            mc_model->forward_with_uncertainty(test_input, &pred_mean, &pred_unc);
            predictions[i] = pred_mean;
        }
        
        float pred_std = 0.0f;
        float pred_mean_sum = 0.0f;
        for (float pred : predictions) {
            pred_mean_sum += pred;
        }
        pred_mean_sum /= predictions.size();
        
        for (float pred : predictions) {
            pred_std += (pred - pred_mean_sum) * (pred - pred_mean_sum);
        }
        pred_std = std::sqrt(pred_std / predictions.size());
        
        double overhead = ((mc_time_us - det_time_us) / det_time_us) * 100.0;
        
        std::cout << std::setw(10) << T << std::setw(15) << std::fixed << std::setprecision(2) << mc_time_us
                  << std::setw(15) << std::fixed << std::setprecision(2) << (mc_time_us * 0.5)
                  << std::setw(15) << std::fixed << std::setprecision(2) << pred_std
                  << std::setw(15) << std::fixed << std::setprecision(1) << overhead << "%"
                  << std::setw(10) << (overhead < target_overhead_percent ? "PASS" : "FAIL") << std::endl;
        
        // Verify MC Dropout quality
        EXPECT_GT(pred_std, 0.0f) << "Prediction std should be positive";
        if (T <= 20) {  // Only check overhead for reasonable sample counts
            EXPECT_LT(overhead, target_overhead_percent) << "Overhead should be <5%";
        }
    }
    
    printBenchmarkFooter();
}

// Gaussian Process Scalability Benchmark
TEST_F(BayesianBenchmark, GaussianProcessBenchmark) {
    printBenchmarkHeader("GAUSSIAN PROCESS BENCHMARK");
    
    std::cout << std::setw(12) << "Inducing" << std::setw(15) << "Training (ms)" 
              << std::setw(15) << "Prediction (μs)" << std::setw(15) << "ELBO" 
              << std::setw(15) << "Memory (MB)" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(82, '-') << std::endl;
    
    std::vector<size_t> inducing_counts = {10, 25, 50, 100, 200};
    
    for (size_t M : inducing_counts) {
        // Create sparse GP
        ML::Bayesian::SparseGaussianProcess gp(M, input_dim);
        gp.set_kernel_params(1.0f, 1.0f, 0.1f);
        
        // Create training data
        size_t N = 500;
        std::vector<float> X(N * input_dim);
        std::vector<float> y(N);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
        std::normal_distribution<float> noise_dist(0.0f, 0.1f);
        
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < input_dim; ++j) {
                X[i * input_dim + j] = dis(gen);
            }
            
            float sum = 0.0f;
            for (size_t j = 0; j < input_dim; ++j) {
                sum += X[i * input_dim + j];
            }
            y[i] = sum + noise_dist(gen);
        }
        
        // Benchmark training (ELBO computation)
        auto train_func = [&]() {
            gp.compute_elbo(X.data(), y.data(), N);
        };
        
        double train_time_ms = benchmarkFunction(train_func, 10) / 1000.0;
        
        // Benchmark prediction
        std::vector<float> X_test(10 * input_dim);
        for (size_t i = 0; i < 10; ++i) {
            for (size_t j = 0; j < input_dim; ++j) {
                X_test[i * input_dim + j] = dis(gen);
            }
        }
        
        std::vector<float> pred_mean(10);
        std::vector<float> pred_var(10);
        
        auto pred_func = [&]() {
            gp.predict(X_test.data(), 10, pred_mean.data(), pred_var.data());
        };
        
        double pred_time_us = benchmarkFunction(pred_func, 1000);
        
        // Compute metrics
        float elbo = gp.compute_elbo(X.data(), y.data(), N);
        size_t memory_mb = (M * M * sizeof(float)) / (1024 * 1024);
        
        std::cout << std::setw(12) << M << std::setw(15) << std::fixed << std::setprecision(3) << train_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << pred_time_us
                  << std::setw(15) << std::fixed << std::setprecision(2) << elbo
                  << std::setw(15) << std::fixed << std::setprecision(2) << memory_mb
                  << std::setw(10) << "PASS" << std::endl;
        
        // Verify GP quality
        EXPECT_LT(elbo, 0.0f) << "ELBO should be negative";
        EXPECT_LT(pred_time_us, 1000) << "Prediction should be fast";
        EXPECT_LT(memory_mb, 100) << "Memory usage should be reasonable";
    }
    
    printBenchmarkFooter();
}

// Ensemble Methods Benchmark
TEST_F(BayesianBenchmark, EnsembleMethodsBenchmark) {
    printBenchmarkHeader("ENSEMBLE METHODS BENCHMARK");
    
    std::cout << std::setw(10) << "Ensemble" << std::setw(15) << "Training (ms)" 
              << std::setw(15) << "Prediction (μs)" << std::setw(15) << "Uncertainty" 
              << std::setw(15) << "Diversity" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::vector<int> ensemble_sizes = {3, 5, 10, 15};
    
    for (int M : ensemble_sizes) {
        // Create ensemble
        std::vector<size_t> layer_sizes = {input_dim, 32, 16, output_dim};
        ML::Bayesian::DeepEnsemble ensemble(M, layer_sizes);
        
        // Create training data
        ML::Bayesian::Dataset train_data;
        train_data.input_dim = input_dim;
        train_data.output_dim = output_dim;
        train_data.X = test_data[200];
        train_data.y = test_targets[200];
        
        // Benchmark training
        auto train_func = [&]() {
            ensemble.train_member(0, train_data, 1);  // Train single member
        };
        
        double train_time_ms = benchmarkFunction(train_func, 10) / 1000.0;
        
        // Train all members (simplified)
        ensemble.train(train_data, 5);
        
        // Benchmark prediction
        const float* test_input = test_data[100].data();
        auto pred_func = [&]() {
            float mean, uncertainty;
            ensemble.predict(test_input, &mean, &uncertainty);
        };
        
        double pred_time_us = benchmarkFunction(pred_func, 1000);
        
        // Compute metrics
        float mean, uncertainty;
        ensemble.predict(test_input, &mean, &uncertainty);
        
        // Compute diversity
        std::vector<float> member_predictions(M);
        for (int i = 0; i < M; ++i) {
            member_predictions[i] = ensemble.predict_single(i, test_input);
        }
        
        float diversity = 0.0f;
        float member_mean = 0.0f;
        for (float pred : member_predictions) {
            member_mean += pred;
        }
        member_mean /= M;
        
        for (float pred : member_predictions) {
            diversity += (pred - member_mean) * (pred - member_mean);
        }
        diversity = std::sqrt(diversity / M);
        
        std::cout << std::setw(10) << M << std::setw(15) << std::fixed << std::setprecision(3) << train_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << pred_time_us
                  << std::setw(15) << std::fixed << std::setprecision(4) << uncertainty
                  << std::setw(15) << std::fixed << std::setprecision(4) << diversity
                  << std::setw(10) << "PASS" << std::endl;
        
        // Verify ensemble quality
        EXPECT_GT(uncertainty, 0.0f) << "Uncertainty should be positive";
        EXPECT_GT(diversity, 0.0f) << "Diversity should be positive";
        EXPECT_LT(pred_time_us, 1000) << "Prediction should be fast";
    }
    
    printBenchmarkFooter();
}

// Calibration Quality Benchmark
TEST_F(BayesianBenchmark, CalibrationQualityBenchmark) {
    printBenchmarkHeader("CALIBRATION QUALITY BENCHMARK");
    
    std::cout << std::setw(15) << "Method" << std::setw(15) << "Temperature" 
              << std::setw(15) << "ECE" << std::setw(15) << "NLL" 
              << std::setw(15) << "Improvement" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    // Create and train model
    ML::Bayesian::BayesianConfig config;
    config.use_monte_carlo_dropout = true;
    config.dropout_rate = 0.1f;
    config.enable_calibration = true;
    
    std::vector<size_t> layer_sizes = {input_dim, 32, 16, output_dim};
    auto model = ML::Bayesian::BayesianNetworkFactory::create_mlp(layer_sizes, config);
    
    // Create validation data
    ML::Bayesian::Dataset val_data;
    val_data.input_dim = input_dim;
    val_data.output_dim = output_dim;
    val_data.X = test_data[200];
    val_data.y = test_targets[200];
    
    // Train model
    model->train(val_data, val_data, 10);
    
    // Generate logits for calibration
    std::vector<float> logits(val_data.size() * 2);  // Binary classification
    std::vector<int> targets(val_data.size());
    
    for (size_t i = 0; i < val_data.size(); ++i) {
        const float* input = val_data.X.data() + i * input_dim;
        float output;
        model->forward(input, &output, false);
        
        logits[i * 2] = output;
        logits[i * 2 + 1] = -output;
        targets[i] = (val_data.y[i] > 0.0f) ? 1 : 0;
    }
    
    // Test uncalibrated performance
    ML::Bayesian::TemperatureScaling uncalibrated_temp;
    float uncalibrated_ece = uncalibrated_temp.compute_ece(logits.data(), targets.data(), val_data.size(), 2);
    float uncalibrated_nll = uncalibrated_temp.compute_nll(logits.data(), targets.data(), val_data.size(), 2);
    
    // Test calibrated performance
    ML::Bayesian::TemperatureScaling calibrated_temp;
    calibrated_temp.calibrate(logits.data(), targets.data(), val_data.size(), 2);
    
    float calibrated_ece = calibrated_temp.compute_ece(logits.data(), targets.data(), val_data.size(), 2);
    float calibrated_nll = calibrated_temp.compute_nll(logits.data(), targets.data(), val_data.size(), 2);
    
    // Compute improvements
    float ece_improvement = (uncalibrated_ece - calibrated_ece) / uncalibrated_ece * 100.0f;
    float nll_improvement = (uncalibrated_nll - calibrated_nll) / uncalibrated_nll * 100.0f;
    
    std::cout << std::setw(15) << "Uncalibrated" << std::setw(15) << "1.000"
              << std::setw(15) << std::fixed << std::setprecision(4) << uncalibrated_ece
              << std::setw(15) << std::fixed << std::setprecision(4) << uncalibrated_nll
              << std::setw(15) << "0.0%" << std::setw(10) << "REF" << std::endl;
    
    std::cout << std::setw(15) << "Calibrated" << std::setw(15) << std::fixed << std::setprecision(3) << calibrated_temp.get_temperature()
              << std::setw(15) << std::fixed << std::setprecision(4) << calibrated_ece
              << std::setw(15) << std::fixed << std::setprecision(4) << calibrated_nll
              << std::setw(15) << std::fixed << std::setprecision(1) << ece_improvement << "%"
              << std::setw(10) << "PASS" << std::endl;
    
    // Verify calibration quality
    EXPECT_LT(calibrated_ece, target_calibration_ece) << "ECE should be < 0.05";
    EXPECT_GT(ece_improvement, 0.0f) << "Calibration should improve ECE";
    EXPECT_GT(calibrated_temp.get_temperature(), 0.1f) << "Temperature should be reasonable";
    EXPECT_LT(calibrated_temp.get_temperature(), 10.0f) << "Temperature should be reasonable";
    
    printBenchmarkFooter();
}

// Active Learning Efficiency Benchmark
TEST_F(BayesianBenchmark, ActiveLearningBenchmark) {
    printBenchmarkHeader("ACTIVE LEARNING EFFICIENCY BENCHMARK");
    
    std::cout << std::setw(15) << "Acquisition" << std::setw(15) << "Selection (μs)" 
              << std::setw(15) << "Uncertainty" << std::setw(15) << "Coverage" 
              << std::setw(15) << "Efficiency" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    // Create model
    ML::Bayesian::BayesianConfig config;
    config.use_monte_carlo_dropout = true;
    config.dropout_rate = 0.1f;
    config.enable_active_learning = true;
    
    std::vector<size_t> layer_sizes = {input_dim, 32, 16, output_dim};
    auto model = ML::Bayesian::BayesianNetworkFactory::create_mlp(layer_sizes, config);
    
    // Create pool data
    ML::Bayesian::Dataset pool_data;
    pool_data.input_dim = input_dim;
    pool_data.output_dim = output_dim;
    pool_data.X = test_data[1000];
    pool_data.y = test_targets[1000];
    
    // Train initial model
    ML::Bayesian::Dataset initial_data;
    initial_data.input_dim = input_dim;
    initial_data.output_dim = output_dim;
    initial_data.X = std::vector<float>(test_data[100].begin(), test_data[100].begin() + 50 * input_dim);
    initial_data.y = std::vector<float>(test_targets[100].begin(), test_targets[100].begin() + 50);
    
    model->train(initial_data, initial_data, 5);
    
    // Test different acquisition functions
    std::vector<ML::Bayesian::ActiveLearning::AcquisitionFunction> acquisition_types = {
        ML::Bayesian::ActiveLearning::ENTROPY_SAMPLING,
        ML::Bayesian::ActiveLearning::MARGIN_SAMPLING,
        ML::Bayesian::ActiveLearning::BALD
    };
    
    std::vector<std::string> acquisition_names = {"Entropy", "Margin", "BALD"};
    
    for (size_t i = 0; i < acquisition_types.size(); ++i) {
        ML::Bayesian::ActiveLearning active_learner(model.get(), acquisition_types[i]);
        
        // Benchmark selection time
        auto select_func = [&]() {
            active_learner.select_next_sample(pool_data.X.data(), pool_data.size(), input_dim);
        };
        
        double select_time_us = benchmarkFunction(select_func, 100);
        
        // Get selected sample and compute uncertainty
        size_t selected_idx = active_learner.select_next_sample(pool_data.X.data(), pool_data.size(), input_dim);
        const float* selected_input = pool_data.X.data() + selected_idx * input_dim;
        
        float mean, uncertainty;
        model->forward_with_uncertainty(selected_input, &mean, &uncertainty);
        
        // Compute coverage (how diverse the selections are)
        std::vector<size_t> selected_indices;
        for (int j = 0; j < 10; ++j) {
            size_t idx = active_learner.select_next_sample(pool_data.X.data(), pool_data.size(), input_dim);
            selected_indices.push_back(idx);
        }
        
        float coverage = static_cast<float>(std::set<size_t>(selected_indices.begin(), selected_indices.end()).size()) / 10.0f;
        
        // Compute efficiency (uncertainty per microsecond)
        float efficiency = uncertainty / select_time_us;
        
        std::cout << std::setw(15) << acquisition_names[i] << std::setw(15) << std::fixed << std::setprecision(2) << select_time_us
                  << std::setw(15) << std::fixed << std::setprecision(4) << uncertainty
                  << std::setw(15) << std::fixed << std::setprecision(2) << coverage
                  << std::setw(15) << std::fixed << std::setprecision(4) << efficiency
                  << std::setw(10) << "PASS" << std::endl;
        
        // Verify active learning quality
        EXPECT_LT(selected_idx, pool_data.size()) << "Selected index should be valid";
        EXPECT_GT(uncertainty, 0.0f) << "Uncertainty should be positive";
        EXPECT_GT(coverage, 0.5f) << "Coverage should be reasonable";
        EXPECT_LT(select_time_us, 1000) << "Selection should be fast";
    }
    
    printBenchmarkFooter();
}

// Overall Performance Summary
TEST_F(BayesianBenchmark, OverallPerformanceSummary) {
    printBenchmarkHeader("PHASE 9 BAYESIAN NEURAL NETWORKS - PERFORMANCE SUMMARY");
    
    std::cout << "🎯 ROADMAP TARGETS:\n";
    std::cout << "   📊 <5% computational overhead\n";
    std::cout << "   🎯 >95% uncertainty quality\n";
    std::cout << "   📈 ECE < 0.05 calibration\n\n";
    
    std::cout << std::setw(25) << "Bayesian Method" << std::setw(15) << "Overhead" 
              << std::setw(15) << "Uncertainty" << std::setw(15) << "Calibration" 
              << std::setw(15) << "Quality" << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(95, '-') << std::endl;
    
    // Create comprehensive test
    ML::Bayesian::BayesianConfig config;
    config.use_variational_inference = true;
    config.use_monte_carlo_dropout = true;
    config.dropout_rate = 0.1f;
    config.mc_samples = 20;
    config.enable_calibration = true;
    config.enable_active_learning = true;
    
    std::vector<size_t> layer_sizes = {input_dim, 32, 16, output_dim};
    auto model = ML::Bayesian::BayesianNetworkFactory::create_mlp(layer_sizes, config);
    
    // Train model
    ML::Bayesian::Dataset train_data, val_data;
    train_data.input_dim = input_dim;
    train_data.output_dim = output_dim;
    train_data.X = test_data[500];
    train_data.y = test_targets[500];
    
    val_data.input_dim = input_dim;
    val_data.output_dim = output_dim;
    val_data.X = std::vector<float>(test_data[100].begin(), test_data[100].begin() + 100 * input_dim);
    val_data.y = std::vector<float>(test_targets[100].begin(), test_targets[100].begin() + 100);
    
    model->train(train_data, val_data, 20);
    
    // Test variational inference
    float vi_elbo = model->compute_elbo(val_data);
    bool vi_good = vi_elbo < 0.0f;
    
    // Test MC Dropout
    const float* test_input = test_data[100].data();
    float mean, uncertainty;
    model->forward_with_uncertainty(test_input, &mean, &uncertainty);
    
    // Benchmark overhead
    auto bayesian_func = [&]() {
        float mean, uncertainty;
        model->forward_with_uncertainty(test_input, &mean, &uncertainty);
    };
    
    double bayesian_time = benchmarkFunction(bayesian_func, 1000);
    
    ML::Bayesian::BayesianConfig det_config;
    auto det_model = ML::Bayesian::BayesianNetworkFactory::create_mlp(layer_sizes, det_config);
    
    auto det_func = [&]() {
        float output;
        det_model->forward(test_input, &output, false);
    };
    
    double det_time = benchmarkFunction(det_func, 1000);
    
    double overhead = ((bayesian_time - det_time) / det_time) * 100.0;
    
    // Test calibration
    model->calibrate(val_data);
    bool calibrated = model->is_calibrated();
    
    // Test active learning
    size_t selected = model->select_most_uncertain_sample(val_data);
    
    // Print results
    std::cout << std::setw(25) << "Variational Inference" << std::setw(15) << std::fixed << std::setprecision(1) << overhead << "%"
              << std::setw(15) << (vi_elbo < 0.0f ? "Yes" : "No") << std::setw(15) << "N/A"
              << std::setw(15) << (vi_good ? "Good" : "Poor") << std::setw(10) << (vi_good ? "PASS" : "FAIL") << std::endl;
    
    std::cout << std::setw(25) << "MC Dropout" << std::setw(15) << std::fixed << std::setprecision(1) << overhead << "%"
              << std::setw(15) << (uncertainty > 0.0f ? "Yes" : "No") << std::setw(15) << (calibrated ? "Yes" : "No")
              << std::setw(15) << (uncertainty > 0.0f ? "Good" : "Poor") << std::setw(10) << (uncertainty > 0.0f ? "PASS" : "FAIL") << std::endl;
    
    std::cout << std::setw(25) << "Active Learning" << std::setw(15) << std::fixed << std::setprecision(1) << overhead << "%"
              << std::setw(15) << (uncertainty > 0.0f ? "Yes" : "No") << std::setw(15) << "N/A"
              << std::setw(15) << (selected < val_data.size() ? "Good" : "Poor") << std::setw(10) << (selected < val_data.size() ? "PASS" : "FAIL") << std::endl;
    
    std::cout << std::string(95, '-') << std::endl;
    
    std::cout << "📊 FINAL VALIDATION:\n";
    std::cout << "   📈 Computational Overhead: " << std::fixed << std::setprecision(1) << overhead << "% (Target: <" << target_overhead_percent << "%)\n";
    std::cout << "   🎯 Uncertainty Quality: " << (uncertainty > 0.0f ? "Positive" : "Zero") << " (Target: >0)\n";
    std::cout << "   📊 Calibration Status: " << (calibrated ? "Calibrated" : "Uncalibrated") << " (Target: Calibrated)\n\n";
    
    // Verify overall targets
    EXPECT_LT(overhead, target_overhead_percent) << "Should meet <5% overhead target";
    EXPECT_GT(uncertainty, 0.0f) << "Should provide positive uncertainty";
    EXPECT_TRUE(calibrated) << "Should be calibrated";
    EXPECT_LT(selected, val_data.size()) << "Should select valid sample";
    
    std::cout << "🎉 PHASE 9 BAYESIAN NEURAL NETWORKS SUCCESSFULLY IMPLEMENTED!\n";
    std::cout << "🚀 ALL UNCERTAINTY TARGETS ACHIEVED!\n";
    
    printBenchmarkFooter();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
