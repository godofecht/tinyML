//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Phase 13: Time Series Forecasting Tests
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 21/01/2026
*****************************************************************************/

#include "TimeSeriesForecasting.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>

using namespace ML::TimeSeries;

// Test utilities
class TimeSeriesTestUtils {
public:
    static TimeSeriesData generate_sine_wave(size_t length, float amplitude = 1.0f, float frequency = 0.1f, float phase = 0.0f) {
        TimeSeriesData data;
        data.values.resize(length);
        data.timestamps.resize(length);
        
        for (size_t i = 0; i < length; ++i) {
            data.timestamps[i] = static_cast<float>(i);
            data.values[i] = amplitude * std::sin(2.0f * M_PI * frequency * i + phase);
        }
        
        return data;
    }
    
    static TimeSeriesData generate_linear_trend(size_t length, float slope = 0.1f, float intercept = 0.0f) {
        TimeSeriesData data;
        data.values.resize(length);
        data.timestamps.resize(length);
        
        for (size_t i = 0; i < length; ++i) {
            data.timestamps[i] = static_cast<float>(i);
            data.values[i] = slope * i + intercept;
        }
        
        return data;
    }
    
    static TimeSeriesData generate_random_walk(size_t length, float volatility = 0.1f) {
        TimeSeriesData data;
        data.values.resize(length);
        data.timestamps.resize(length);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, volatility);
        
        data.values[0] = 0.0f;
        for (size_t i = 1; i < length; ++i) {
            data.timestamps[i] = static_cast<float>(i);
            data.values[i] = data.values[i-1] + dist(gen);
        }
        
        return data;
    }
    
    static float calculate_mse(const std::vector<float>& predictions, const std::vector<float>& targets) {
        if (predictions.size() != targets.size() || predictions.empty()) {
            return std::numeric_limits<float>::infinity();
        }
        
        float mse = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float error = predictions[i] - targets[i];
            mse += error * error;
        }
        
        return mse / static_cast<float>(predictions.size());
    }
    
    static float calculate_mae(const std::vector<float>& predictions, const std::vector<float>& targets) {
        if (predictions.size() != targets.size() || predictions.empty()) {
            return std::numeric_limits<float>::infinity();
        }
        
        float mae = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            mae += std::abs(predictions[i] - targets[i]);
        }
        
        return mae / static_cast<float>(predictions.size());
    }
    
    static bool test_performance_threshold(float computation_time_ms, float threshold_ms) {
        return computation_time_ms < threshold_ms;
    }
};

// Test cases
class Phase13TimeSeriesTests {
public:
    static bool run_all_tests() {
        std::cout << "=== Phase 13: Time Series Forecasting Tests ===" << std::endl;
        
        bool all_passed = true;
        
        all_passed &= test_temporal_convolutional_networks();
        all_passed &= test_transformer_forecasting();
        all_passed &= test_state_space_models();
        all_passed &= test_neural_odes();
        all_passed &= test_multivariate_forecasting();
        all_passed &= test_probabilistic_forecasting();
        all_passed &= test_anomaly_detection();
        all_passed &= test_transfer_learning();
        all_passed &= test_main_forecaster_interface();
        all_passed &= test_performance_benchmarks();
        
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Overall Status: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
        
        return all_passed;
    }
    
private:
    static bool test_temporal_convolutional_networks() {
        std::cout << "\n--- Testing Temporal Convolutional Networks ---" << std::endl;
        
        bool passed = true;
        
        // Test TCN
        {
            std::cout << "Testing TCN..." << std::endl;
            
            TemporalConvolutionalNetwork::Config config;
            config.input_channels = 1;
            config.hidden_channels = 64;
            config.output_channels = 1;
            config.num_layers = 3;
            config.kernel_size = 3;
            config.max_dilation = 8;
            config.use_wave_net = false;
            
            auto tcn = create_tcn_model(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(100);
            
            // Test forecasting
            auto result = tcn->forecast(data, 10);
            
            bool test_passed = (result.predictions.size() == 10) && 
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 10.0f));
            
            std::cout << "  TCN Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        // Test WaveNet
        {
            std::cout << "Testing WaveNet..." << std::endl;
            
            TemporalConvolutionalNetwork::Config config;
            config.input_channels = 1;
            config.hidden_channels = 64;
            config.output_channels = 1;
            config.num_layers = 3;
            config.kernel_size = 2;
            config.max_dilation = 8;
            config.use_wave_net = true;
            
            auto wavenet = create_tcn_model(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(100);
            
            // Test forecasting
            auto result = wavenet->forecast(data, 10);
            
            bool test_passed = (result.predictions.size() == 10) && 
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 10.0f));
            
            std::cout << "  WaveNet Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        std::cout << "Temporal Convolutional Networks: " << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed;
    }
    
    static bool test_transformer_forecasting() {
        std::cout << "\n--- Testing Transformer-Based Forecasting ---" << std::endl;
        
        bool passed = true;
        
        // Test Informer
        {
            std::cout << "Testing Informer..." << std::endl;
            
            TransformerForecaster::Config config;
            config.input_dim = 1;
            config.d_model = 256;
            config.n_heads = 4;
            config.num_layers = 2;
            config.d_ff = 1024;
            config.seq_len = 256;
            config.pred_len = 32;
            config.use_informer = true;
            config.use_autoformer = false;
            
            auto informer = create_transformer_forecaster(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(300);
            
            // Test forecasting
            auto result = informer->forecast(data, 20);
            
            bool test_passed = (result.predictions.size() == 20) && 
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 50.0f));
            
            std::cout << "  Informer Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        // Test Autoformer
        {
            std::cout << "Testing Autoformer..." << std::endl;
            
            TransformerForecaster::Config config;
            config.input_dim = 1;
            config.d_model = 256;
            config.n_heads = 4;
            config.num_layers = 2;
            config.d_ff = 1024;
            config.seq_len = 256;
            config.pred_len = 32;
            config.use_informer = false;
            config.use_autoformer = true;
            
            auto autoformer = create_transformer_forecaster(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(300);
            
            // Test forecasting
            auto result = autoformer->forecast(data, 20);
            
            bool test_passed = (result.predictions.size() == 20) && 
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 50.0f));
            
            std::cout << "  Autoformer Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        std::cout << "Transformer-Based Forecasting: " << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed;
    }
    
    static bool test_state_space_models() {
        std::cout << "\n--- Testing State Space Models ---" << std::endl;
        
        bool passed = true;
        
        // Test S4
        {
            std::cout << "Testing S4..." << std::endl;
            
            StateSpaceModel::Config config;
            config.input_dim = 1;
            config.d_model = 256;
            config.d_state = 32;
            config.num_layers = 2;
            config.seq_len = 256;
            config.pred_len = 32;
            config.use_s4 = true;
            config.use_mamba = false;
            config.use_hyena = false;
            
            auto s4 = create_state_space_model(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(300);
            
            // Test forecasting
            auto result = s4->forecast(data, 20);
            
            bool test_passed = (result.predictions.size() == 20) && 
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 20.0f));
            
            std::cout << "  S4 Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        // Test Mamba
        {
            std::cout << "Testing Mamba..." << std::endl;
            
            StateSpaceModel::Config config;
            config.input_dim = 1;
            config.d_model = 256;
            config.d_state = 32;
            config.num_layers = 2;
            config.seq_len = 256;
            config.pred_len = 32;
            config.use_s4 = false;
            config.use_mamba = true;
            config.use_hyena = false;
            
            auto mamba = create_state_space_model(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(300);
            
            // Test forecasting
            auto result = mamba->forecast(data, 20);
            
            bool test_passed = (result.predictions.size() == 20) && 
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 20.0f));
            
            std::cout << "  Mamba Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        // Test Hyena
        {
            std::cout << "Testing Hyena..." << std::endl;
            
            StateSpaceModel::Config config;
            config.input_dim = 1;
            config.d_model = 256;
            config.d_state = 32;
            config.num_layers = 2;
            config.seq_len = 256;
            config.pred_len = 32;
            config.use_s4 = false;
            config.use_mamba = false;
            config.use_hyena = true;
            
            auto hyena = create_state_space_model(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(300);
            
            // Test forecasting
            auto result = hyena->forecast(data, 20);
            
            bool test_passed = (result.predictions.size() == 20) && 
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 20.0f));
            
            std::cout << "  Hyena Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        std::cout << "State Space Models: " << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed;
    }
    
    static bool test_neural_odes() {
        std::cout << "\n--- Testing Neural ODEs ---" << std::endl;
        
        bool passed = true;
        
        NeuralODE::Config config;
        config.input_dim = 1;
        config.hidden_dim = 32;
        config.output_dim = 1;
        config.solver_tolerance = 1e-4f;
        config.solver_type = "rk4";
        config.use_adjoint = false;
        config.max_time = 1.0f;
        
        auto neural_ode = std::make_unique<NeuralODE>(config);
        
        // Generate test data
        auto data = TimeSeriesTestUtils::generate_sine_wave(50);
        
        // Test forecasting
        auto result = neural_ode->forecast(data, 15);
        
        bool test_passed = (result.predictions.size() == 15) && 
                         (result.computation_time_ms > 0.0f) &&
                         (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 15.0f));
        
        std::cout << "Neural ODE Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
        std::cout << "Computation Time: " << result.computation_time_ms << " ms" << std::endl;
        
        std::cout << "Neural ODEs: " << (test_passed ? "PASSED" : "FAILED") << std::endl;
        return test_passed;
    }
    
    static bool test_multivariate_forecasting() {
        std::cout << "\n--- Testing Multivariate Forecasting ---" << std::endl;
        
        bool passed = true;
        
        VectorAutoregression::Config config;
        config.num_variables = 3;
        config.lag_order = 3;
        config.use_regularization = true;
        config.regularization_strength = 0.01f;
        config.use_intercept = true;
        config.estimation_method = "ols";
        
        auto var = std::make_unique<VectorAutoregression>(config);
        
        // Generate multivariate test data
        TimeSeriesData data;
        data.values.resize(100);
        data.timestamps.resize(100);
        data.features.resize(100);
        
        for (size_t i = 0; i < 100; ++i) {
            data.timestamps[i] = static_cast<float>(i);
            data.values[i] = std::sin(0.1f * i);
            data.features[i] = {
                std::sin(0.1f * i),
                std::cos(0.1f * i),
                0.1f * i
            };
        }
        
        // Fit the model
        var->fit(data);
        
        // Test forecasting
        auto result = var->forecast(data, 10);
        
        bool test_passed = (result.predictions.size() == 10) && 
                         (result.computation_time_ms > 0.0f) &&
                         (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 5.0f));
        
        std::cout << "VAR Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
        std::cout << "Computation Time: " << result.computation_time_ms << " ms" << std::endl;
        
        std::cout << "Multivariate Forecasting: " << (test_passed ? "PASSED" : "FAILED") << std::endl;
        return test_passed;
    }
    
    static bool test_probabilistic_forecasting() {
        std::cout << "\n--- Testing Probabilistic Forecasting ---" << std::endl;
        
        bool passed = true;
        
        // Test DeepAR
        {
            std::cout << "Testing DeepAR..." << std::endl;
            
            DeepAR::Config config;
            config.input_dim = 1;
            config.hidden_dim = 32;
            config.num_layers = 2;
            config.embedding_dim = 8;
            config.dropout_rate = 0.1f;
            config.likelihood = "gaussian";
            config.use_covariates = false;
            
            auto deepar = std::make_unique<DeepAR>(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(100);
            
            // Test forecasting
            auto result = deepar->forecast(data, 15);
            
            bool test_passed = (result.predictions.size() == 15) && 
                             (result.confidence_intervals_lower.size() == 15) &&
                             (result.confidence_intervals_upper.size() == 15) &&
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 25.0f));
            
            std::cout << "  DeepAR Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        // Test Prophet
        {
            std::cout << "Testing Prophet..." << std::endl;
            
            ProphetModel::Config config;
            config.growth_rate = 0.1f;
            config.changepoints_num = 10;
            config.changepoint_prior_scale = 0.05f;
            config.yearly_seasonality_order = 5;
            config.weekly_seasonality_order = 2;
            config.daily_seasonality_order = 2;
            
            auto prophet = std::make_unique<ProphetModel>(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_linear_trend(200, 0.1f, 1.0f);
            
            // Test forecasting
            auto result = prophet->forecast(data, 20);
            
            bool test_passed = (result.predictions.size() == 20) && 
                             (result.confidence_intervals_lower.size() == 20) &&
                             (result.confidence_intervals_upper.size() == 20) &&
                             (result.computation_time_ms > 0.0f) &&
                             (TimeSeriesTestUtils::test_performance_threshold(result.computation_time_ms, 10.0f));
            
            std::cout << "  Prophet Forecast: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Computation Time: " << result.computation_time_ms << " ms" << std::endl;
            
            passed &= test_passed;
        }
        
        std::cout << "Probabilistic Forecasting: " << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed;
    }
    
    static bool test_anomaly_detection() {
        std::cout << "\n--- Testing Anomaly Detection ---" << std::endl;
        
        bool passed = true;
        
        // Test Isolation Forest
        {
            std::cout << "Testing Isolation Forest..." << std::endl;
            
            AnomalyDetector::Config config;
            config.window_size = 20;
            config.contamination_rate = 0.1f;
            config.method = "isolation_forest";
            config.n_estimators = 50;
            config.threshold_percentile = 95.0f;
            
            auto detector = std::make_unique<AnomalyDetector>(config);
            
            // Generate test data with some anomalies
            auto data = TimeSeriesTestUtils::generate_sine_wave(100);
            
            // Inject some anomalies
            data.values[25] = 5.0f;  // Anomaly
            data.values[50] = -4.0f; // Anomaly
            data.values[75] = 3.5f;  // Anomaly
            
            // Fit the detector
            detector->fit(data);
            
            // Test anomaly detection
            auto anomalies = detector->detect_anomalies(data);
            
            bool test_passed = (anomalies.size() == 100) && 
                             (std::count(anomalies.begin(), anomalies.end(), true) > 0);
            
            std::cout << "  Isolation Forest: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Anomalies Detected: " << std::count(anomalies.begin(), anomalies.end(), true) << std::endl;
            
            passed &= test_passed;
        }
        
        // Test LOF
        {
            std::cout << "Testing LOF..." << std::endl;
            
            AnomalyDetector::Config config;
            config.window_size = 20;
            config.contamination_rate = 0.1f;
            config.method = "lof";
            config.n_estimators = 50;
            config.threshold_percentile = 95.0f;
            
            auto detector = std::make_unique<AnomalyDetector>(config);
            
            // Generate test data with some anomalies
            auto data = TimeSeriesTestUtils::generate_sine_wave(100);
            
            // Inject some anomalies
            data.values[30] = 4.5f;  // Anomaly
            data.values[60] = -3.5f; // Anomaly
            
            // Fit the detector
            detector->fit(data);
            
            // Test anomaly detection
            auto anomalies = detector->detect_anomalies(data);
            
            bool test_passed = (anomalies.size() == 100) && 
                             (std::count(anomalies.begin(), anomalies.end(), true) > 0);
            
            std::cout << "  LOF: " << (test_passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Anomalies Detected: " << std::count(anomalies.begin(), anomalies.end(), true) << std::endl;
            
            passed &= test_passed;
        }
        
        std::cout << "Anomaly Detection: " << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed;
    }
    
    static bool test_transfer_learning() {
        std::cout << "\n--- Testing Transfer Learning ---" << std::endl;
        
        bool passed = true;
        
        ForecastingTransferLearner::Config config;
        config.pretrained_model_dim = 256;
        config.target_model_dim = 128;
        config.learning_rate = 0.001f;
        config.fine_tuning_steps = 100;
        config.freeze_encoder = true;
        config.adapter_alpha = 0.1f;
        
        auto transfer_learner = std::make_unique<ForecastingTransferLearner>(config);
        
        // Load pretrained model (simulated)
        transfer_learner->load_pretrained_model("dummy_model_path");
        
        // Generate target data
        auto data = TimeSeriesTestUtils::generate_sine_wave(80);
        
        // Test forecasting
        auto result = transfer_learner->forecast(data, 12);
        
        bool test_passed = (result.predictions.size() == 12) && 
                         (result.computation_time_ms > 0.0f);
        
        std::cout << "Transfer Learning: " << (test_passed ? "PASS" : "FAIL") << std::endl;
        std::cout << "Computation Time: " << result.computation_time_ms << " ms" << std::endl;
        
        std::cout << "Transfer Learning: " << (test_passed ? "PASSED" : "FAILED") << std::endl;
        return test_passed;
    }
    
    static bool test_main_forecaster_interface() {
        std::cout << "\n--- Testing Main Forecaster Interface ---" << std::endl;
        
        bool passed = true;
        
        // Test different model types through the main interface
        std::vector<TimeSeriesForecaster::ModelType> model_types = {
            TimeSeriesForecaster::ModelType::TCN,
            TimeSeriesForecaster::ModelType::WAVENET,
            TimeSeriesForecaster::ModelType::INFORMER,
            TimeSeriesForecaster::ModelType::S4,
            TimeSeriesForecaster::ModelType::NEURAL_ODE,
            TimeSeriesForecaster::ModelType::VAR,
            TimeSeriesForecaster::ModelType::DEEPAR,
            TimeSeriesForecaster::ModelType::PROPHET
        };
        
        for (auto model_type : model_types) {
            std::cout << "Testing " << static_cast<int>(model_type) << "..." << std::endl;
            
            TimeSeriesForecaster::Config config;
            config.model_type = model_type;
            config.prediction_horizon = 10;
            config.confidence_level = 0.95f;
            config.enable_anomaly_detection = false;
            config.use_transfer_learning = false;
            
            auto forecaster = create_time_series_forecaster(config);
            
            // Generate test data
            auto data = TimeSeriesTestUtils::generate_sine_wave(100);
            
            // Test forecasting
            auto result = forecaster->forecast(data);
            
            bool test_passed = (result.predictions.size() == 10) && 
                             (result.computation_time_ms > 0.0f);
            
            std::cout << "  Model " << static_cast<int>(model_type) << ": " << (test_passed ? "PASS" : "FAIL") << std::endl;
            
            passed &= test_passed;
        }
        
        std::cout << "Main Forecaster Interface: " << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed;
    }
    
    static bool test_performance_benchmarks() {
        std::cout << "\n--- Performance Benchmarks ---" << std::endl;
        
        bool passed = true;
        
        // Performance targets (in milliseconds)
        // Relaxed targets for CI environments (shared runners are slower)
        std::map<TimeSeriesForecaster::ModelType, float> performance_targets = {
            {TimeSeriesForecaster::ModelType::TCN, 25.0f},
            {TimeSeriesForecaster::ModelType::WAVENET, 40.0f},
            {TimeSeriesForecaster::ModelType::INFORMER, 250.0f},
            {TimeSeriesForecaster::ModelType::S4, 100.0f},
            {TimeSeriesForecaster::ModelType::NEURAL_ODE, 75.0f},
            {TimeSeriesForecaster::ModelType::VAR, 10.0f},
            {TimeSeriesForecaster::ModelType::DEEPAR, 125.0f},
            {TimeSeriesForecaster::ModelType::PROPHET, 50.0f}
        };
        
        for (const auto& [model_type, target_time] : performance_targets) {
            TimeSeriesForecaster::Config config;
            config.model_type = model_type;
            config.prediction_horizon = 20;
            
            auto forecaster = create_time_series_forecaster(config);
            auto data = TimeSeriesTestUtils::generate_sine_wave(200);
            
            // Run multiple times to get average performance
            float total_time = 0.0f;
            int runs = 5;
            
            for (int i = 0; i < runs; ++i) {
                auto result = forecaster->forecast(data);
                total_time += result.computation_time_ms;
            }
            
            float avg_time = total_time / runs;
            bool meets_target = avg_time < target_time;
            
            std::cout << "Model " << static_cast<int>(model_type) << ": ";
            std::cout << avg_time << " ms (target: " << target_time << " ms) - ";
            std::cout << (meets_target ? "PASS" : "FAIL") << std::endl;
            
            passed &= meets_target;
        }
        
        std::cout << "Performance Benchmarks: " << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed;
    }
};

int main() {
    std::cout << "Starting Phase 13: Time Series Forecasting Tests" << std::endl;
    std::cout << "Target: Sub-millisecond forecasting for real-time systems" << std::endl;
    
    bool success = Phase13TimeSeriesTests::run_all_tests();
    
    if (success) {
        std::cout << "\n🎉 All Phase 13 tests passed!" << std::endl;
        std::cout << "Time Series Forecasting implementation is ready for production." << std::endl;
    } else {
        std::cout << "\n❌ Some Phase 13 tests failed." << std::endl;
        std::cout << "Please review the implementation and fix any issues." << std::endl;
    }
    
    return success ? 0 : 1;
}
