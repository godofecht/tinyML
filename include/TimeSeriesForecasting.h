//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Time Series Forecasting Models for Real-Time Systems
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 21/01/2026
*****************************************************************************/

#ifndef TIME_SERIES_FORECASTING_H
#define TIME_SERIES_FORECASTING_H

#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <chrono>
#include <map>
#include "XSIMDOperations.h"
#include "LightweightAttention.h"

namespace ML {
namespace TimeSeries {

// Forward declarations
class TCNLayer;
class WaveNetLayer;
class InformerLayer;
class AutoformerLayer;
class S4Layer;
class MambaLayer;
class HyenaLayer;
class NeuralODE;
class VectorAutoregression;
class DeepAR;
class ProphetModel;
class AnomalyDetector;
class ForecastingTransferLearner;

// Common structures and utilities
struct TimeSeriesData {
    std::vector<float> values;
    std::vector<float> timestamps;
    std::vector<std::vector<float>> features;  // Optional exogenous variables
    
    size_t length() const { return values.size(); }
    bool has_features() const { return !features.empty() && features.size() == values.size(); }
    void clear() { values.clear(); timestamps.clear(); features.clear(); }
};

struct ForecastResult {
    std::vector<float> predictions;
    std::vector<float> confidence_intervals_lower;
    std::vector<float> confidence_intervals_upper;
    float prediction_error;
    float computation_time_ms;
    
    ForecastResult(size_t horizon = 0) {
        predictions.resize(horizon);
        confidence_intervals_lower.resize(horizon);
        confidence_intervals_upper.resize(horizon);
        prediction_error = 0.0f;
        computation_time_ms = 0.0f;
    }
};

// ============================================================================
// TEMPORAL CONVOLUTIONAL NETWORKS (TCN & WaveNet)
// ============================================================================

class TCNLayer {
public:
    struct Config {
        size_t input_channels = 1;
        size_t output_channels = 64;
        size_t kernel_size = 3;
        size_t dilation = 1;
        size_t stride = 1;
        float dropout_rate = 0.1f;
        bool use_residual = true;
        bool use_batch_norm = true;
    };

    TCNLayer(const Config& config);
    ~TCNLayer() = default;

    std::vector<float> forward(const std::vector<float>& input);
    void reset_state();
    
    const Config& get_config() const { return config_; }
    size_t get_output_channels() const { return config_.output_channels; }

private:
    Config config_;
    
    // Convolution weights
    std::vector<float> conv_weights_;     // [output_channels, input_channels, kernel_size]
    std::vector<float> conv_bias_;        // [output_channels]
    
    // Batch normalization parameters
    std::vector<float> bn_gamma_;         // [output_channels]
    std::vector<float> bn_beta_;          // [output_channels]
    std::vector<float> bn_running_mean_;   // [output_channels]
    std::vector<float> bn_running_var_;    // [output_channels]
    
    // Residual connection weights
    std::vector<float> residual_weights_; // [input_channels, output_channels]
    std::vector<float> residual_bias_;    // [output_channels]
    
    // Temporary buffers
    std::vector<float> conv_buffer_;
    std::vector<float> activation_buffer_;
    std::vector<float> residual_buffer_;
    
    void initialize_weights();
    std::vector<float> causal_convolution(const std::vector<float>& input);
    std::vector<float> apply_batch_norm(const std::vector<float>& input);
    std::vector<float> apply_residual_connection(const std::vector<float>& input, 
                                               const std::vector<float>& output);
};

class WaveNetLayer {
public:
    struct Config {
        size_t input_channels = 1;
        size_t residual_channels = 64;
        size_t dilation_channels = 64;
        size_t skip_channels = 64;
        size_t kernel_size = 2;  // WaveNet uses kernel_size=2
        size_t dilation = 1;
        float dropout_rate = 0.1f;
        bool use_gated_activation = true;
    };

    WaveNetLayer(const Config& config);
    ~WaveNetLayer() = default;

    std::pair<std::vector<float>, std::vector<float>> forward(
        const std::vector<float>& input, const std::vector<float>& residual_input);
    void reset_state();
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // Dilated convolution weights
    std::vector<float> conv_filter_weights_;  // [dilation_channels, input_channels, kernel_size]
    std::vector<float> conv_gate_weights_;     // [dilation_channels, input_channels, kernel_size]
    std::vector<float> conv_filter_bias_;      // [dilation_channels]
    std::vector<float> conv_gate_bias_;        // [dilation_channels]
    
    // 1x1 convolution weights
    std::vector<float> conv_1x1_residual_weights_;  // [residual_channels, dilation_channels]
    std::vector<float> conv_1x1_skip_weights_;      // [skip_channels, dilation_channels]
    std::vector<float> conv_1x1_residual_bias_;     // [residual_channels]
    std::vector<float> conv_1x1_skip_bias_;         // [skip_channels]
    
    // Temporary buffers
    std::vector<float> filter_buffer_;
    std::vector<float> gate_buffer_;
    std::vector<float> activation_buffer_;
    
    void initialize_weights();
    std::vector<float> dilated_convolution(const std::vector<float>& input, 
                                          const std::vector<float>& weights, 
                                          const std::vector<float>& bias);
    std::vector<float> gated_activation(const std::vector<float>& filter, 
                                       const std::vector<float>& gate);
};

class TemporalConvolutionalNetwork {
public:
    struct Config {
        size_t input_channels = 1;
        size_t hidden_channels = 64;
        size_t output_channels = 1;
        size_t num_layers = 4;
        size_t kernel_size = 3;
        size_t max_dilation = 16;
        float dropout_rate = 0.1f;
        bool use_wave_net = false;
        size_t receptive_field = 0;  // Auto-computed if 0
    };

    TemporalConvolutionalNetwork(const Config& config);
    ~TemporalConvolutionalNetwork() = default;

    ForecastResult forecast(const TimeSeriesData& data, size_t horizon);
    void train(const std::vector<TimeSeriesData>& training_data);
    void reset_state();
    
    const Config& get_config() const { return config_; }
    size_t get_receptive_field() const;

private:
    Config config_;
    std::vector<std::unique_ptr<TCNLayer>> tcn_layers_;
    std::vector<std::unique_ptr<WaveNetLayer>> wavenet_layers_;
    std::vector<float> final_projection_weights_;
    std::vector<float> final_projection_bias_;
    
    void initialize_network();
    std::vector<float> process_sequence(const std::vector<float>& input);
    size_t compute_receptive_field() const;
};

// ============================================================================
// TRANSFORMER-BASED FORECASTING (Informer & Autoformer)
// ============================================================================

class InformerLayer {
public:
    struct Config {
        size_t d_model = 512;
        size_t n_heads = 8;
        size_t d_ff = 2048;
        size_t seq_len = 512;
        size_t label_len = 72;
        size_t pred_len = 72;
        float dropout_rate = 0.1f;
        float attention_dropout = 0.1f;
        bool use_prob_sparse_attention = true;
        float prob_sparse_factor = 0.25f;
    };

    InformerLayer(const Config& config);
    ~InformerLayer() = default;

    std::vector<float> forward(const std::vector<float>& input);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    std::unique_ptr<ML::RealTime::LightweightAttention> attention_;
    std::vector<float> ff_weights1_, ff_weights2_;
    std::vector<float> ff_bias1_, ff_bias2_;
    std::vector<float> norm_weights_, norm_bias_;
    
    std::vector<float> attention_output_;
    std::vector<float> ff_output_;
    
    void initialize_weights();
    std::vector<float> prob_sparse_attention(const std::vector<float>& q, 
                                           const std::vector<float>& k, 
                                           const std::vector<float>& v);
};

class AutoformerLayer {
public:
    struct Config {
        size_t d_model = 512;
        size_t n_heads = 8;
        size_t d_ff = 2048;
        size_t seq_len = 512;
        size_t label_len = 72;
        size_t pred_len = 72;
        float dropout_rate = 0.1f;
        size_t moving_avg_window = 25;
        bool use_autocorrelation = true;
    };

    AutoformerLayer(const Config& config);
    ~AutoformerLayer() = default;

    std::vector<float> forward(const std::vector<float>& input);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    std::unique_ptr<ML::RealTime::LightweightAttention> attention_;
    std::vector<float> ff_weights1_, ff_weights2_;
    std::vector<float> ff_bias1_, ff_bias2_;
    std::vector<float> norm_weights_, norm_bias_;
    std::vector<float> moving_avg_weights_;
    
    std::vector<float> attention_output_;
    std::vector<float> ff_output_;
    std::vector<float> decomposition_output_;
    
    void initialize_weights();
    std::vector<float> autocorrelation_attention(const std::vector<float>& q, 
                                               const std::vector<float>& k, 
                                               const std::vector<float>& v);
    std::vector<float> series_decomposition(const std::vector<float>& input);
};

class TransformerForecaster {
public:
    struct Config {
        size_t input_dim = 1;
        size_t d_model = 512;
        size_t n_heads = 8;
        size_t num_layers = 3;
        size_t d_ff = 2048;
        size_t seq_len = 512;
        size_t label_len = 72;
        size_t pred_len = 72;
        float dropout_rate = 0.1f;
        bool use_informer = true;
        bool use_autoformer = false;
        bool use_embedding = true;
    };

    TransformerForecaster(const Config& config);
    ~TransformerForecaster() = default;

    ForecastResult forecast(const TimeSeriesData& data, size_t horizon);
    void train(const std::vector<TimeSeriesData>& training_data);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    std::vector<std::unique_ptr<InformerLayer>> informer_layers_;
    std::vector<std::unique_ptr<AutoformerLayer>> autoformer_layers_;
    std::vector<float> embedding_weights_;
    std::vector<float> projection_weights_;
    std::vector<float> projection_bias_;
    
    void initialize_network();
    std::vector<float> embed_input(const TimeSeriesData& data);
    std::vector<float> decode_output(const std::vector<float>& encoded);
};

// ============================================================================
// STATE SPACE MODELS (S4, Mamba, Hyena)
// ============================================================================

class S4Layer {
public:
    struct Config {
        size_t d_model = 512;
        size_t d_state = 64;
        size_t seq_len = 512;
        float dropout_rate = 0.1f;
        bool use_normal_plus_plus = true;
        size_t n_ssm = 1;
    };

    S4Layer(const Config& config);
    ~S4Layer() = default;

    std::vector<float> forward(const std::vector<float>& input);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // SSM parameters
    std::vector<float> Lambda_;           // Complex eigenvalues
    std::vector<float> Lambda_re_;        // Real part
    std::vector<float> Lambda_im_;        // Imaginary part
    std::vector<float> P_;                // Projection matrix
    std::vector<float> B_;                // Input matrix
    std::vector<float> C_;                // Output matrix
    std::vector<float> D_;                // Skip connection
    std::vector<float> dt_;               // Time step
    
    // Projection weights
    std::vector<float> in_proj_weights_;  // [d_model, d_model]
    std::vector<float> out_proj_weights_; // [d_model, d_model]
    
    // Temporary buffers
    std::vector<float> state_buffer_;
    std::vector<float> output_buffer_;
    
    void initialize_weights();
    std::vector<float> discrete_ssm(const std::vector<float>& input);
    std::vector<float> convolution_kernel();
};

class MambaLayer {
public:
    struct Config {
        size_t d_model = 512;
        size_t d_state = 64;
        size_t d_conv = 4;
        size_t d_inner = 2048;
        size_t seq_len = 512;
        float dropout_rate = 0.1f;
        bool use_silu_activation = true;
        bool use_conv1d = true;
    };

    MambaLayer(const Config& config);
    ~MambaLayer() = default;

    std::vector<float> forward(const std::vector<float>& input);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // SSM parameters
    std::vector<float> A_log_;            // Log of state matrix
    std::vector<float> D_;                // Skip connection
    std::vector<float> dt_bias_;          // Time step bias
    std::vector<float> B_proj_;           // Input projection
    std::vector<float> C_proj_;           // Output projection
    
    // Convolution weights
    std::vector<float> conv1d_weights_;   // [d_inner, d_conv]
    std::vector<float> conv1d_bias_;      // [d_inner]
    
    // Projection weights
    std::vector<float> in_proj_weights_;  // [d_inner, d_model]
    std::vector<float> out_proj_weights_; // [d_model, d_inner]
    
    // Temporary buffers
    std::vector<float> conv_buffer_;
    std::vector<float> ssm_buffer_;
    std::vector<float> activation_buffer_;
    
    void initialize_weights();
    std::vector<float> selective_scan(const std::vector<float>& input);
    std::vector<float> apply_convolution(const std::vector<float>& input);
};

class HyenaLayer {
public:
    struct Config {
        size_t d_model = 512;
        size_t d_state = 64;
        size_t num_orders = 2;
        size_t seq_len = 512;
        float dropout_rate = 0.1f;
        bool use_long_conv = true;
        size_t kernel_size = 8;
    };

    HyenaLayer(const Config& config);
    ~HyenaLayer() = default;

    std::vector<float> forward(const std::vector<float>& input);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // Hyena filter parameters
    std::vector<float> filter_weights_;   // [d_model, kernel_size]
    std::vector<float> gate_weights_;     // [d_model, kernel_size]
    std::vector<float> output_weights_;   // [d_model, d_model]
    
    // Projection weights
    std::vector<float> in_proj_weights_;  // [d_model, d_model]
    std::vector<float> out_proj_weights_; // [d_model, d_model]
    
    // Temporary buffers
    std::vector<float> filter_buffer_;
    std::vector<float> gate_buffer_;
    std::vector<float> output_buffer_;
    
    void initialize_weights();
    std::vector<float> long_convolution(const std::vector<float>& input);
    std::vector<float> hyena_filter(const std::vector<float>& input);
};

class StateSpaceModel {
public:
    struct Config {
        size_t input_dim = 1;
        size_t d_model = 512;
        size_t d_state = 64;
        size_t num_layers = 3;
        size_t seq_len = 512;
        size_t pred_len = 72;
        float dropout_rate = 0.1f;
        bool use_s4 = true;
        bool use_mamba = false;
        bool use_hyena = false;
    };

    StateSpaceModel(const Config& config);
    ~StateSpaceModel() = default;

    ForecastResult forecast(const TimeSeriesData& data, size_t horizon);
    void train(const std::vector<TimeSeriesData>& training_data);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    std::vector<std::unique_ptr<S4Layer>> s4_layers_;
    std::vector<std::unique_ptr<MambaLayer>> mamba_layers_;
    std::vector<std::unique_ptr<HyenaLayer>> hyena_layers_;
    std::vector<float> embedding_weights_;
    std::vector<float> projection_weights_;
    
    void initialize_network();
    std::vector<float> embed_input(const TimeSeriesData& data);
};

// ============================================================================
// NEURAL ODES FOR CONTINUOUS-TIME DYNAMICS
// ============================================================================

class NeuralODE {
public:
    struct Config {
        size_t input_dim = 1;
        size_t hidden_dim = 64;
        size_t output_dim = 1;
        float solver_tolerance = 1e-5f;
        std::string solver_type = "dopri5";  // dopri5, euler, rk4
        bool use_adjoint = true;
        float max_time = 1.0f;
    };

    NeuralODE(const Config& config);
    ~NeuralODE() = default;

    ForecastResult forecast(const TimeSeriesData& data, size_t horizon);
    void train(const std::vector<TimeSeriesData>& training_data);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // ODE function parameters
    std::vector<float> func_weights1_, func_weights2_;
    std::vector<float> func_bias1_, func_bias2_;
    
    // Integration state
    std::vector<float> integration_state_;
    
    void initialize_weights();
    std::vector<float> ode_function(const std::vector<float>& state, float t);
    std::vector<float> integrate_ode(const std::vector<float>& initial_state, 
                                   float t0, float t1);
    std::vector<float> dopri5_step(const std::vector<float>& state, float t, float h);
};

// ============================================================================
// MULTIVARIATE FORECASTING (Vector Autoregression)
// ============================================================================

class VectorAutoregression {
public:
    struct Config {
        size_t num_variables = 1;
        size_t lag_order = 1;
        bool use_regularization = true;
        float regularization_strength = 0.01f;
        bool use_intercept = true;
        std::string estimation_method = "ols";  // ols, ridge, lasso
    };

    VectorAutoregression(const Config& config);
    ~VectorAutoregression() = default;

    ForecastResult forecast(const TimeSeriesData& data, size_t horizon);
    void fit(const TimeSeriesData& data);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // VAR coefficients
    std::vector<std::vector<std::vector<float>>> coefficients_;  // [lag_order, num_vars, num_vars]
    std::vector<float> intercept_;                               // [num_variables]
    
    // Covariance matrices
    std::vector<std::vector<float>> covariance_matrix_;
    std::vector<std::vector<float>> precision_matrix_;
    
    void fit_ols(const TimeSeriesData& data);
    void fit_ridge(const TimeSeriesData& data);
    std::vector<float> predict_next_step(const std::vector<std::vector<float>>& lagged_values);
};

// ============================================================================
// PROBABILISTIC FORECASTING (DeepAR & Prophet)
// ============================================================================

class DeepAR {
public:
    struct Config {
        size_t input_dim = 1;
        size_t hidden_dim = 64;
        size_t num_layers = 2;
        size_t embedding_dim = 10;
        float dropout_rate = 0.1f;
        std::string likelihood = "gaussian";  // gaussian, student_t, negative_binomial
        bool use_covariates = false;
        size_t num_covariates = 0;
    };

    DeepAR(const Config& config);
    ~DeepAR() = default;

    ForecastResult forecast(const TimeSeriesData& data, size_t horizon);
    void train(const std::vector<TimeSeriesData>& training_data);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // LSTM parameters
    std::vector<float> lstm_weights_i2h_, lstm_weights_h2h_;
    std::vector<float> lstm_biases_i2h_, lstm_biases_h2h_;
    
    // Distribution parameters
    std::vector<float> loc_weights_, scale_weights_;
    std::vector<float> loc_bias_, scale_bias_;
    
    // Embedding weights
    std::vector<float> embedding_weights_;
    
    void initialize_weights();
    std::vector<float> lstm_forward(const std::vector<float>& input);
    std::pair<float, float> predict_distribution(const std::vector<float>& hidden_state);
};

class ProphetModel {
public:
    struct Config {
        float growth_rate = 1.0f;
        size_t changepoints_num = 25;
        float changepoint_prior_scale = 0.05f;
        size_t yearly_seasonality_order = 10;
        size_t weekly_seasonality_order = 3;
        size_t daily_seasonality_order = 4;
        float seasonality_prior_scale = 10.0f;
        float holidays_prior_scale = 10.0f;
        bool include_holidays = false;
    };

    ProphetModel(const Config& config);
    ~ProphetModel() = default;

    ForecastResult forecast(const TimeSeriesData& data, size_t horizon);
    void fit(const TimeSeriesData& data);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // Trend parameters
    std::vector<float> changepoints_;
    std::vector<float> changepoint_rates_;
    float base_growth_rate_;
    
    // Seasonality parameters
    std::vector<float> yearly_seasonality_;
    std::vector<float> weekly_seasonality_;
    std::vector<float> daily_seasonality_;
    
    // Holiday parameters
    std::map<std::string, std::vector<std::pair<float, float>>> holidays_;
    
    void fit_trend(const TimeSeriesData& data);
    void fit_seasonality(const TimeSeriesData& data);
    float predict_trend(float time);
    float predict_seasonality(float time);
};

// ============================================================================
// ANOMALY DETECTION (Unsupervised Methods)
// ============================================================================

class AnomalyDetector {
public:
    struct Config {
        size_t window_size = 50;
        float contamination_rate = 0.1f;
        std::string method = "isolation_forest";  // isolation_forest, lof, one_class_svm
        size_t n_estimators = 100;
        float threshold_percentile = 95.0f;
        bool use_seasonal_decomposition = true;
    };

    AnomalyDetector(const Config& config);
    ~AnomalyDetector() = default;

    std::vector<bool> detect_anomalies(const TimeSeriesData& data);
    void fit(const TimeSeriesData& data);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // Isolation forest parameters
    std::vector<std::vector<float>> isolation_trees_;
    std::vector<float> tree_heights_;
    
    // LOF parameters
    std::vector<std::vector<float>> training_data_;
    std::vector<float> lof_scores_;
    
    void fit_isolation_forest(const TimeSeriesData& data);
    void fit_lof(const TimeSeriesData& data);
    std::vector<bool> detect_isolation_forest(const TimeSeriesData& data);
    std::vector<bool> detect_lof(const TimeSeriesData& data);
    std::vector<float> seasonal_decompose(const TimeSeriesData& data);
};

// ============================================================================
// TRANSFER LEARNING FOR FORECASTING
// ============================================================================

class ForecastingTransferLearner {
public:
    struct Config {
        size_t pretrained_model_dim = 512;
        size_t target_model_dim = 256;
        float learning_rate = 0.001f;
        size_t fine_tuning_steps = 1000;
        bool freeze_encoder = true;
        float adapter_alpha = 0.1f;
    };

    ForecastingTransferLearner(const Config& config);
    ~ForecastingTransferLearner() = default;

    ForecastResult forecast(const TimeSeriesData& data, size_t horizon);
    void load_pretrained_model(const std::string& model_path);
    void fine_tune(const std::vector<TimeSeriesData>& target_data);
    
    const Config& get_config() const { return config_; }

private:
    Config config_;
    
    // Pretrained model parameters
    std::vector<float> pretrained_weights_;
    std::vector<float> pretrained_biases_;
    
    // Adapter layers
    std::vector<float> adapter_weights_;
    std::vector<float> adapter_biases_;
    
    // Fine-tuning parameters
    std::vector<float> fine_tuning_weights_;
    std::vector<float> fine_tuning_biases_;
    
    void initialize_adapters();
    std::vector<float> apply_adapters(const std::vector<float>& features);
};

// ============================================================================
// MAIN TIME SERIES FORECASTING INTERFACE
// ============================================================================

class TimeSeriesForecaster {
public:
    enum class ModelType {
        TCN,
        WAVENET,
        INFORMER,
        AUTOFORMER,
        S4,
        MAMBA,
        HYENA,
        NEURAL_ODE,
        VAR,
        DEEPAR,
        PROPHET,
        ANOMALY_DETECTOR
    };

    struct Config {
        ModelType model_type = ModelType::TCN;
        size_t prediction_horizon = 72;
        float confidence_level = 0.95f;
        bool enable_anomaly_detection = false;
        bool use_transfer_learning = false;
        std::string pretrained_model_path = "";
    };

    TimeSeriesForecaster(const Config& config);
    ~TimeSeriesForecaster() = default;

    ForecastResult forecast(const TimeSeriesData& data);
    void train(const std::vector<TimeSeriesData>& training_data);
    std::vector<bool> detect_anomalies(const TimeSeriesData& data);
    
    const Config& get_config() const { return config_; }
    ModelType get_model_type() const { return config_.model_type; }

private:
    Config config_;
    
    std::unique_ptr<TemporalConvolutionalNetwork> tcn_model_;
    std::unique_ptr<TransformerForecaster> transformer_model_;
    std::unique_ptr<StateSpaceModel> ssm_model_;
    std::unique_ptr<NeuralODE> neural_ode_model_;
    std::unique_ptr<VectorAutoregression> var_model_;
    std::unique_ptr<DeepAR> deepar_model_;
    std::unique_ptr<ProphetModel> prophet_model_;
    std::unique_ptr<AnomalyDetector> anomaly_detector_;
    std::unique_ptr<ForecastingTransferLearner> transfer_learner_;
    
    void initialize_model();
};

// Factory functions
std::unique_ptr<TemporalConvolutionalNetwork> create_tcn_model(
    const TemporalConvolutionalNetwork::Config& config);

std::unique_ptr<TransformerForecaster> create_transformer_forecaster(
    const TransformerForecaster::Config& config);

std::unique_ptr<StateSpaceModel> create_state_space_model(
    const StateSpaceModel::Config& config);

std::unique_ptr<TimeSeriesForecaster> create_time_series_forecaster(
    const TimeSeriesForecaster::Config& config);

} // namespace TimeSeries
} // namespace ML

#endif // TIME_SERIES_FORECASTING_H
