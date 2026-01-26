//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Bayesian Neural Networks Header
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef BAYESIAN_NEURAL_NETWORK_H
#define BAYESIAN_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <cmath>

namespace ML {
namespace Bayesian {

// Forward declarations
class BayesianLinear;
class MCDropout;
class SparseGaussianProcess;
class DeepEnsemble;
class TemperatureScaling;
class ActiveLearning;
class BayesianTrainer;

// Configuration structures
struct BayesianConfig {
    // Prior parameters
    float prior_mean = 0.0f;
    float prior_std = 1.0f;
    float kl_weight = 1.0f;
    
    // Monte Carlo Dropout
    bool use_monte_carlo_dropout = true;
    float dropout_rate = 0.1f;
    int mc_samples = 20;
    
    // Variational inference
    bool use_variational_inference = false;
    bool use_local_reparameterization = true;
    float variational_lr = 0.001f;
    
    // Ensemble methods
    bool use_ensemble = false;
    int ensemble_size = 5;
    bool enable_swa = false;
    
    // Gaussian Process
    bool use_sparse_gp = false;
    size_t num_inducing = 100;
    float gp_lengthscale = 1.0f;
    float gp_signal_variance = 1.0f;
    float gp_noise_variance = 0.1f;
    
    // Calibration
    bool enable_calibration = true;
    float temperature = 1.0f;
    
    // Active learning
    bool enable_active_learning = false;
    enum AcquisitionFunction {
        ENTROPY_SAMPLING,
        MARGIN_SAMPLING,
        BALD
    } acquisition_type = BALD;
    
    // Performance optimization
    bool enable_quantization = false;
    bool enable_gradient_checkpointing = false;
    int min_mc_samples = 10;
    int max_mc_samples = 50;
    float uncertainty_threshold = 0.1f;
};

// Dataset structure
struct Dataset {
    std::vector<float> X;
    std::vector<float> y;
    std::vector<int> labels;
    
    std::unique_ptr<Dataset> train, val, test;
    
    size_t input_dim = 0;
    size_t output_dim = 0;

    Dataset() = default;
    Dataset(size_t in_dim, size_t out_dim) : input_dim(in_dim), output_dim(out_dim) {}
    
    size_t size() const { return input_dim > 0 ? X.size() / input_dim : 0; }

    
    void add_sample(const float* x, float y);
    void add_sample(const float* x, int label);
    void remove_sample(size_t index);
    void shuffle();
    void split_train_val_test(float train_ratio = 0.7, float val_ratio = 0.15);
};

// Bayesian Linear Layer
class BayesianLinear {
private:
    // Weight distributions (mean and log variance)
    std::vector<float> weight_mean;
    std::vector<float> weight_logvar;
    std::vector<float> bias_mean;
    std::vector<float> bias_logvar;
    
    // Sampled weights for forward pass
    std::vector<float> sampled_weights;
    std::vector<float> sampled_bias;
    
    // Gradients
    std::vector<float> grad_weight_mean;
    std::vector<float> grad_weight_logvar;
    std::vector<float> grad_bias_mean;
    std::vector<float> grad_bias_logvar;
    
    size_t input_size, output_size;
    bool use_local_reparameterization;
    bool is_training;
    
    // Random number generation
    std::mt19937 rng;
    std::normal_distribution<float> normal_dist;
    
public:
    BayesianLinear(size_t in_features, size_t out_features, 
                   bool local_reparam = true);
    
    // Forward pass
    void forward(const float* input, float* output, bool training = true);
    void backward(const float* input, const float* grad_output, 
                 float* grad_input);
    
    // Parameter management
    void sample_weights();
    void reset_parameters();
    void update_parameters(float learning_rate);
    void perturb_weights(float noise_std = 0.01f);
    
    // Regularization
    float compute_kl_divergence() const;
    
    // Accessors
    size_t get_input_size() const { return input_size; }
    size_t get_output_size() const { return output_size; }
    void set_training_mode(bool training) { is_training = training; }
    
    // For quantization
    void quantize_parameters();
    void dequantize_parameters();

    // Accessors for visualization
    const std::vector<float>& get_weight_mean() const { return weight_mean; }
    const std::vector<float>& get_bias_mean() const { return bias_mean; }
};

// Monte Carlo Dropout Layer
class MCDropout {
private:
    float dropout_rate;
    std::vector<bool> mask;
    std::vector<float> scaled_output;
    bool is_training;
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform_dist;
    
public:
    MCDropout(float rate = 0.1f);
    
    // Forward pass
    void forward(const float* input, float* output, size_t size);
    
    // Monte Carlo inference
    void mc_inference(const float* input, float* mean_output, 
                     float* uncertainty_output, size_t size, int T = 20);
    
    // Utility
    void set_training_mode(bool training) { is_training = training; }
    float get_dropout_rate() const { return dropout_rate; }
};

// Sparse Gaussian Process
class SparseGaussianProcess {
private:
    // Inducing points
    std::vector<float> inducing_inputs;
    std::vector<float> inducing_mean;
    std::vector<float> inducing_cov;
    std::vector<float> inducing_cholesky;
    
    // Kernel parameters
    float lengthscale;
    float signal_variance;
    float noise_variance;
    
    // Variational parameters
    std::vector<float> variational_mean;
    std::vector<float> variational_logvar;
    
    size_t num_inducing;
    size_t input_dim;
    
public:
    SparseGaussianProcess(size_t M, size_t D);
    
    // Kernel functions
    float rbf_kernel(const float* x1, const float* x2) const;
    float matern_kernel(const float* x1, const float* x2, float nu = 1.5f) const;
    void compute_kernel_matrix(const float* X1, const float* X2, 
                              size_t N1, size_t N2, float* K) const;
    
    // Inference
    void predict(const float* X_test, size_t N_test,
                float* mean, float* variance) const;
    
    // Training
    float compute_elbo(const float* X, const float* y, size_t N) const;
    void optimize_inducing_points(const float* X, size_t N, int iterations = 100);
    
    // Parameter management
    void set_kernel_params(float lengthscale, float signal_var, float noise_var);
    void set_inducing_points(const float* Z, size_t M);
};

// Deep Ensemble
class DeepEnsemble {
private:
    struct EnsembleMember {
        std::vector<std::unique_ptr<BayesianLinear>> layers;
        std::vector<std::unique_ptr<MCDropout>> dropout_layers;
        float learning_rate;
        bool is_trained;
    };
    
    std::vector<EnsembleMember> models;
    size_t num_models;
    size_t input_dim;
    size_t output_dim;
    
    // SWA parameters
    bool enable_swa;
    size_t swa_start;
    float swa_lr;
    std::vector<float> swa_weights;
    size_t swa_updates;
    
public:
    DeepEnsemble(size_t num_models, const std::vector<size_t>& layer_sizes);
    
    // Training
    void train(const Dataset& data, int epochs = 100);
    void train_member(size_t model_idx, const Dataset& data, int epochs);
    
    // Prediction
    void predict(const float* x, float* mean, float* uncertainty) const;
    float predict_single(size_t model_idx, const float* x) const;
    
    // SWA
    void set_enable_swa(size_t swa_start_epoch, float swa_learning_rate);
    void update_swa_weights();
    void apply_swa_weights();
    
    // Utility
    size_t get_num_models() const { return num_models; }
    bool is_trained() const;
};

// Temperature Scaling for Calibration
class TemperatureScaling {
private:
    float temperature;
    bool is_calibrated;
    
public:
    TemperatureScaling() : temperature(1.0f), is_calibrated(false) {}
    
    // Calibration
    void calibrate(const float* logits, const int* targets, 
                   size_t N, size_t num_classes);
    
    // Application
    void apply_temperature(const float* logits, float* scaled_logits, 
                          size_t num_classes) const;
    void apply_temperature_batch(const float* logits, float* scaled_logits,
                                size_t batch_size, size_t num_classes) const;
    
    // Utility
    float get_temperature() const { return temperature; }
    bool is_calibrated_flag() const { return is_calibrated; }
    
    // Evaluation metrics
    float compute_ece(const float* probs, const int* targets, 
                      size_t N, size_t num_classes, int n_bins = 10) const;
    float compute_nll(const float* probs, const int* targets,
                      size_t N, size_t num_classes) const;
};

// Active Learning
class ActiveLearning {
private:
    std::unique_ptr<class BayesianNeuralNetwork> model;
    BayesianConfig::AcquisitionFunction acquisition_type;
    int mc_samples;
    
public:
    enum AcquisitionFunction {
        ENTROPY_SAMPLING,
        MARGIN_SAMPLING,
        BALD
    };
    
    ActiveLearning(class BayesianNeuralNetwork* model, 
                   AcquisitionFunction type = BALD);
    
    // Sample selection
    size_t select_next_sample(const float* X_pool, size_t pool_size, 
                             size_t input_dim) const;
    std::vector<size_t> select_batch_samples(const float* X_pool, size_t pool_size,
                                            size_t input_dim, size_t batch_size) const;
    
    // Acquisition functions
    float compute_entropy(const std::vector<float>& probabilities) const;
    float compute_margin_sampling(const std::vector<float>& probabilities) const;
    float compute_bald(const float* x, size_t input_dim, int T = 20) const;
    
    // Utility
    void set_acquisition_function(BayesianConfig::AcquisitionFunction type) { acquisition_type = type; }
    void set_mc_samples(int samples) { mc_samples = samples; }
};

// Main Bayesian Neural Network Class
class BayesianNeuralNetwork {
private:
    std::vector<std::unique_ptr<BayesianLinear>> layers;
    std::vector<std::unique_ptr<MCDropout>> dropout_layers;
    std::unique_ptr<SparseGaussianProcess> gp_layer;
    std::unique_ptr<DeepEnsemble> ensemble;
    std::unique_ptr<TemperatureScaling> temperature_scaling;
    
    BayesianConfig config;
    size_t input_dim;
    size_t output_dim;
    bool is_trained;
    
    // Training state
    float current_loss;
    float current_elbo;
    int current_epoch;

    friend class BayesianTrainer;
    
public:
    BayesianNeuralNetwork(const std::vector<size_t>& layer_sizes, 
                         const BayesianConfig& config);
    
    // Forward pass
    void forward(const float* input, float* output, bool training = true);
    void forward_with_uncertainty(const float* input, float* mean, 
                                 float* uncertainty);
    
    // Training
    void train(const Dataset& train_data, const Dataset& val_data, 
               int epochs = 100);
    void train_epoch(const Dataset& data);
    float compute_loss(const Dataset& data);
    float compute_elbo(const Dataset& data);
    
    // Prediction
    float predict_single(const float* input);
    std::vector<float> predict_probabilities(const float* input);
    void predict_with_uncertainty(const float* input, float* mean, float* uncertainty);
    
    // Uncertainty estimation
    std::vector<float> monte_carlo_predictions(const float* input, int T = 20);
    float compute_predictive_variance(const float* input, int T = 20);
    
    // Calibration
    void calibrate(const Dataset& val_data);
    bool is_calibrated() const;
    
    // Active learning
    void enable_active_learning();
    size_t select_most_uncertain_sample(const Dataset& pool_data);
    
    // Model management
    void save_model(const std::string& filepath) const;
    void load_model(const std::string& filepath);
    void reset_parameters();
    
    // Add this method to allow weight perturbation
    void perturb_weights(float noise_std = 0.01f) {
        for (auto& layer : layers) {
            layer->perturb_weights(noise_std);
        }
    }

    // Get activations for visualization
    std::vector<std::vector<float>> get_activations(const float* input) {
        std::vector<std::vector<float>> activations;
        std::vector<float> current_input(input, input + input_dim);
        
        for (auto& layer : layers) {
            std::vector<float> output(layer->get_output_size());
            layer->forward(current_input.data(), output.data(), false); // false = inference mode
            activations.push_back(output);
            current_input = output;
        }
        return activations;
    }

    // Utility
    size_t get_input_dim() const { return input_dim; }
    size_t get_output_dim() const { return output_dim; }
    bool get_trained_status() const { return is_trained; }
    const BayesianConfig& get_config() const { return config; }
    
    // Performance metrics
    float compute_nll(const Dataset& test_data) const;
    float compute_ece(const Dataset& test_data, int n_bins = 10) const;
    std::vector<float> compute_calibration_curve(const Dataset& test_data, 
                                                  int n_bins = 10) const;

    // Visualization
    std::vector<std::vector<float>> get_weights() const {
        std::vector<std::vector<float>> all_weights;
        for (const auto& layer : layers) {
            all_weights.push_back(layer->get_weight_mean());
        }
        return all_weights;
    }
};

// Bayesian Trainer
class BayesianTrainer {
private:
    std::unique_ptr<BayesianNeuralNetwork> model;
    BayesianConfig config;
    
    // Training state
    float best_val_loss;
    int patience_counter;
    int max_patience;
    
    // Learning rate scheduling
    float initial_lr;
    float current_lr;
    float lr_decay;
    
public:
    BayesianTrainer(BayesianNeuralNetwork* model, const BayesianConfig& config);
    
    // Training methods
    void train(const Dataset& train_data, const Dataset& val_data, 
               int epochs = 100);
    void train_epoch(const Dataset& data);
    void validate(const Dataset& val_data);
    
    // Learning rate scheduling
    void step_lr();
    void reduce_lr_on_plateau(float val_loss, float factor = 0.5f, 
                              float min_lr = 1e-6f);
    
    // Early stopping
    bool should_stop_early(float val_loss);
    void reset_early_stopping();
    
    // Active learning loop
    void active_learning_loop(Dataset& pool_data, size_t budget, 
                             size_t batch_size = 10);
    
    // Utility
    float get_current_lr() const { return current_lr; }
    int get_current_epoch() const { return model->current_epoch; }
};

// Factory class for creating Bayesian networks
class BayesianNetworkFactory {
public:
    // Create different types of Bayesian networks
    static std::unique_ptr<BayesianNeuralNetwork> create_mlp(
        const std::vector<size_t>& layer_sizes,
        const BayesianConfig& config);
    
    static std::unique_ptr<BayesianNeuralNetwork> create_cnn(
        const std::vector<std::vector<size_t>>& conv_layers,
        const std::vector<size_t>& fc_layers,
        const BayesianConfig& config);
    
    static std::unique_ptr<BayesianNeuralNetwork> create_gp(
        size_t input_dim, size_t num_inducing,
        const BayesianConfig& config);
    
    static std::unique_ptr<BayesianNeuralNetwork> create_ensemble(
        const std::vector<size_t>& layer_sizes,
        const BayesianConfig& config);
    
    // Configuration helpers
    static BayesianConfig get_default_config();
    static BayesianConfig get_mcd_config(float dropout_rate = 0.1f);
    static BayesianConfig get_variational_config(float kl_weight = 1.0f);
    static BayesianConfig get_ensemble_config(int ensemble_size = 5);
    static BayesianConfig get_gp_config(size_t num_inducing = 100);
};

// Utility functions
namespace Utils {
    // Random number generation
    float gaussian_random(float mean = 0.0f, float std = 1.0f);
    void gaussian_random_batch(float* output, size_t size);
    
    // Mathematical utilities
    float compute_entropy(const std::vector<float>& probs);
    float compute_kl_divergence(const std::vector<float>& p, const std::vector<float>& q);
    float compute_mutual_information(const std::vector<float>& joint, 
                                     const std::vector<float>& marginal_x, 
                                     const std::vector<float>& marginal_y);
    
    // Calibration utilities
    float compute_expected_calibration_error(const std::vector<float>& probs, 
                                            const std::vector<int>& targets,
                                            int n_bins = 10);
    std::vector<float> compute_reliability_diagram(const std::vector<float>& probs,
                                                 const std::vector<int>& targets,
                                                 int n_bins = 10);
    
    // Active learning utilities
    std::vector<size_t> select_diverse_samples(const float* X, size_t N, size_t D,
                                               size_t k, float diversity_threshold = 0.8f);
    float compute_diversity_score(const float* x1, const float* x2, size_t D);
    
    // Performance utilities
    double benchmark_inference_time(BayesianNeuralNetwork* model, 
                                    const float* test_input, int iterations = 1000);
    size_t estimate_memory_usage(const BayesianConfig& config, 
                                const std::vector<size_t>& layer_sizes);
    float compute_computational_overhead(const BayesianConfig& config);
}

} // namespace Bayesian
} // namespace ML

#endif // BAYESIAN_NEURAL_NETWORK_H
