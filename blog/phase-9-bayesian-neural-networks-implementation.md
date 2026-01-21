# Phase 9: Bayesian Neural Networks - Implementation Guide

*Published: January 21, 2026*  
*Category: Engineering*  
*Tags: Bayesian Neural Networks, Uncertainty, TinyML*

---

## Overview

Phase 9 introduces Bayesian Neural Networks to the TinyML framework, providing uncertainty quantification with minimal computational overhead. The implementation focuses on practical, real-time capable methods that can run on edge devices.

## Core Components

### 1. Bayesian Neural Layers

#### Bayesian Linear Layer Implementation

```cpp
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
    
    size_t input_size, output_size;
    bool use_local_reparameterization;
    
public:
    BayesianLinear(size_t in_features, size_t out_features, 
                   bool local_reparam = true);
    
    // Forward pass with weight sampling
    void forward(const float* input, float* output, bool training = true);
    
    // KL divergence computation for regularization
    float compute_kl_divergence() const;
    
    // Variational parameters update
    void update_parameters(const float* grad_mean, const float* grad_logvar, 
                          float learning_rate);
};
```

#### Reparameterization Trick

```cpp
void sample_weights() {
    if (use_local_reparameterization) {
        // Local reparameterization for efficiency
        for (size_t i = 0; i < weight_mean.size(); ++i) {
            float std_dev = std::exp(0.5f * weight_logvar[i]);
            float epsilon = gaussian_random();
            sampled_weights[i] = weight_mean[i] + std_dev * epsilon;
        }
    } else {
        // Standard reparameterization
        for (size_t i = 0; i < weight_mean.size(); ++i) {
            float std_dev = std::exp(0.5f * weight_logvar[i]);
            float epsilon = gaussian_random();
            sampled_weights[i] = weight_mean[i] + std_dev * epsilon;
        }
    }
}
```

### 2. Monte Carlo Dropout Implementation

#### Dropout Layer with Bayesian Interpretation

```cpp
class MCDropout {
private:
    float dropout_rate;
    std::vector<bool> mask;
    std::vector<float> scaled_output;
    bool is_training;
    
public:
    MCDropout(float rate = 0.1f);
    
    void forward(const float* input, float* output, size_t size);
    void set_training_mode(bool training) { is_training = training; }
    
    // Monte Carlo inference
    void mc_inference(const float* input, float* mean_output, 
                     float* uncertainty_output, size_t size, int T = 20);
};
```

#### Monte Carlo Inference

```cpp
void mc_inference(const float* input, float* mean_output, 
                 float* uncertainty_output, size_t size, int T) {
    std::vector<float> predictions(T * size);
    std::vector<float> temp_output(size);
    
    // Enable dropout during inference
    bool original_training = is_training;
    is_training = true;
    
    // Perform T forward passes
    for (int t = 0; t < T; ++t) {
        forward(input, temp_output.data(), size);
        std::copy(temp_output.begin(), temp_output.end(), 
                 predictions.begin() + t * size);
    }
    
    // Compute mean and variance across samples
    for (size_t i = 0; i < size; ++i) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int t = 0; t < T; ++t) {
            float pred = predictions[t * size + i];
            sum += pred;
            sum_sq += pred * pred;
        }
        
        mean_output[i] = sum / T;
        float variance = (sum_sq / T) - (mean_output[i] * mean_output[i]);
        uncertainty_output[i] = std::sqrt(std::max(0.0f, variance));
    }
    
    is_training = original_training;
}
```

### 3. Gaussian Process Implementation

#### Sparse Gaussian Process

```cpp
class SparseGaussianProcess {
private:
    // Inducing points
    std::vector<float> inducing_inputs;
    std::vector<float> inducing_mean;
    std::vector<float> inducing_cov;
    
    // Kernel parameters
    float lengthscale;
    float signal_variance;
    float noise_variance;
    
    size_t num_inducing;
    size_t input_dim;
    
public:
    SparseGaussianProcess(size_t M, size_t D);
    
    // Kernel computation
    float rbf_kernel(const float* x1, const float* x2) const;
    void compute_kernel_matrix(const float* X1, const float* X2, 
                              size_t N1, size_t N2, float* K) const;
    
    // Sparse GP inference
    void predict(const float* X_test, size_t N_test,
                float* mean, float* variance) const;
    
    // Variational inference
    float compute_elbo(const float* X, const float* y, size_t N) const;
};
```

#### RBF Kernel Implementation

```cpp
float rbf_kernel(const float* x1, const float* x2) const {
    float squared_distance = 0.0f;
    for (size_t i = 0; i < input_dim; ++i) {
        float diff = x1[i] - x2[i];
        squared_distance += diff * diff;
    }
    
    return signal_variance * std::exp(-0.5f * squared_distance / 
                                     (lengthscale * lengthscale));
}
```

### 4. Ensemble Methods

#### Deep Ensemble Implementation

```cpp
class DeepEnsemble {
private:
    std::vector<std::unique_ptr<NeuralNetwork>> models;
    size_t num_models;
    
public:
    DeepEnsemble(size_t num_models, const NetworkConfig& config);
    
    void train(const float* X, const float* y, size_t N, size_t epochs);
    
    void predict(const float* x, float* mean, float* uncertainty) const;
    
    // Stochastic Weight Averaging
    void enable_swa(size_t swa_start, float swa_lr);
};
```

#### Ensemble Prediction

```cpp
void predict(const float* x, float* mean, float* uncertainty) const {
    std::vector<float> predictions(num_models);
    
    // Get predictions from all models
    for (size_t i = 0; i < num_models; ++i) {
        predictions[i] = models[i]->forward_single(x);
    }
    
    // Compute ensemble statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (float pred : predictions) {
        sum += pred;
        sum_sq += pred * pred;
    }
    
    *mean = sum / num_models;
    float variance = (sum_sq / num_models) - (*mean * *mean);
    *uncertainty = std::sqrt(std::max(0.0f, variance));
}
```

### 5. Calibration Methods

#### Temperature Scaling

```cpp
class TemperatureScaling {
private:
    float temperature;
    bool is_calibrated;
    
public:
    TemperatureScaling() : temperature(1.0f), is_calibrated(false) {}
    
    void calibrate(const float* logits, const int* targets, 
                   size_t N, size_t num_classes);
    
    void apply_temperature(const float* logits, float* scaled_logits, 
                          size_t num_classes) const;
    
    float get_temperature() const { return temperature; }
};
```

#### Calibration Optimization

```cpp
void calibrate(const float* logits, const int* targets, 
               size_t N, size_t num_classes) {
    // Optimize temperature using gradient descent
    float lr = 0.01f;
    float temp = 1.0f;
    
    for (int epoch = 0; epoch < 100; ++epoch) {
        float grad = 0.0f;
        float nll = 0.0f;
        
        for (size_t i = 0; i < N; ++i) {
            const float* logit = logits + i * num_classes;
            int target = targets[i];
            
            // Apply temperature and compute softmax
            std::vector<float> probs(num_classes);
            float max_logit = *std::max_element(logit, logit + num_classes);
            float sum_exp = 0.0f;
            
            for (size_t j = 0; j < num_classes; ++j) {
                float scaled_logit = logit[j] / temp;
                probs[j] = std::exp(scaled_logit - max_logit / temp);
                sum_exp += probs[j];
            }
            
            for (size_t j = 0; j < num_classes; ++j) {
                probs[j] /= sum_exp;
            }
            
            // Compute NLL and gradient
            nll += -std::log(std::max(1e-8f, probs[target]));
            
            for (size_t j = 0; j < num_classes; ++j) {
                float indicator = (j == target) ? 1.0f : 0.0f;
                grad += (probs[j] - indicator) * logit[j] / (temp * temp);
            }
        }
        
        // Update temperature
        temp -= lr * grad / N;
        temp = std::max(0.1f, std::min(10.0f, temp));
    }
    
    temperature = temp;
    is_calibrated = true;
}
```

### 6. Active Learning

#### Uncertainty-Based Sampling

```cpp
class ActiveLearning {
private:
    std::unique_ptr<BayesianNeuralNetwork> model;
    AcquisitionFunction acquisition_type;
    
public:
    enum AcquisitionFunction {
        ENTROPY_SAMPLING,
        MARGIN_SAMPLING,
        BALD
    };
    
    ActiveLearning(BayesianNeuralNetwork* model, AcquisitionFunction type);
    
    size_t select_next_sample(const float* X_pool, size_t pool_size, 
                             size_t input_dim) const;
    
    float compute_entropy(const std::vector<float>& probabilities) const;
    float compute_bald(const float* x, size_t input_dim, int T = 20) const;
};
```

#### BALD Acquisition Function

```cpp
float compute_bald(const float* x, size_t input_dim, int T) const {
    std::vector<std::vector<float>> predictions(T);
    std::vector<float> mean_prediction;
    
    // Get T predictions with MC dropout
    for (int t = 0; t < T; ++t) {
        predictions[t] = model->predict_probabilities(x, input_dim);
    }
    
    // Compute mean prediction
    size_t num_classes = predictions[0].size();
    mean_prediction.resize(num_classes, 0.0f);
    
    for (int t = 0; t < T; ++t) {
        for (size_t c = 0; c < num_classes; ++c) {
            mean_prediction[c] += predictions[t][c] / T;
        }
    }
    
    // Compute mutual information
    float conditional_entropy = 0.0f;
    
    for (int t = 0; t < T; ++t) {
        conditional_entropy += compute_entropy(predictions[t]);
    }
    conditional_entropy /= T;
    
    float predictive_entropy = compute_entropy(mean_prediction);
    
    return predictive_entropy - conditional_entropy;
}
```

## Performance Optimization

### 1. Efficient Sampling

#### Vectorized Gaussian Sampling

```cpp
class GaussianSampler {
private:
    std::vector<float> ziggurat_table;
    
public:
    // SIMD-optimized Gaussian sampling
    void sample_batch(float* output, size_t size);
    
    // Box-Muller transform for pairs
    void box_muller_pair(float& z1, float& z2);
};
```

#### Adaptive Sampling

```cpp
class AdaptiveSampler {
private:
    float uncertainty_threshold;
    int min_samples, max_samples;
    
public:
    int determine_sample_count(float uncertainty) const {
        if (uncertainty < uncertainty_threshold) {
            return min_samples;
        } else {
            return std::min(max_samples, 
                           static_cast<int>(uncertainty * 10));
        }
    }
};
```

### 2. Memory Optimization

#### Parameter Sharing

```cpp
class ParameterSharing {
private:
    std::unordered_map<size_t, std::vector<size_t>> shared_groups;
    
public:
    void find_shared_parameters(const std::vector<float>& weights, 
                               float similarity_threshold);
    
    void apply_sharing(std::vector<float>& weights);
    
    size_t count_shared_parameters() const;
};
```

#### Gradient Checkpointing

```cpp
class GradientCheckpointing {
private:
    std::vector<std::vector<float>> checkpoints;
    bool use_checkpointing;
    
public:
    void save_checkpoint(const std::vector<float>& activations);
    void restore_checkpoint(std::vector<float>& activations);
    
    void enable_checkpointing(bool enable) { use_checkpointing = enable; }
};
```

## Integration with TinyML Framework

### 1. Bayesian Network Factory

```cpp
class BayesianNetworkFactory {
public:
    static std::unique_ptr<BayesianNeuralNetwork> create(
        const std::string& type,
        const NetworkConfig& config);
    
    static std::unique_ptr<BayesianNeuralNetwork> create_mlp(
        const std::vector<size_t>& layer_sizes,
        const BayesianConfig& config);
    
    static std::unique_ptr<BayesianNeuralNetwork> create_cnn(
        const std::vector<ConvLayerConfig>& conv_layers,
        const BayesianConfig& config);
};
```

### 2. Configuration Management

```cpp
struct BayesianConfig {
    float prior_mean = 0.0f;
    float prior_std = 1.0f;
    float kl_weight = 1.0f;
    bool use_monte_carlo_dropout = true;
    float dropout_rate = 0.1f;
    int mc_samples = 20;
    bool use_local_reparameterization = true;
    bool enable_calibration = true;
    AcquisitionFunction acquisition_type = BALD;
};
```

### 3. Training Pipeline

```cpp
class BayesianTrainer {
private:
    std::unique_ptr<BayesianNeuralNetwork> model;
    BayesianConfig config;
    
public:
    void train(const Dataset& train_data, const Dataset& val_data);
    
    void train_epoch(const Dataset& data);
    float compute_elbo(const Dataset& data);
    void update_parameters(float learning_rate);
    
    // Active learning loop
    void active_learning_loop(Dataset& pool_data, 
                             size_t budget, size_t batch_size);
};
```

## Real-Time Deployment

### 1. Edge Optimization

#### Quantized Bayesian Layers

```cpp
class QuantizedBayesianLinear : public BayesianLinear {
private:
    std::vector<uint8_t> quantized_mean;
    std::vector<uint8_t> quantized_logvar;
    float scale_mean, scale_logvar;
    
public:
    void quantize_parameters();
    void dequantize_parameters();
    
    void forward_quantized(const float* input, float* output);
};
```

#### Fixed-Point Arithmetic

```cpp
class FixedPointBayesian {
private:
    int fractional_bits;
    
public:
    int32_t float_to_fixed(float x) const;
    float fixed_to_float(int32_t x) const;
    
    void fixed_point_gaussian_sample(int32_t* output, size_t size);
};
```

### 2. Streaming Inference

```cpp
class StreamingBayesianInference {
private:
    std::queue<float> input_buffer;
    std::queue<float> uncertainty_buffer;
    
public:
    void process_stream(const float* input_stream, size_t length);
    void get_predictions(float* predictions, float* uncertainties);
    
    void set_buffer_size(size_t size);
};
```

## Testing and Validation

### 1. Unit Tests

```cpp
// Test Bayesian layer forward pass
TEST(BayesianLayerTest, ForwardPass) {
    BayesianLinear layer(10, 5);
    
    std::vector<float> input(10, 1.0f);
    std::vector<float> output(5);
    
    layer.forward(input.data(), output.data(), true);
    
    // Verify output is reasonable
    for (float val : output) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// Test uncertainty estimation
TEST(UncertaintyTest, MonteCarloDropout) {
    MCDropout dropout(0.1f);
    
    std::vector<float> input(100, 0.5f);
    std::vector<float> mean(100), uncertainty(100);
    
    dropout.mc_inference(input.data(), mean.data(), uncertainty.data(), 100, 50);
    
    // Verify uncertainty is positive
    for (float val : uncertainty) {
        EXPECT_GE(val, 0.0f);
    }
}
```

### 2. Integration Tests

```cpp
// Test complete Bayesian pipeline
TEST(BayesianPipelineTest, EndToEnd) {
    BayesianConfig config;
    config.use_monte_carlo_dropout = true;
    config.mc_samples = 20;
    
    auto model = BayesianNetworkFactory::create_mlp({10, 20, 1}, config);
    
    // Generate test data
    Dataset data = generate_synthetic_data(1000, 10);
    
    // Train model
    BayesianTrainer trainer(model.get(), config);
    trainer.train(data.train, data.val);
    
    // Test uncertainty estimation
    std::vector<float> test_input(10);
    float mean, uncertainty;
    model->predict_with_uncertainty(test_input.data(), &mean, &uncertainty);
    
    EXPECT_GT(uncertainty, 0.0f);
}
```

### 3. Performance Benchmarks

```cpp
// Benchmark computational overhead
TEST(BayesianBenchmark, ComputationalOverhead) {
    const int iterations = 1000;
    const size_t input_size = 1000;
    
    // Benchmark deterministic model
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        deterministic_forward(input_data, output_data, input_size);
    }
    auto deterministic_time = measure_time(start);
    
    // Benchmark Bayesian model
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        bayesian_forward(input_data, output_data, input_size);
    }
    auto bayesian_time = measure_time(start);
    
    double overhead = bayesian_time / deterministic_time;
    EXPECT_LT(overhead, 1.05) << "Overhead should be <5%";
}
```

## Usage Examples

### 1. Basic Bayesian Neural Network

```cpp
#include "BayesianNeuralNetwork.h"

int main() {
    // Create Bayesian MLP
    BayesianConfig config;
    config.use_monte_carlo_dropout = true;
    config.dropout_rate = 0.1f;
    config.mc_samples = 20;
    
    auto model = BayesianNetworkFactory::create_mlp({784, 256, 128, 10}, config);
    
    // Train model
    Dataset mnist_data = load_mnist();
    BayesianTrainer trainer(model.get(), config);
    trainer.train(mnist_data.train, mnist_data.val);
    
    // Make prediction with uncertainty
    std::vector<float> test_image(784);
    float prediction, uncertainty;
    model->predict_with_uncertainty(test_image.data(), &prediction, &uncertainty);
    
    std::cout << "Prediction: " << prediction 
              << ", Uncertainty: " << uncertainty << std::endl;
    
    return 0;
}
```

### 2. Active Learning Loop

```cpp
void active_learning_example() {
    // Initialize with small labeled set
    Dataset labeled_data = get_initial_labeled_data(100);
    Dataset unlabeled_pool = get_unlabeled_pool(10000);
    
    BayesianConfig config;
    config.acquisition_type = BALD;
    
    auto model = BayesianNetworkFactory::create_mlp({100, 50, 1}, config);
    ActiveLearning active_learner(model.get(), BALD);
    
    for (int iteration = 0; iteration < 100; ++iteration) {
        // Train current model
        BayesianTrainer trainer(model.get(), config);
        trainer.train(labeled_data.train, labeled_data.val);
        
        // Select most uncertain sample
        size_t selected_idx = active_learner.select_next_sample(
            unlabeled_pool.X.data(), unlabeled_pool.size(), 100);
        
        // Get label (simulated)
        int label = get_ground_truth_label(unlabeled_pool.X.data() + selected_idx * 100);
        
        // Add to labeled set
        labeled_data.add_sample(unlabeled_pool.X.data() + selected_idx * 100, label);
        unlabeled_pool.remove_sample(selected_idx);
        
        std::cout << "Iteration " << iteration 
                  << ": Labeled samples = " << labeled_data.size() << std::endl;
    }
}
```

### 3. Ensemble Method

```cpp
void ensemble_example() {
    // Create deep ensemble
    NetworkConfig config = create_mlp_config({100, 50, 1});
    DeepEnsemble ensemble(5, config);
    
    // Train ensemble
    Dataset data = generate_regression_data(1000, 100);
    ensemble.train(data.X, data.y, data.size, 100);
    
    // Make ensemble prediction
    std::vector<float> test_input(100);
    float mean, uncertainty;
    ensemble.predict(test_input.data(), &mean, &uncertainty);
    
    std::cout << "Ensemble prediction: " << mean 
              << " ± " << uncertainty << std::endl;
}
```

## Performance Targets

### Computational Requirements

| Method | Memory Overhead | Time Overhead | Accuracy Impact |
|--------|----------------|---------------|-----------------|
| MC Dropout | 0% | 2-3% | Minimal |
| Variational | 100% | 10-15% | Moderate |
| Ensemble | 500% | 500% | High |
| Sparse GP | 50% | 50% | High |

### Real-Time Targets

- **Latency**: <5ms for 1000-sample inference
- **Memory**: <10MB total footprint
- **Accuracy**: <5% accuracy loss vs deterministic
- **Calibration**: ECE < 0.05

## Conclusion

The Phase 9 implementation provides a comprehensive Bayesian neural network framework that maintains real-time performance while adding uncertainty quantification. The modular design allows selecting appropriate methods based on computational constraints and accuracy requirements.

The implementation focuses on practical methods like Monte Carlo dropout and ensembles that provide good uncertainty estimates with minimal overhead, while also including advanced methods like variational inference and Gaussian processes for applications requiring higher accuracy.

---

*Next: [Phase 9 Test Suite](../tests/test_phase9_bayesian.cpp)*
