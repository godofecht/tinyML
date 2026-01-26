//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Bayesian Neural Networks Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
 *****************************************************************************/

#include "BayesianNeuralNetwork.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace ML {
namespace Bayesian {

// ==================== BayesianLinear Implementation ====================

BayesianLinear::BayesianLinear(size_t in_features, size_t out_features, 
                               bool local_reparam)
    : input_size(in_features), output_size(out_features), 
      use_local_reparameterization(local_reparam), is_training(true),
      rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      normal_dist(0.0f, 1.0f) {
    
    // Initialize parameters
    size_t weight_count = input_size * output_size;
    weight_mean.resize(weight_count);
    weight_logvar.resize(weight_count);
    sampled_weights.resize(weight_count);
    grad_weight_mean.resize(weight_count);
    grad_weight_logvar.resize(weight_count);
    
    bias_mean.resize(output_size);
    bias_logvar.resize(output_size);
    sampled_bias.resize(output_size);
    grad_bias_mean.resize(output_size);
    grad_bias_logvar.resize(output_size);
    
    reset_parameters();
}

void BayesianLinear::reset_parameters() {
    // Initialize weights with Xavier initialization
    float std_dev = std::sqrt(2.0f / (input_size + output_size));
    
    for (size_t i = 0; i < weight_mean.size(); ++i) {
        weight_mean[i] = normal_dist(rng) * std_dev;
        weight_logvar[i] = std::log(0.1f); // Small initial variance
    }
    
    for (size_t i = 0; i < bias_mean.size(); ++i) {
        bias_mean[i] = 0.0f;
        bias_logvar[i] = std::log(0.1f);
    }
}

void BayesianLinear::perturb_weights(float noise_std) {
    std::normal_distribution<float> noise_dist(0.0f, noise_std);
    for (size_t i = 0; i < weight_mean.size(); ++i) {
        weight_mean[i] += noise_dist(rng);
    }
    for (size_t i = 0; i < bias_mean.size(); ++i) {
        bias_mean[i] += noise_dist(rng);
    }
}

void BayesianLinear::sample_weights() {
    if (!is_training) {
        // Use mean weights during inference
        std::copy(weight_mean.begin(), weight_mean.end(), sampled_weights.begin());
        std::copy(bias_mean.begin(), bias_mean.end(), sampled_bias.begin());
        return;
    }
    
    // Sample weights using reparameterization trick
    for (size_t i = 0; i < weight_mean.size(); ++i) {
        float std_dev = std::exp(0.5f * weight_logvar[i]);
        float epsilon = normal_dist(rng);
        sampled_weights[i] = weight_mean[i] + std_dev * epsilon;
    }
    
    for (size_t i = 0; i < bias_mean.size(); ++i) {
        float std_dev = std::exp(0.5f * bias_logvar[i]);
        float epsilon = normal_dist(rng);
        sampled_bias[i] = bias_mean[i] + std_dev * epsilon;
    }
}

void BayesianLinear::forward(const float* input, float* output, bool training) {
    bool was_training = is_training;
    is_training = training;
    
    sample_weights();
    
    // Matrix-vector multiplication: y = x * W + b
    for (size_t i = 0; i < output_size; ++i) {
        output[i] = sampled_bias[i];
        
        for (size_t j = 0; j < input_size; ++j) {
            output[i] += input[j] * sampled_weights[i * input_size + j];
        }
    }
    
    is_training = was_training;
}

float BayesianLinear::compute_kl_divergence() const {
    float kl = 0.0f;
    // Simplified KL divergence for Gaussian approximation
    for (size_t i = 0; i < weight_mean.size(); ++i) {
        float std = std::exp(0.5f * weight_logvar[i]);
        float var = std * std;
        // KL(q(w)||p(w)) where p(w) ~ N(0, 1)
        kl += 0.5f * (weight_mean[i] * weight_mean[i] + var - 1.0f - weight_logvar[i]);
    }
    return kl;
}

// ==================== BayesianNeuralNetwork Constructor ====================
BayesianNeuralNetwork::BayesianNeuralNetwork(const std::vector<size_t>& layer_sizes,
                                             const BayesianConfig& cfg)
    : config(cfg),
      input_dim(layer_sizes.empty() ? 0 : layer_sizes.front()),
      output_dim(layer_sizes.empty() ? 0 : layer_sizes.back()),
      is_trained(false),
      current_loss(0.0f),
      current_elbo(0.0f),
      current_epoch(0) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("BayesianNeuralNetwork requires at least input and output sizes");
    }
    for (size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
        layers.emplace_back(std::make_unique<BayesianLinear>(
            layer_sizes[i], layer_sizes[i + 1], config.use_local_reparameterization));
        if (config.use_monte_carlo_dropout && i + 1 < layer_sizes.size() - 1) {
            // Optional: attach dropout after hidden layers. Avoid instantiation if not implemented elsewhere.
            // dropout_layers.emplace_back(std::make_unique<MCDropout>(config.dropout_rate));
        }
    }
}

// ==================== BayesianNeuralNetwork Forward Methods ====================
void BayesianNeuralNetwork::forward(const float* input, float* output, bool training) {
    std::vector<float> current(input_dim);
    for (size_t i = 0; i < input_dim; ++i) current[i] = input[i];
    for (const auto& layer : layers) {
        std::vector<float> next(layer->get_output_size());
        layer->forward(current.data(), next.data(), training);
        current.swap(next);
    }
    if (!current.empty()) {
        *output = current[0];
    } else {
        *output = 0.0f;
    }
}

void BayesianNeuralNetwork::forward_with_uncertainty(const float* input, float* mean, float* uncertainty) {
    if (config.use_monte_carlo_dropout && config.mc_samples > 1) {
        std::vector<float> preds(config.mc_samples);
        for (int t = 0; t < config.mc_samples; ++t) {
            float out = 0.0f;
            forward(input, &out, true);
            preds[t] = out;
        }
        float sum = std::accumulate(preds.begin(), preds.end(), 0.0f);
        float mu = sum / static_cast<float>(preds.size());
        float sq_sum = 0.0f;
        for (float p : preds) {
            float diff = p - mu;
            sq_sum += diff * diff;
        }
        float var = sq_sum / static_cast<float>(preds.size());
        *mean = mu;
        *uncertainty = std::sqrt(var);
    } else {
        float out = 0.0f;
        forward(input, &out, false);
        *mean = out;
        *uncertainty = 0.0f;
    }
}

// ==================== BayesianNetworkFactory Implementation ====================

std::unique_ptr<BayesianNeuralNetwork> BayesianNetworkFactory::create_mlp(
    const std::vector<size_t>& layer_sizes, const BayesianConfig& config) {
    return std::make_unique<BayesianNeuralNetwork>(layer_sizes, config);
}

// ==================== Dataset Implementation ====================
void Dataset::add_sample(const float* x, float y_scalar) {
    X.insert(X.end(), x, x + input_dim);
    y.push_back(y_scalar);
}

void Dataset::add_sample(const float* x, int label) {
    X.insert(X.end(), x, x + input_dim);
    y.push_back(static_cast<float>(label));
    labels.push_back(label);
}

// ==================== BayesianLinear Backprop & Update ====================

void BayesianLinear::backward(const float* input, const float* grad_output, float* grad_input) {
    for (size_t i = 0; i < output_size; ++i) {
        float go = grad_output[i];
        
        // Bias gradients
        grad_bias_mean[i] += go;
        // Bias logvar gradient: dL/dlogvar = dL/dy * dy/dlogvar = go * (sampled - mean) * 0.5
        grad_bias_logvar[i] += go * (sampled_bias[i] - bias_mean[i]) * 0.5f;
        
        for (size_t j = 0; j < input_size; ++j) {
            float in = input[j];
            // Weight gradients
            grad_weight_mean[i * input_size + j] += go * in;
            
            // Weight logvar gradient
            size_t idx = i * input_size + j;
            grad_weight_logvar[idx] += go * in * (sampled_weights[idx] - weight_mean[idx]) * 0.5f;
            
            // Input gradients (for previous layer)
            if (grad_input) {
                // Use sampled_weights used in forward pass
                grad_input[j] += go * sampled_weights[i * input_size + j]; 
            }
        }
    }
}

void BayesianLinear::update_parameters(float learning_rate) {
    for (size_t i = 0; i < weight_mean.size(); ++i) {
        weight_mean[i] -= learning_rate * grad_weight_mean[i];
        grad_weight_mean[i] = 0.0f; // Reset
        
        weight_logvar[i] -= learning_rate * grad_weight_logvar[i];
        grad_weight_logvar[i] = 0.0f;
    }
    for (size_t i = 0; i < bias_mean.size(); ++i) {
        bias_mean[i] -= learning_rate * grad_bias_mean[i];
        grad_bias_mean[i] = 0.0f; // Reset
        
        bias_logvar[i] -= learning_rate * grad_bias_logvar[i];
        grad_bias_logvar[i] = 0.0f;
    }
}

// ==================== BayesianNeuralNetwork Training Methods ====================

float BayesianNeuralNetwork::compute_loss(const Dataset& data) {
    float total_loss = 0.0f;
    size_t n_samples = data.size();
    if (n_samples == 0) return 0.0f;
    
    std::vector<float> input_vec(input_dim);
    float output;
    
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < input_dim; ++j) {
            input_vec[j] = data.X[i * input_dim + j];
        }
        forward(input_vec.data(), &output, false);
        float diff = output - data.y[i];
        total_loss += diff * diff;
    }
    return total_loss / n_samples;
}

void BayesianNeuralNetwork::train_epoch(const Dataset& data) {
    float learning_rate = 0.01f;
    size_t n_samples = data.size();
    if (n_samples == 0) return;
    
    std::vector<float> input_vec(input_dim);
    std::vector<std::vector<float>> layer_inputs; // Store inputs for backward pass
    
    for (size_t i = 0; i < n_samples; ++i) {
        // Prepare input
        for (size_t j = 0; j < input_dim; ++j) {
            input_vec[j] = data.X[i * input_dim + j];
        }
        
        // Forward pass (store intermediates)
        layer_inputs.clear();
        layer_inputs.push_back(input_vec);
        
        std::vector<float> current = input_vec;
        for (const auto& layer : layers) {
            std::vector<float> next(layer->get_output_size());
            layer->forward(current.data(), next.data(), true);
            layer_inputs.push_back(next); // Input to next layer (which is output of current)
            current = next;
        }
        
        // Adjust layer_inputs: 
        // layer_inputs[0] is input to layers[0]
        // layer_inputs[1] is output of layers[0] == input to layers[1]
        // ...
        
        float prediction = current[0]; // Assuming single output
        float target = data.y[i];
        
        // Backward pass
        float loss_grad = 2.0f * (prediction - target); // MSE gradient
        
        std::vector<float> grad_output = {loss_grad};
        std::vector<float> grad_input;
        
        for (int l = layers.size() - 1; l >= 0; --l) {
            size_t in_size = layers[l]->get_input_size();
            grad_input.assign(in_size, 0.0f);
            
            // Input to this layer is stored in layer_inputs[l]
            layers[l]->backward(layer_inputs[l].data(), grad_output.data(), grad_input.data());
            
            grad_output = grad_input;
        }
        
        // Update weights (SGD)
        for (const auto& layer : layers) {
            layer->update_parameters(learning_rate);
        }
    }
}

} // namespace Bayesian
} // namespace ML
