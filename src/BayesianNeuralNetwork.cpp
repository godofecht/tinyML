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

} // namespace Bayesian
} // namespace ML
