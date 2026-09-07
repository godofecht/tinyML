//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Generative Models Implementation for TinyML
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 21/01/2026
*****************************************************************************/

#include "GenerativeModels.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace ML {
namespace Generative {

// =============================================================================
// SAMPLER IMPLEMENTATION
// =============================================================================

std::random_device Sampler::rd_;
std::mt19937 Sampler::gen_(rd_());

std::mt19937& Sampler::get_generator() {
    return gen_;
}

void Sampler::gaussian_sample(size_t size, float mean, float std, Tensor& output) {
    output.resize(size);
    std::normal_distribution<float> dist(mean, std);
    for (size_t i = 0; i < size; ++i) {
        output[i] = dist(gen_);
    }
}

void Sampler::uniform_sample(size_t size, float min, float max, Tensor& output) {
    output.resize(size);
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < size; ++i) {
        output[i] = dist(gen_);
    }
}

float Sampler::gaussian_float(float mean, float std) {
    std::normal_distribution<float> dist(mean, std);
    return dist(gen_);
}

float Sampler::uniform_float(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen_);
}

void Sampler::reparameterize(const Tensor& mu, const Tensor& logvar, Tensor& output) {
    output.resize(mu.size());
    for (size_t i = 0; i < mu.size(); ++i) {
        float eps = gaussian_float(0.0f, 1.0f);
        output[i] = mu[i] + std::exp(0.5f * logvar[i]) * eps;
    }
}

// Add this method to VAE implementation (assumed to be in this file or I will append it)
// Since I cannot see the VAE implementation in the snippet, I will append it.

// =============================================================================
// VAE IMPLEMENTATION
// =============================================================================

// (Adding perturb_weights implementation)
void VAE::perturb_weights(float noise_std) {
    for(auto& w : encoder_w1_) w += Sampler::gaussian_float(0.0f, noise_std);
    for(auto& b : encoder_b1_) b += Sampler::gaussian_float(0.0f, noise_std);
    // Add other layers if they exist, for now just w1/b1 for demo
    for(auto& w : decoder_w1_) w += Sampler::gaussian_float(0.0f, noise_std);
    for(auto& b : decoder_b1_) b += Sampler::gaussian_float(0.0f, noise_std);
}
// =============================================================================

void GenerativeActivations::leaky_relu(const Tensor& input, Tensor& output, float alpha) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = (input[i] > 0) ? input[i] : alpha * input[i];
    }
}

void GenerativeActivations::swish(const Tensor& input, Tensor& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
    }
}

void GenerativeActivations::mish(const Tensor& input, Tensor& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float exp_input = std::exp(input[i]);
        float exp_input_squared = exp_input * exp_input;
        output[i] = input[i] * std::tanh(std::log(1.0f + exp_input));
    }
}

void GenerativeActivations::gelu(const Tensor& input, Tensor& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 0.5f * input[i] * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (input[i] + 0.044715f * input[i] * input[i] * input[i])));
    }
}

void GenerativeActivations::tanh(const Tensor& input, Tensor& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
}

void GenerativeActivations::sigmoid(const Tensor& input, Tensor& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

// Single value implementations
float GenerativeActivations::leaky_relu_single(float x, float alpha) {
    return (x > 0) ? x : alpha * x;
}

float GenerativeActivations::swish_single(float x) {
    return x / (1.0f + std::exp(-x));
}

float GenerativeActivations::mish_single(float x) {
    float exp_input = std::exp(x);
    return x * std::tanh(std::log(1.0f + exp_input));
}

float GenerativeActivations::gelu_single(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

float GenerativeActivations::tanh_single(float x) {
    return std::tanh(x);
}

float GenerativeActivations::sigmoid_single(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// =============================================================================
// GENERATIVE LOSSES IMPLEMENTATION
// =============================================================================

float GenerativeLosses::binary_cross_entropy(const Tensor& pred, const Tensor& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float p = std::max(0.0001f, std::min(0.9999f, pred[i]));
        loss += -target[i] * std::log(p) - (1.0f - target[i]) * std::log(1.0f - p);
    }
    return loss / pred.size();
}

float GenerativeLosses::mean_squared_error(const Tensor& pred, const Tensor& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return loss / pred.size();
}

float GenerativeLosses::kullback_leibler_divergence(const Tensor& mu, const Tensor& logvar) {
    float loss = 0.0f;
    for (size_t i = 0; i < mu.size(); ++i) {
        loss += -0.5f * (1.0f + logvar[i] - mu[i] * mu[i] - std::exp(logvar[i]));
    }
    return loss / mu.size();
}

float GenerativeLosses::wasserstein_loss(const Tensor& pred, bool is_real) {
    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        loss += is_real ? -pred[i] : pred[i];
    }
    return loss / pred.size();
}

float GenerativeLosses::hinge_loss(const Tensor& pred, bool is_real) {
    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        if (is_real) {
            loss += std::max(0.0f, 1.0f - pred[i]);
        } else {
            loss += std::max(0.0f, 1.0f + pred[i]);
        }
    }
    return loss / pred.size();
}

float GenerativeLosses::perceptual_loss(const Tensor& pred, const Tensor& target) {
    // Simplified perceptual loss using MSE as placeholder
    return mean_squared_error(pred, target);
}

// Single value implementations
float GenerativeLosses::hinge_loss_single(float pred, bool is_real) {
    if (is_real) {
        return std::max(0.0f, 1.0f - pred);
    } else {
        return std::max(0.0f, 1.0f + pred);
    }
}

// =============================================================================
// VAE IMPLEMENTATION
// =============================================================================

VAE::VAE(const Config& config) : config_(config), recon_loss_(0.0f), kl_loss_(0.0f), total_loss_(0.0f) {
    initialize_weights();
}

void VAE::initialize_weights() {
    // Initialize encoder weights
    encoder_w1_.resize(config_.input_dim * config_.hidden_dim);
    encoder_b1_.resize(config_.hidden_dim);
    encoder_mu_w_.resize(config_.hidden_dim * config_.latent_dim);
    encoder_mu_b_.resize(config_.latent_dim);
    encoder_logvar_w_.resize(config_.hidden_dim * config_.latent_dim);
    encoder_logvar_b_.resize(config_.latent_dim);
    
    // Initialize decoder weights
    decoder_w1_.resize(config_.latent_dim * config_.hidden_dim);
    decoder_b1_.resize(config_.hidden_dim);
    decoder_w2_.resize(config_.hidden_dim * config_.input_dim);
    decoder_b2_.resize(config_.input_dim);
    
    // Initialize with small random values
    auto init_weight = [](Tensor& weight) {
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = Sampler::gaussian_float(0.0f, 0.02f);
        }
    };
    
    auto init_bias = [](Tensor& bias) {
        std::fill(bias.begin(), bias.end(), 0.0f);
    };
    
    init_weight(encoder_w1_);
    init_bias(encoder_b1_);
    init_weight(encoder_mu_w_);
    init_bias(encoder_mu_b_);
    init_weight(encoder_logvar_w_);
    init_bias(encoder_logvar_b_);
    init_weight(decoder_w1_);
    init_bias(decoder_b1_);
    init_weight(decoder_w2_);
    init_bias(decoder_b2_);
    
    // Conditional weights
    if (config_.conditional) {
        cond_encoder_w_.resize(config_.condition_dim * config_.hidden_dim);
        cond_decoder_w_.resize(config_.condition_dim * config_.hidden_dim);
        init_weight(cond_encoder_w_);
        init_weight(cond_decoder_w_);
    }
}

void VAE::encoder_forward(const Tensor& input, const Tensor& condition, 
                         Tensor& hidden, Tensor& mu, Tensor& logvar) {
    // Input to hidden layer
    hidden.resize(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = encoder_b1_[i];
        for (size_t j = 0; j < config_.input_dim; ++j) {
            sum += input[j] * encoder_w1_[j * config_.hidden_dim + i];
        }
        // Add condition if available
        if (config_.conditional && !condition.empty()) {
            for (size_t j = 0; j < config_.condition_dim; ++j) {
                sum += condition[j] * cond_encoder_w_[j * config_.hidden_dim + i];
            }
        }
        hidden[i] = std::tanh(sum);
    }
    
    // Hidden to mu and logvar
    mu.resize(config_.latent_dim);
    logvar.resize(config_.latent_dim);
    for (size_t i = 0; i < config_.latent_dim; ++i) {
        float mu_sum = encoder_mu_b_[i];
        float logvar_sum = encoder_logvar_b_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            mu_sum += hidden[j] * encoder_mu_w_[j * config_.latent_dim + i];
            logvar_sum += hidden[j] * encoder_logvar_w_[j * config_.latent_dim + i];
        }
        mu[i] = mu_sum;
        logvar[i] = logvar_sum;
    }
}

void VAE::decoder_forward(const LatentVector& latent, const Tensor& condition, 
                         Tensor& hidden, Tensor& output) {
    // Latent to hidden layer
    hidden.resize(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = decoder_b1_[i];
        for (size_t j = 0; j < config_.latent_dim; ++j) {
            sum += latent[j] * decoder_w1_[j * config_.hidden_dim + i];
        }
        // Add condition if available
        if (config_.conditional && !condition.empty()) {
            for (size_t j = 0; j < config_.condition_dim; ++j) {
                sum += condition[j] * cond_decoder_w_[j * config_.hidden_dim + i];
            }
        }
        hidden[i] = std::tanh(sum);
    }
    
    // Hidden to output
    output.resize(config_.input_dim);
    for (size_t i = 0; i < config_.input_dim; ++i) {
        float sum = decoder_b2_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            sum += hidden[j] * decoder_w2_[j * config_.input_dim + i];
        }
        output[i] = std::tanh(sum);
    }
}

void VAE::encode(const Tensor& input, Tensor& mu, Tensor& logvar) {
    if (input.empty()) {
        mu.assign(config_.latent_dim, 0.0f);
        logvar.assign(config_.latent_dim, 0.0f);
        return;
    }
    Tensor hidden;
    encoder_forward(input, {}, hidden, mu, logvar);
}

void VAE::decode(const LatentVector& latent, Tensor& output) {
    Tensor hidden;
    decoder_forward(latent, {}, hidden, output);
}

std::vector<Tensor> VAE::get_activations(const Tensor& input) {
    std::vector<Tensor> activations;
    
    // 1. Encoder Hidden Layer
    Tensor hidden;
    hidden.resize(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = encoder_b1_[i];
        for (size_t j = 0; j < config_.input_dim; ++j) {
            if (j < input.size()) // Safety check
                sum += input[j] * encoder_w1_[j * config_.hidden_dim + i];
        }
        hidden[i] = std::tanh(sum);
    }
    activations.push_back(hidden);
    
    // 2. Mu and LogVar (Latent Parameters)
    Tensor mu(config_.latent_dim), logvar(config_.latent_dim);
    for (size_t i = 0; i < config_.latent_dim; ++i) {
        float mu_sum = encoder_mu_b_[i];
        float logvar_sum = encoder_logvar_b_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            mu_sum += hidden[j] * encoder_mu_w_[j * config_.latent_dim + i];
            logvar_sum += hidden[j] * encoder_logvar_w_[j * config_.latent_dim + i];
        }
        mu[i] = mu_sum;
        logvar[i] = logvar_sum;
    }
    activations.push_back(mu); // Only pushing Mu for visualization simplicity
    // activations.push_back(logvar); 
    
    // 3. Latent Vector (Sampled)
    Tensor latent;
    Sampler::reparameterize(mu, logvar, latent);
    activations.push_back(latent);
    
    // 4. Decoder Hidden Layer
    Tensor dec_hidden;
    dec_hidden.resize(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = decoder_b1_[i];
        for (size_t j = 0; j < config_.latent_dim; ++j) {
            sum += latent[j] * decoder_w1_[j * config_.hidden_dim + i];
        }
        dec_hidden[i] = std::tanh(sum);
    }
    activations.push_back(dec_hidden);
    
    // 5. Output Layer
    Tensor output;
    output.resize(config_.input_dim);
    for (size_t i = 0; i < config_.input_dim; ++i) {
        float sum = decoder_b2_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            sum += dec_hidden[j] * decoder_w2_[j * config_.input_dim + i];
        }
        output[i] = std::tanh(sum);
    }
    activations.push_back(output);
    
    return activations;
}

std::vector<Tensor> VAE::get_decoder_activations(const LatentVector& latent) {
    std::vector<Tensor> activations;
    
    // 1. Latent Vector
    activations.push_back(latent);
    
    // 2. Decoder Hidden Layer
    Tensor dec_hidden;
    dec_hidden.resize(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = decoder_b1_[i];
        for (size_t j = 0; j < config_.latent_dim; ++j) {
            sum += latent[j] * decoder_w1_[j * config_.hidden_dim + i];
        }
        dec_hidden[i] = std::tanh(sum);
    }
    activations.push_back(dec_hidden);
    
    // 3. Output Layer
    Tensor output;
    output.resize(config_.input_dim);
    for (size_t i = 0; i < config_.input_dim; ++i) {
        float sum = decoder_b2_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            sum += dec_hidden[j] * decoder_w2_[j * config_.input_dim + i];
        }
        output[i] = std::tanh(sum);
    }
    activations.push_back(output);
    
    return activations;
}

void VAE::forward(const Tensor& input, Tensor& output, Tensor& mu, Tensor& logvar) {
    Tensor hidden;
    encoder_forward(input, {}, hidden, mu, logvar);
    
    Tensor latent;
    Sampler::reparameterize(mu, logvar, latent);
    
    decoder_forward(latent, {}, hidden, output);
}

float VAE::train_step(const Tensor& input, const Tensor& condition) {
    Tensor mu, logvar, output;
    
    if (condition.empty()) {
        forward(input, output, mu, logvar);
    } else {
        Tensor hidden;
        encoder_forward(input, condition, hidden, mu, logvar);
        
        Tensor latent;
        Sampler::reparameterize(mu, logvar, latent);
        
        decoder_forward(latent, condition, hidden, output);
    }
    
    // Calculate losses
    recon_loss_ = GenerativeLosses::mean_squared_error(output, input);
    kl_loss_ = GenerativeLosses::kullback_leibler_divergence(mu, logvar);
    total_loss_ = recon_loss_ + config_.beta * kl_loss_;
    
    return total_loss_;
}

void VAE::generate(const Tensor& condition, Tensor& output, size_t num_samples) {
    Tensor latent;
    Sampler::gaussian_sample(config_.latent_dim, 0.0f, 1.0f, latent);
    
    Tensor hidden;
    decoder_forward(latent, condition, hidden, output);
}

void VAE::interpolate(const LatentVector& z1, const LatentVector& z2, 
                     Tensor& output, float alpha) {
    Tensor latent(z1.size());
    for (size_t i = 0; i < z1.size(); ++i) {
        latent[i] = (1.0f - alpha) * z1[i] + alpha * z2[i];
    }
    
    Tensor hidden;
    decoder_forward(latent, {}, hidden, output);
}

// =============================================================================
// GAN IMPLEMENTATION
// =============================================================================

GAN::GAN(const Config& config) : config_(config), gen_loss_(0.0f), disc_loss_(0.0f), training_step_(0) {
    initialize_weights();
}

void GAN::initialize_weights() {
    // Initialize generator weights
    gen_w1_.resize(config_.latent_dim * config_.hidden_dim);
    gen_b1_.resize(config_.hidden_dim);
    gen_w2_.resize(config_.hidden_dim * config_.hidden_dim);
    gen_b2_.resize(config_.hidden_dim);
    gen_w3_.resize(config_.hidden_dim * config_.input_dim);
    gen_b3_.resize(config_.input_dim);
    
    // Initialize discriminator weights
    disc_w1_.resize(config_.input_dim * config_.hidden_dim);
    disc_b1_.resize(config_.hidden_dim);
    disc_w2_.resize(config_.hidden_dim * config_.hidden_dim);
    disc_b2_.resize(config_.hidden_dim);
    disc_w3_.resize(config_.hidden_dim * 1);
    disc_b3_.resize(1);
    
    auto init_weight = [](Tensor& weight) {
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = Sampler::gaussian_float(0.0f, 0.02f);
        }
    };
    
    auto init_bias = [](Tensor& bias) {
        std::fill(bias.begin(), bias.end(), 0.0f);
    };
    
    init_weight(gen_w1_);
    init_bias(gen_b1_);
    init_weight(gen_w2_);
    init_bias(gen_b2_);
    init_weight(gen_w3_);
    init_bias(gen_b3_);
    
    init_weight(disc_w1_);
    init_bias(disc_b1_);
    init_weight(disc_w2_);
    init_bias(disc_b2_);
    init_weight(disc_w3_);
    init_bias(disc_b3_);
    
    // Conditional weights
    if (config_.conditional) {
        cond_gen_w_.resize(config_.condition_dim * config_.hidden_dim);
        cond_disc_w_.resize(config_.condition_dim * config_.hidden_dim);
        init_weight(cond_gen_w_);
        init_weight(cond_disc_w_);
    }
    
    // StyleGAN specific weights
    if (config_.gan_type == "stylegan") {
        mapping_w_.resize(config_.latent_dim * config_.hidden_dim);
        mapping_b_.resize(config_.hidden_dim);
        style_w_.resize(config_.hidden_dim * config_.hidden_dim);
        style_b_.resize(config_.hidden_dim);
        init_weight(mapping_w_);
        init_bias(mapping_b_);
        init_weight(style_w_);
        init_bias(style_b_);
    }
}

void GAN::generator_forward(const Tensor& noise, const Tensor& condition, Tensor& output) {
    // Layer 1: noise to hidden
    Tensor hidden1(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = gen_b1_[i];
        for (size_t j = 0; j < config_.latent_dim; ++j) {
            sum += noise[j] * gen_w1_[j * config_.hidden_dim + i];
        }
        if (config_.conditional && !condition.empty()) {
            for (size_t j = 0; j < config_.condition_dim; ++j) {
                sum += condition[j] * cond_gen_w_[j * config_.hidden_dim + i];
            }
        }
        hidden1[i] = GenerativeActivations::leaky_relu_single(sum);
    }
    
    // Layer 2: hidden to hidden
    Tensor hidden2(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = gen_b2_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            sum += hidden1[j] * gen_w2_[j * config_.hidden_dim + i];
        }
        hidden2[i] = GenerativeActivations::leaky_relu_single(sum);
    }
    
    // Layer 3: hidden to output
    output.resize(config_.input_dim);
    for (size_t i = 0; i < config_.input_dim; ++i) {
        float sum = gen_b3_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            sum += hidden2[j] * gen_w3_[j * config_.input_dim + i];
        }
        output[i] = std::tanh(sum);
    }
}

float GAN::discriminator_forward(const Tensor& input, const Tensor& condition) {
    // Layer 1: input to hidden
    Tensor hidden1(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = disc_b1_[i];
        for (size_t j = 0; j < config_.input_dim; ++j) {
            sum += input[j] * disc_w1_[j * config_.hidden_dim + i];
        }
        if (config_.conditional && !condition.empty()) {
            for (size_t j = 0; j < config_.condition_dim; ++j) {
                sum += condition[j] * cond_disc_w_[j * config_.hidden_dim + i];
            }
        }
        hidden1[i] = GenerativeActivations::leaky_relu_single(sum);
    }
    
    // Layer 2: hidden to hidden
    Tensor hidden2(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = disc_b2_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            sum += hidden1[j] * disc_w2_[j * config_.hidden_dim + i];
        }
        hidden2[i] = GenerativeActivations::leaky_relu_single(sum);
    }
    
    // Layer 3: hidden to output (single value)
    float sum = disc_b3_[0];
    for (size_t j = 0; j < config_.hidden_dim; ++j) {
        sum += hidden2[j] * disc_w3_[j * 1];
    }
    
    return sum;
}

void GAN::generate(const Tensor& noise, const Tensor& condition, Tensor& output) {
    generator_forward(noise, condition, output);
}

void GAN::generate_batch(size_t batch_size, const Tensor& condition, std::vector<Tensor>& outputs) {
    outputs.resize(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        Tensor noise;
        Sampler::gaussian_sample(config_.latent_dim, 0.0f, 1.0f, noise);
        generator_forward(noise, condition, outputs[i]);
    }
}

float GAN::discriminate(const Tensor& input, const Tensor& condition) {
    return discriminator_forward(input, condition);
}

std::vector<float> GAN::discriminate_batch(const std::vector<Tensor>& inputs, const Tensor& condition) {
    std::vector<float> outputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        outputs[i] = discriminator_forward(inputs[i], condition);
    }
    return outputs;
}

void GAN::train_step(const std::vector<Tensor>& real_data, const Tensor& condition) {
    train_discriminator(real_data, condition);
    train_generator(condition);
    training_step_++;
}

void GAN::train_discriminator(const std::vector<Tensor>& real_data, const Tensor& condition) {
    // Train on real data
    std::vector<float> real_scores = discriminate_batch(real_data, condition);
    
    // Generate fake data
    std::vector<Tensor> fake_data(real_data.size());
    generate_batch(real_data.size(), condition, fake_data);
    
    // Train on fake data
    std::vector<float> fake_scores = discriminate_batch(fake_data, condition);
    
    // Calculate discriminator loss
    float real_loss = 0.0f, fake_loss = 0.0f;
    for (size_t i = 0; i < real_scores.size(); ++i) {
        if (config_.gan_type == "wgan") {
            real_loss += -real_scores[i];
            fake_loss += fake_scores[i];
        } else {
            real_loss += GenerativeLosses::hinge_loss_single(real_scores[i], true);
            fake_loss += GenerativeLosses::hinge_loss_single(fake_scores[i], false);
        }
    }
    
    disc_loss_ = (real_loss + fake_loss) / real_scores.size();
}

void GAN::train_generator(const Tensor& condition) {
    // Generate fake data
    std::vector<Tensor> fake_data(1);
    generate_batch(1, condition, fake_data);
    
    // Train generator to fool discriminator
    std::vector<float> fake_scores = discriminate_batch(fake_data, condition);
    
    float gen_loss = 0.0f;
    for (size_t i = 0; i < fake_scores.size(); ++i) {
        if (config_.gan_type == "wgan") {
            gen_loss += -fake_scores[i];
        } else {
            gen_loss += GenerativeLosses::hinge_loss_single(fake_scores[i], true);
        }
    }
    
    gen_loss_ = gen_loss / fake_scores.size();
}

bool GAN::is_converged() const {
    return training_step_ > 1000 && std::abs(gen_loss_ - disc_loss_) < 0.1f;
}

// =============================================================================
// DIFFUSION MODEL IMPLEMENTATION
// =============================================================================

DiffusionModel::DiffusionModel(const Config& config) : config_(config), loss_(0.0f) {
    initialize_diffusion_schedule();
    initialize_weights();
}

void DiffusionModel::initialize_diffusion_schedule() {
    betas_.resize(config_.timesteps);
    alphas_.resize(config_.timesteps);
    alpha_cumprod_.resize(config_.timesteps);
    sqrt_alpha_cumprod_.resize(config_.timesteps);
    sqrt_one_minus_alpha_cumprod_.resize(config_.timesteps);
    
    for (int t = 0; t < config_.timesteps; ++t) {
        if (config_.schedule == "linear") {
            betas_[t] = config_.beta_start + (config_.beta_end - config_.beta_start) * t / (config_.timesteps - 1);
        } else if (config_.schedule == "cosine") {
            float alpha_cumprod = std::cos((t / config_.timesteps + 0.008) * M_PI / 2) / std::cos(0.008 * M_PI / 2);
            betas_[t] = 1.0f - alpha_cumprod / (t > 0 ? alpha_cumprod_[t-1] : 1.0f);
        } else {
            betas_[t] = config_.beta_start + (config_.beta_end - config_.beta_start) * t / (config_.timesteps - 1);
        }
        
        alphas_[t] = 1.0f - betas_[t];
        alpha_cumprod_[t] = t > 0 ? alpha_cumprod_[t-1] * alphas_[t] : alphas_[t];
        sqrt_alpha_cumprod_[t] = std::sqrt(alpha_cumprod_[t]);
        sqrt_one_minus_alpha_cumprod_[t] = std::sqrt(1.0f - alpha_cumprod_[t]);
    }
}

void DiffusionModel::initialize_weights() {
    // Initialize noise prediction network weights
    noise_w1_.resize(config_.input_dim * config_.hidden_dim);
    noise_b1_.resize(config_.hidden_dim);
    noise_w2_.resize(config_.hidden_dim * config_.hidden_dim);
    noise_b2_.resize(config_.hidden_dim);
    noise_w3_.resize(config_.hidden_dim * config_.input_dim);
    noise_b3_.resize(config_.input_dim);
    
    // Time embedding weights
    time_embedding_w_.resize(128 * config_.hidden_dim);
    time_embedding_b_.resize(config_.hidden_dim);
    
    auto init_weight = [](Tensor& weight) {
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = Sampler::gaussian_float(0.0f, 0.02f);
        }
    };
    
    auto init_bias = [](Tensor& bias) {
        std::fill(bias.begin(), bias.end(), 0.0f);
    };
    
    init_weight(noise_w1_);
    init_bias(noise_b1_);
    init_weight(noise_w2_);
    init_bias(noise_b2_);
    init_weight(noise_w3_);
    init_bias(noise_b3_);
    init_weight(time_embedding_w_);
    init_bias(time_embedding_b_);
    
    // Conditional weights
    if (config_.conditional) {
        cond_noise_w_.resize(config_.condition_dim * config_.hidden_dim);
        init_weight(cond_noise_w_);
    }
}

void DiffusionModel::time_embedding(int t, Tensor& embedding) {
    embedding.resize(128);
    
    // Sinusoidal time embedding
    for (int i = 0; i < 64; ++i) {
        float freq = std::pow(10000.0f, -2.0f * i / 128.0f);
        embedding[2*i] = std::sin(t * freq);
        embedding[2*i+1] = std::cos(t * freq);
    }
}

void DiffusionModel::unet_forward(const Tensor& xt, const Tensor& time_emb, 
                                 const Tensor& condition, Tensor& epsilon_pred) {
    // Time embedding to hidden
    Tensor time_hidden(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = time_embedding_b_[i];
        for (size_t j = 0; j < 128; ++j) {
            sum += time_emb[j] * time_embedding_w_[j * config_.hidden_dim + i];
        }
        time_hidden[i] = GenerativeActivations::gelu_single(sum);
    }
    
    // Layer 1: xt + time embedding to hidden
    Tensor hidden1(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = noise_b1_[i] + time_hidden[i];
        for (size_t j = 0; j < config_.input_dim; ++j) {
            sum += xt[j] * noise_w1_[j * config_.hidden_dim + i];
        }
        if (config_.conditional && !condition.empty()) {
            for (size_t j = 0; j < config_.condition_dim; ++j) {
                sum += condition[j] * cond_noise_w_[j * config_.hidden_dim + i];
            }
        }
        hidden1[i] = GenerativeActivations::gelu_single(sum);
    }
    
    // Layer 2: hidden to hidden
    Tensor hidden2(config_.hidden_dim);
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float sum = noise_b2_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            sum += hidden1[j] * noise_w2_[j * config_.hidden_dim + i];
        }
        hidden2[i] = GenerativeActivations::gelu_single(sum);
    }
    
    // Layer 3: hidden to output
    epsilon_pred.resize(config_.input_dim);
    for (size_t i = 0; i < config_.input_dim; ++i) {
        float sum = noise_b3_[i];
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            sum += hidden2[j] * noise_w3_[j * config_.input_dim + i];
        }
        epsilon_pred[i] = sum;
    }
}

void DiffusionModel::forward_diffusion(const Tensor& x0, int t, Tensor& xt, Tensor& epsilon) {
    xt.resize(x0.size());
    epsilon.resize(x0.size());
    
    // Sample noise
    Sampler::gaussian_sample(x0.size(), 0.0f, 1.0f, epsilon);
    
    // Apply forward diffusion
    float sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_[t];
    float sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_[t];
    
    for (size_t i = 0; i < x0.size(); ++i) {
        xt[i] = sqrt_alpha_cumprod_t * x0[i] + sqrt_one_minus_alpha_cumprod_t * epsilon[i];
    }
}

void DiffusionModel::predict_noise(const Tensor& xt, int t, const Tensor& condition, Tensor& epsilon_pred) {
    Tensor time_emb;
    time_embedding(t, time_emb);
    unet_forward(xt, time_emb, condition, epsilon_pred);
}

float DiffusionModel::train_step(const Tensor& x0, const Tensor& condition) {
    // Sample random timestep
    std::uniform_int_distribution<int> t_dist(0, config_.timesteps - 1);
    int t = t_dist(Sampler::get_generator());
    
    // Forward diffusion
    Tensor xt, epsilon;
    forward_diffusion(x0, t, xt, epsilon);
    
    // Predict noise
    Tensor epsilon_pred;
    predict_noise(xt, t, condition, epsilon_pred);
    
    // Calculate loss
    loss_ = GenerativeLosses::mean_squared_error(epsilon_pred, epsilon);
    
    return loss_;
}

void DiffusionModel::reverse_diffusion(const Tensor& condition, Tensor& output, size_t num_samples) {
    // Start with pure noise
    Tensor xt;
    Sampler::gaussian_sample(config_.input_dim, 0.0f, 1.0f, xt);
    
    // Reverse diffusion process
    for (int t = config_.timesteps - 1; t >= 0; --t) {
        sample_step(xt, t, condition, xt);
    }
    
    output = xt;
}

void DiffusionModel::sample_step(const Tensor& xt, int t, const Tensor& condition, Tensor& xtm1) {
    if (t == 0) {
        // Final step, no noise added
        Tensor epsilon_pred;
        predict_noise(xt, t, condition, epsilon_pred);
        
        xtm1.resize(xt.size());
        for (size_t i = 0; i < xt.size(); ++i) {
            xtm1[i] = (xt[i] - (1.0f - alphas_[t]) / std::sqrt(1.0f - alpha_cumprod_[t]) * epsilon_pred[i]) / std::sqrt(alphas_[t]);
        }
    } else {
        // Intermediate step with noise
        Tensor epsilon_pred;
        predict_noise(xt, t, condition, epsilon_pred);
        
        Tensor noise;
        Sampler::gaussian_sample(xt.size(), 0.0f, 1.0f, noise);
        
        xtm1.resize(xt.size());
        for (size_t i = 0; i < xt.size(); ++i) {
            float mean = (xt[i] - (1.0f - alphas_[t]) / std::sqrt(1.0f - alpha_cumprod_[t]) * epsilon_pred[i]) / std::sqrt(alphas_[t]);
            xtm1[i] = mean + std::sqrt(betas_[t]) * noise[i];
        }
    }
}

// =============================================================================
// FACTORY IMPLEMENTATION
// =============================================================================

std::unique_ptr<VAE> GenerativeModelFactory::create_vae(const VAE::Config& config) {
    return std::make_unique<VAE>(config);
}

std::unique_ptr<GAN> GenerativeModelFactory::create_gan(const GAN::Config& config) {
    return std::make_unique<GAN>(config);
}

std::unique_ptr<DiffusionModel> GenerativeModelFactory::create_diffusion(const DiffusionModel::Config& config) {
    return std::make_unique<DiffusionModel>(config);
}

std::unique_ptr<VAE> GenerativeModelFactory::create_lightweight_vae(size_t input_dim, size_t latent_dim) {
    VAE::Config config;
    config.input_dim = input_dim;
    config.latent_dim = latent_dim;
    config.hidden_dim = std::min(input_dim, 256UL);
    config.learning_rate = 0.001f;
    config.beta = 1.0f;
    config.conditional = false;
    return create_vae(config);
}

std::unique_ptr<GAN> GenerativeModelFactory::create_lightweight_gan(size_t latent_dim, size_t output_dim) {
    GAN::Config config;
    config.latent_dim = latent_dim;
    config.input_dim = output_dim;
    config.hidden_dim = std::min(output_dim, 256UL);
    config.learning_rate = 0.0002f;
    config.gan_type = "standard";
    config.conditional = false;
    return create_gan(config);
}

std::unique_ptr<DiffusionModel> GenerativeModelFactory::create_lightweight_diffusion(size_t input_dim) {
    DiffusionModel::Config config;
    config.input_dim = input_dim;
    config.hidden_dim = std::min(input_dim, 256UL);
    config.timesteps = 100;  // Reduced for lightweight
    config.beta_start = 0.0001f;
    config.beta_end = 0.02f;
    config.schedule = "linear";
    config.conditional = false;
    return create_diffusion(config);
}

} // namespace Generative
} // namespace ML
