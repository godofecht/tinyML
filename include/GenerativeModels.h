//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Generative Models for TinyML - VAE, GAN, Diffusion, Flows, Autoregressive
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 21/01/2026
*****************************************************************************/

#ifndef GENERATIVE_MODELS_H
#define GENERATIVE_MODELS_H

#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>
#include "XSIMDOperations.h"

namespace ML {
namespace Generative {

// Forward declarations
class VAE;
class GAN;
class DiffusionModel;
class NormalizingFlow;
class AutoregressiveModel;
class EnergyBasedModel;
class ConditionalGenerator;

// Utility functions and common types
using Tensor = std::vector<float>;
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
using LatentVector = Tensor;

// Sampling utilities
class Sampler {
public:
    static void gaussian_sample(size_t size, float mean, float std, Tensor& output);
    static void uniform_sample(size_t size, float min, float max, Tensor& output);
    static float gaussian_float(float mean = 0.0f, float std = 1.0f);
    static float uniform_float(float min = 0.0f, float max = 1.0f);
    
    // Reparameterization trick for VAE
    static void reparameterize(const Tensor& mu, const Tensor& logvar, Tensor& output);
    
    // Access to random generator
    static std::mt19937& get_generator();
    
private:
    static std::random_device rd_;
    static std::mt19937 gen_;
};

// Activation functions for generative models
class GenerativeActivations {
public:
    static void leaky_relu(const Tensor& input, Tensor& output, float alpha = 0.2f);
    static void swish(const Tensor& input, Tensor& output);
    static void mish(const Tensor& input, Tensor& output);
    static void gelu(const Tensor& input, Tensor& output);
    static void tanh(const Tensor& input, Tensor& output);
    static void sigmoid(const Tensor& input, Tensor& output);
    
    // Single value versions for internal use
    static float leaky_relu_single(float x, float alpha = 0.2f);
    static float swish_single(float x);
    static float mish_single(float x);
    static float gelu_single(float x);
    static float tanh_single(float x);
    static float sigmoid_single(float x);
};

// Loss functions for generative models
class GenerativeLosses {
public:
    static float binary_cross_entropy(const Tensor& pred, const Tensor& target);
    static float mean_squared_error(const Tensor& pred, const Tensor& target);
    static float kullback_leibler_divergence(const Tensor& mu, const Tensor& logvar);
    static float wasserstein_loss(const Tensor& pred, bool is_real);
    static float hinge_loss(const Tensor& pred, bool is_real);
    static float perceptual_loss(const Tensor& pred, const Tensor& target);
    
    // Single value versions for internal use
    static float hinge_loss_single(float pred, bool is_real);
};

// =============================================================================
// VARIATIONAL AUTOENCODERS (VAE)
// =============================================================================

class VAE {
public:
    struct Config {
        size_t input_dim;
        size_t latent_dim;
        size_t hidden_dim;
        float learning_rate = 0.001f;
        float beta = 1.0f;  // For β-VAE
        bool conditional = false;
        size_t condition_dim = 0;
    };
    
    VAE(const Config& config);
    
    // Forward pass
    void encode(const Tensor& input, Tensor& mu, Tensor& logvar);
    void decode(const LatentVector& latent, Tensor& output);
    void forward(const Tensor& input, Tensor& output, Tensor& mu, Tensor& logvar);
    
    // Training
    float train_step(const Tensor& input, const Tensor& condition = {});
    void generate(const Tensor& condition, Tensor& output, size_t num_samples = 1);
    void interpolate(const LatentVector& z1, const LatentVector& z2, 
                     Tensor& output, float alpha = 0.5f);
    
    // Utility
    float get_reconstruction_loss() const { return recon_loss_; }
    float get_kl_loss() const { return kl_loss_; }
    float get_total_loss() const { return total_loss_; }
    
    // Add this to public section of VAE
    void perturb_weights(float noise_std = 0.01f);
    
    // Accessors for visualization
    const Tensor& get_encoder_w1() const { return encoder_w1_; }
    const Tensor& get_decoder_w1() const { return decoder_w1_; }
    
    // Get intermediate activations for visualization
    std::vector<Tensor> get_activations(const Tensor& input);
    std::vector<Tensor> get_decoder_activations(const LatentVector& latent);

private:
    Config config_;
    
    // Encoder weights
    Tensor encoder_w1_, encoder_b1_;
    Tensor encoder_mu_w_, encoder_mu_b_;
    Tensor encoder_logvar_w_, encoder_logvar_b_;
    
    // Decoder weights
    Tensor decoder_w1_, decoder_b1_;
    Tensor decoder_w2_, decoder_b2_;
    
    // Conditional weights (if enabled)
    Tensor cond_encoder_w_, cond_decoder_w_;
    
    // Loss tracking
    float recon_loss_, kl_loss_, total_loss_;
    
    // Internal methods
    void initialize_weights();
    void encoder_forward(const Tensor& input, const Tensor& condition, 
                        Tensor& hidden, Tensor& mu, Tensor& logvar);
    void decoder_forward(const LatentVector& latent, const Tensor& condition, 
                        Tensor& hidden, Tensor& output);
};

// =============================================================================
// GENERATIVE ADVERSARIAL NETWORKS (GAN)
// =============================================================================

class GAN {
public:
    struct Config {
        size_t latent_dim;
        size_t input_dim;
        size_t hidden_dim;
        float learning_rate = 0.0002f;
        float lambda_gp = 10.0f;  // For WGAN-GP
        std::string gan_type = "standard";  // "standard", "wgan", "stylegan"
        bool conditional = false;
        size_t condition_dim = 0;
    };
    
    GAN(const Config& config);
    
    // Generator methods
    void generate(const Tensor& noise, const Tensor& condition, Tensor& output);
    void generate_batch(size_t batch_size, const Tensor& condition, 
                       std::vector<Tensor>& outputs);
    
    // Discriminator methods
    float discriminate(const Tensor& input, const Tensor& condition);
    std::vector<float> discriminate_batch(const std::vector<Tensor>& inputs, 
                                          const Tensor& condition);
    
    // Training
    void train_step(const std::vector<Tensor>& real_data, const Tensor& condition);
    void train_discriminator(const std::vector<Tensor>& real_data, 
                            const Tensor& condition);
    void train_generator(const Tensor& condition);
    
    // StyleGAN specific methods
    void style_mixing(const Tensor& noise1, const Tensor& noise2, Tensor& output);
    void adaptive_instance_norm(const Tensor& input, const Tensor& style, 
                               Tensor& output);
    
    // Utility
    float get_generator_loss() const { return gen_loss_; }
    float get_discriminator_loss() const { return disc_loss_; }
    bool is_converged() const;
    
private:
    Config config_;
    
    // Generator weights
    Tensor gen_w1_, gen_b1_;
    Tensor gen_w2_, gen_b2_;
    Tensor gen_w3_, gen_b3_;
    
    // Discriminator weights
    Tensor disc_w1_, disc_b1_;
    Tensor disc_w2_, disc_b2_;
    Tensor disc_w3_, disc_b3_;
    
    // StyleGAN specific weights
    Tensor mapping_w_, mapping_b_;
    Tensor style_w_, style_b_;
    
    // Conditional weights
    Tensor cond_gen_w_, cond_disc_w_;
    
    // Loss tracking
    float gen_loss_, disc_loss_;
    int training_step_;
    
    // Internal methods
    void initialize_weights();
    void generator_forward(const Tensor& noise, const Tensor& condition, 
                          Tensor& output);
    float discriminator_forward(const Tensor& input, const Tensor& condition);
    void compute_gradient_penalty(const std::vector<Tensor>& real_data, 
                                 const std::vector<Tensor>& fake_data, 
                                 const Tensor& condition);
};

// =============================================================================
// DIFFUSION MODELS
// =============================================================================

class DiffusionModel {
public:
    struct Config {
        size_t input_dim;
        size_t hidden_dim;
        size_t timesteps = 1000;
        float beta_start = 0.0001f;
        float beta_end = 0.02f;
        std::string schedule = "linear";  // "linear", "cosine", "sigmoid"
        bool conditional = false;
        size_t condition_dim = 0;
    };
    
    DiffusionModel(const Config& config);
    
    // Forward diffusion process
    void forward_diffusion(const Tensor& x0, int t, Tensor& xt, Tensor& epsilon);
    void forward_diffusion_batch(const std::vector<Tensor>& x0, 
                                const std::vector<int>& timesteps,
                                std::vector<Tensor>& xt, 
                                std::vector<Tensor>& epsilon);
    
    // Reverse diffusion (sampling)
    void reverse_diffusion(const Tensor& condition, Tensor& output, 
                          size_t num_samples = 1);
    void sample_step(const Tensor& xt, int t, const Tensor& condition, 
                    Tensor& xtm1);
    
    // Training
    float train_step(const Tensor& x0, const Tensor& condition);
    void train_batch(const std::vector<Tensor>& x0, const Tensor& condition);
    
    // Noise prediction network
    void predict_noise(const Tensor& xt, int t, const Tensor& condition, 
                      Tensor& epsilon_pred);
    
    // Utility
    float get_loss() const { return loss_; }
    std::vector<float> get_betas() const { return betas_; }
    std::vector<float> get_alphas() const { return alphas_; }
    
private:
    Config config_;
    
    // Diffusion schedule parameters
    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alpha_cumprod_;
    std::vector<float> sqrt_alpha_cumprod_;
    std::vector<float> sqrt_one_minus_alpha_cumprod_;
    
    // Noise prediction network weights
    Tensor noise_w1_, noise_b1_;
    Tensor noise_w2_, noise_b2_;
    Tensor noise_w3_, noise_b3_;
    Tensor time_embedding_w_, time_embedding_b_;
    
    // Conditional weights
    Tensor cond_noise_w_;
    
    // Loss tracking
    float loss_;
    
    // Internal methods
    void initialize_diffusion_schedule();
    void initialize_weights();
    void time_embedding(int t, Tensor& embedding);
    void unet_forward(const Tensor& xt, const Tensor& time_emb, 
                     const Tensor& condition, Tensor& epsilon_pred);
};

// =============================================================================
// NORMALIZING FLOWS
// =============================================================================

class NormalizingFlow {
public:
    struct Config {
        size_t data_dim;
        size_t hidden_dim;
        size_t num_flows = 8;
        std::string flow_type = "realnvp";  // "realnvp", "glow", "maf"
        float learning_rate = 0.001f;
    };
    
    NormalizingFlow(const Config& config);
    
    // Forward and inverse transformations
    void forward(const Tensor& z, Tensor& x, Tensor& log_det_jacobian);
    void inverse(const Tensor& x, Tensor& z, Tensor& log_det_jacobian);
    
    // Sampling and density estimation
    void sample(size_t num_samples, std::vector<Tensor>& samples);
    float log_probability(const Tensor& x);
    
    // Training
    float train_step(const std::vector<Tensor>& data);
    void train_batch(const std::vector<Tensor>& data);
    
    // Flow transformations
    void realnvp_forward(const Tensor& z, const Tensor& mask, 
                        Tensor& x, Tensor& log_det);
    void realnvp_inverse(const Tensor& x, const Tensor& mask, 
                        Tensor& z, Tensor& log_det);
    void glow_forward(const Tensor& z, Tensor& x, Tensor& log_det);
    void glow_inverse(const Tensor& x, Tensor& z, Tensor& log_det);
    
    // Utility
    float get_loss() const { return loss_; }
    float get_avg_log_det() const { return avg_log_det_; }
    
private:
    Config config_;
    
    // Flow transformation weights
    std::vector<Tensor> scale_w_, scale_b_;
    std::vector<Tensor> translate_w_, translate_b_;
    
    // Glow specific weights
    std::vector<Tensor> invert_w_, invert_b_;
    std::vector<Tensor> conv_w_, conv_b_;
    
    // Masking for RealNVP
    std::vector<Tensor> masks_;
    
    // Loss tracking
    float loss_;
    float avg_log_det_;
    
    // Internal methods
    void initialize_weights();
    void create_masks();
    void coupling_layer_forward(const Tensor& z, const Tensor& mask,
                               const Tensor& scale_w, const Tensor& scale_b,
                               const Tensor& translate_w, const Tensor& translate_b,
                               Tensor& x, Tensor& log_det);
    void coupling_layer_inverse(const Tensor& x, const Tensor& mask,
                               const Tensor& scale_w, const Tensor& scale_b,
                               const Tensor& translate_w, const Tensor& translate_b,
                               Tensor& z, Tensor& log_det);
};

// =============================================================================
// AUTOREGRESSIVE MODELS
// =============================================================================

class AutoregressiveModel {
public:
    struct Config {
        size_t data_dim;
        size_t hidden_dim;
        size_t num_layers = 2;
        std::string model_type = "pixelcnn";  // "pixelcnn", "wavenet"
        size_t receptive_field = 3;
        float learning_rate = 0.001f;
        int num_categories = 256;  // For discrete data
    };
    
    AutoregressiveModel(const Config& config);
    
    // Forward pass (probability computation)
    void forward(const Tensor& input, Tensor& output_probs);
    float log_probability(const Tensor& input);
    
    // Generation (sampling)
    void generate(Tensor& output, size_t num_samples = 1);
    void generate_autoregressive(Tensor& output);
    
    // Training
    float train_step(const Tensor& input);
    void train_batch(const std::vector<Tensor>& inputs);
    
    // Model-specific methods
    void pixelcnn_forward(const Tensor& input, Tensor& output);
    void wavenet_forward(const Tensor& input, Tensor& output);
    
    // Causal convolutions
    void causal_convolution(const Tensor& input, const Tensor& kernel, 
                           Tensor& output);
    void dilated_convolution(const Tensor& input, const Tensor& kernel,
                            int dilation, Tensor& output);
    
    // Utility
    float get_loss() const { return loss_; }
    float get_perplexity() const { return perplexity_; }
    
private:
    Config config_;
    
    // Convolution weights
    std::vector<Tensor> conv_w_, conv_b_;
    std::vector<Tensor> dilated_conv_w_, dilated_conv_b_;
    
    // Gated activations (for WaveNet)
    std::vector<Tensor> gate_w_, gate_b_;
    std::vector<Tensor> filter_w_, filter_b_;
    
    // Output weights
    Tensor output_w_, output_b_;
    
    // Loss tracking
    float loss_;
    float perplexity_;
    
    // Internal methods
    void initialize_weights();
    void masked_convolution(const Tensor& input, const Tensor& kernel,
                           const Tensor& mask, Tensor& output);
    void gated_activation(const Tensor& input, Tensor& output);
};

// =============================================================================
// ENERGY-BASED MODELS
// =============================================================================

class EnergyBasedModel {
public:
    struct Config {
        size_t data_dim;
        size_t hidden_dim;
        size_t num_layers = 3;
        float learning_rate = 0.001f;
        float temperature = 1.0f;
        std::string sampling_method = "langevin";  // "langevin", "mcmc"
        int num_samples = 100;
        int step_size = 10;
        float noise_std = 0.01f;
    };
    
    EnergyBasedModel(const Config& config);
    
    // Energy computation
    float energy(const Tensor& x);
    void energy_batch(const std::vector<Tensor>& inputs, 
                     std::vector<float>& energies);
    
    // Probability computation
    float probability(const Tensor& x);
    void probability_batch(const std::vector<Tensor>& inputs,
                          std::vector<float>& probs);
    
    // Sampling methods
    void sample_langevin(Tensor& sample, int num_steps = 1000);
    void sample_mcmc(Tensor& sample, int num_steps = 1000);
    void contrastive_divergence(const Tensor& data, Tensor& negative_sample);
    
    // Training
    float train_step(const Tensor& input);
    void train_batch(const std::vector<Tensor>& inputs);
    
    // Contrastive learning
    float contrastive_loss(const Tensor& x1, const Tensor& x2);
    void train_contrastive(const std::vector<Tensor>& pos_pairs,
                           const std::vector<Tensor>& neg_pairs);
    
    // Utility
    float get_loss() const { return loss_; }
    float get_avg_energy() const { return avg_energy_; }
    
private:
    Config config_;
    
    // Energy network weights
    std::vector<Tensor> energy_w_, energy_b_;
    
    // Loss tracking
    float loss_;
    float avg_energy_;
    
    // Internal methods
    void initialize_weights();
    float energy_network_forward(const Tensor& x);
    void langevin_dynamics(const Tensor& x, Tensor& sample, 
                          int num_steps, float step_size, float noise_std);
    void mcmc_step(const Tensor& current, Tensor& next, float temperature);
};

// =============================================================================
// IMPLICIT GENERATIVE MODELS
// =============================================================================

class ImplicitGenerativeModel {
public:
    struct Config {
        size_t latent_dim;
        size_t data_dim;
        size_t hidden_dim;
        float learning_rate = 0.001f;
        std::string model_type = "glo";  // "glo", "deep_generator"
        float regularization_lambda = 1.0f;
    };
    
    ImplicitGenerativeModel(const Config& config);
    
    // Implicit generation
    void generate(const Tensor& noise, Tensor& output);
    void generate_batch(size_t batch_size, std::vector<Tensor>& outputs);
    
    // Implicit optimization
    void optimize_latent(const Tensor& target, Tensor& latent, int max_iter = 1000);
    void optimize_latent_batch(const std::vector<Tensor>& targets,
                              std::vector<Tensor>& latents, int max_iter = 1000);
    
    // Training
    float train_step(const std::vector<Tensor>& real_data);
    void train_batch(const std::vector<Tensor>& real_data);
    
    // GLO specific methods
    void glo_optimization(const Tensor& target, Tensor& latent, Tensor& output);
    float glo_loss(const Tensor& target, const Tensor& output, const Tensor& latent);
    
    // Deep generator specific methods
    void deep_generator_forward(const Tensor& noise, Tensor& output);
    void adversarial_training(const std::vector<Tensor>& real_data);
    
    // Utility
    float get_loss() const { return loss_; }
    float get_optimization_loss() const { return opt_loss_; }
    
private:
    Config config_;
    
    // Generator weights
    Tensor gen_w1_, gen_b1_;
    Tensor gen_w2_, gen_b2_;
    Tensor gen_w3_, gen_b3_;
    
    // Discriminator weights (for adversarial training)
    Tensor disc_w1_, disc_b1_;
    Tensor disc_w2_, disc_b2_;
    
    // Loss tracking
    float loss_;
    float opt_loss_;
    
    // Internal methods
    void initialize_weights();
    void generator_forward(const Tensor& noise, Tensor& output);
    float discriminator_forward(const Tensor& input);
};

// =============================================================================
// CONDITIONAL GENERATION
// =============================================================================

class ConditionalGenerator {
public:
    struct Config {
        size_t condition_dim;
        size_t output_dim;
        size_t latent_dim;
        size_t hidden_dim;
        std::string generation_type = "text_to_image";  // "text_to_image", "image_to_image"
        std::string base_model = "gan";  // "vae", "gan", "diffusion"
        float learning_rate = 0.001f;
    };
    
    ConditionalGenerator(const Config& config);
    
    // Conditional generation
    void generate(const Tensor& condition, Tensor& output);
    void generate_batch(const std::vector<Tensor>& conditions,
                       std::vector<Tensor>& outputs);
    
    // Text-to-image specific
    void text_to_image(const std::string& text, Tensor& image);
    void encode_text(const std::string& text, Tensor& embedding);
    
    // Image-to-image specific
    void image_to_image(const Tensor& input_image, Tensor& output_image);
    void style_transfer(const Tensor& content, const Tensor& style, Tensor& output);
    
    // Training
    float train_step(const Tensor& condition, const Tensor& target);
    void train_batch(const std::vector<Tensor>& conditions,
                    const std::vector<Tensor>& targets);
    
    // Multi-modal generation
    void multi_modal_generate(const std::vector<Tensor>& conditions,
                              std::vector<Tensor>& outputs);
    
    // Utility
    float get_loss() const { return loss_; }
    float get_condition_loss() const { return condition_loss_; }
    
private:
    Config config_;
    
    // Base model (composition)
    std::unique_ptr<VAE> vae_model_;
    std::unique_ptr<GAN> gan_model_;
    std::unique_ptr<DiffusionModel> diffusion_model_;
    
    // Condition processing weights
    Tensor condition_encoder_w_, condition_encoder_b_;
    Tensor condition_attention_w_, condition_attention_b_;
    
    // Text encoding weights
    Tensor text_encoder_w_, text_encoder_b_;
    
    // Loss tracking
    float loss_;
    float condition_loss_;
    
    // Internal methods
    void initialize_weights();
    void process_condition(const Tensor& condition, Tensor& processed_condition);
    void attention_fusion(const Tensor& condition, const Tensor& latent, Tensor& fused);
};

// =============================================================================
// FACTORY CLASS FOR GENERATIVE MODELS
// =============================================================================

class GenerativeModelFactory {
public:
    static std::unique_ptr<VAE> create_vae(const VAE::Config& config);
    static std::unique_ptr<GAN> create_gan(const GAN::Config& config);
    static std::unique_ptr<DiffusionModel> create_diffusion(const DiffusionModel::Config& config);
    static std::unique_ptr<NormalizingFlow> create_flow(const NormalizingFlow::Config& config);
    static std::unique_ptr<AutoregressiveModel> create_autoregressive(const AutoregressiveModel::Config& config);
    static std::unique_ptr<EnergyBasedModel> create_ebm(const EnergyBasedModel::Config& config);
    static std::unique_ptr<ImplicitGenerativeModel> create_implicit(const ImplicitGenerativeModel::Config& config);
    static std::unique_ptr<ConditionalGenerator> create_conditional(const ConditionalGenerator::Config& config);
    
    // Pre-configured models for different use cases
    static std::unique_ptr<VAE> create_lightweight_vae(size_t input_dim, size_t latent_dim);
    static std::unique_ptr<GAN> create_lightweight_gan(size_t latent_dim, size_t output_dim);
    static std::unique_ptr<DiffusionModel> create_lightweight_diffusion(size_t input_dim);
    static std::unique_ptr<ConditionalGenerator> create_text_to_image(size_t vocab_size, size_t image_dim);
};

} // namespace Generative
} // namespace ML

#endif // GENERATIVE_MODELS_H
