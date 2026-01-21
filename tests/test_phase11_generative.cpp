//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Generative Models Test Suite for TinyML
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 21/01/2026
*****************************************************************************/

#include <gtest/gtest.h>
#include "GenerativeModels.h"
#include <chrono>
#include <random>

using namespace ML::Generative;

class GenerativeModelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        test_data_ = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
        condition_ = {1.0f, 0.5f, 0.0f};
    }
    
    Tensor test_data_;
    Tensor condition_;
};

// =============================================================================
// SAMPLER TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, SamplerGaussianSample) {
    Tensor output;
    Sampler::gaussian_sample(1000, 0.0f, 1.0f, output);
    
    EXPECT_EQ(output.size(), 1000);
    
    // Check mean and std dev approximately
    float mean = 0.0f;
    for (float val : output) {
        mean += val;
    }
    mean /= output.size();
    
    EXPECT_NEAR(mean, 0.0f, 0.1f);  // Should be close to 0
}

TEST_F(GenerativeModelsTest, SamplerUniformSample) {
    Tensor output;
    Sampler::uniform_sample(100, 0.0f, 1.0f, output);
    
    EXPECT_EQ(output.size(), 100);
    
    // Check range
    for (float val : output) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
}

TEST_F(GenerativeModelsTest, SamplerReparameterize) {
    Tensor mu = {0.0f, 1.0f, 2.0f};
    Tensor logvar = {-1.0f, -0.5f, 0.0f};
    Tensor output;
    
    Sampler::reparameterize(mu, logvar, output);
    
    EXPECT_EQ(output.size(), 3);
    
    // Check that output is different from mu (due to noise)
    bool different = false;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::abs(output[i] - mu[i]) > 0.1f) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

// =============================================================================
// ACTIVATION TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, GenerativeActivationsLeakyReLU) {
    Tensor input = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    Tensor output;
    
    GenerativeActivations::leaky_relu(input, output, 0.2f);
    
    EXPECT_EQ(output.size(), 5);
    EXPECT_FLOAT_EQ(output[0], -0.2f);  // -1.0 * 0.2
    EXPECT_FLOAT_EQ(output[1], -0.1f);  // -0.5 * 0.2
    EXPECT_FLOAT_EQ(output[2], 0.0f);   // 0.0
    EXPECT_FLOAT_EQ(output[3], 0.5f);   // 0.5
    EXPECT_FLOAT_EQ(output[4], 1.0f);   // 1.0
}

TEST_F(GenerativeModelsTest, GenerativeActivationsSigmoid) {
    Tensor input = {-2.0f, 0.0f, 2.0f};
    Tensor output;
    
    GenerativeActivations::sigmoid(input, output);
    
    EXPECT_EQ(output.size(), 3);
    EXPECT_NEAR(output[0], 0.119f, 0.01f);  // sigmoid(-2)
    EXPECT_NEAR(output[1], 0.5f, 0.01f);    // sigmoid(0)
    EXPECT_NEAR(output[2], 0.881f, 0.01f);  // sigmoid(2)
}

// =============================================================================
// LOSS TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, GenerativeLossesMSE) {
    Tensor pred = {1.0f, 2.0f, 3.0f};
    Tensor target = {1.5f, 2.5f, 2.5f};
    
    float loss = GenerativeLosses::mean_squared_error(pred, target);
    
    EXPECT_NEAR(loss, 0.25f, 0.01f);  // ((0.5^2 + 0.5^2 + 0.5^2) / 3)
}

TEST_F(GenerativeModelsTest, GenerativeLossesKLDivergence) {
    Tensor mu = {0.0f, 0.0f};
    Tensor logvar = {-1.0f, -1.0f};
    
    float kl_loss = GenerativeLosses::kullback_leibler_divergence(mu, logvar);
    
    EXPECT_GT(kl_loss, 0.0f);  // KL should be positive
}

// =============================================================================
// VAE TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, VAEConstruction) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 2;
    config.hidden_dim = 16;
    config.beta = 1.0f;
    config.conditional = false;
    
    VAE vae(config);
    
    EXPECT_EQ(vae.get_total_loss(), 0.0f);
}

TEST_F(GenerativeModelsTest, VAEEncodeDecode) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 2;
    config.hidden_dim = 16;
    
    VAE vae(config);
    
    Tensor mu, logvar;
    vae.encode(test_data_, mu, logvar);
    
    EXPECT_EQ(mu.size(), 2);
    EXPECT_EQ(logvar.size(), 2);
    
    Tensor output;
    vae.decode(mu, output);
    
    EXPECT_EQ(output.size(), 10);
}

TEST_F(GenerativeModelsTest, VAEForward) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 2;
    config.hidden_dim = 16;
    
    VAE vae(config);
    
    Tensor output, mu, logvar;
    vae.forward(test_data_, output, mu, logvar);
    
    EXPECT_EQ(output.size(), 10);
    EXPECT_EQ(mu.size(), 2);
    EXPECT_EQ(logvar.size(), 2);
}

TEST_F(GenerativeModelsTest, VAETraining) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 2;
    config.hidden_dim = 16;
    
    VAE vae(config);
    
    float loss = vae.train_step(test_data_);
    
    EXPECT_GT(loss, 0.0f);
    EXPECT_GT(vae.get_reconstruction_loss(), 0.0f);
    EXPECT_GT(vae.get_kl_loss(), 0.0f);
    EXPECT_GT(vae.get_total_loss(), 0.0f);
}

TEST_F(GenerativeModelsTest, VAEGeneration) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 2;
    config.hidden_dim = 16;
    
    VAE vae(config);
    
    Tensor output;
    vae.generate({}, output);
    
    EXPECT_EQ(output.size(), 10);
}

TEST_F(GenerativeModelsTest, VAEInterpolation) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 2;
    config.hidden_dim = 16;
    
    VAE vae(config);
    
    Tensor z1 = {0.0f, 0.0f};
    Tensor z2 = {1.0f, 1.0f};
    Tensor output;
    
    vae.interpolate(z1, z2, output, 0.5f);
    
    EXPECT_EQ(output.size(), 10);
}

// =============================================================================
// GAN TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, GANConstruction) {
    GAN::Config config;
    config.latent_dim = 2;
    config.input_dim = 10;
    config.hidden_dim = 16;
    config.gan_type = "standard";
    config.conditional = false;
    
    GAN gan(config);
    
    EXPECT_EQ(gan.get_generator_loss(), 0.0f);
    EXPECT_EQ(gan.get_discriminator_loss(), 0.0f);
}

TEST_F(GenerativeModelsTest, GANGeneration) {
    GAN::Config config;
    config.latent_dim = 2;
    config.input_dim = 10;
    config.hidden_dim = 16;
    
    GAN gan(config);
    
    Tensor noise = {0.5f, -0.5f};
    Tensor output;
    
    gan.generate(noise, {}, output);
    
    EXPECT_EQ(output.size(), 10);
}

TEST_F(GenerativeModelsTest, GANDiscrimination) {
    GAN::Config config;
    config.latent_dim = 2;
    config.input_dim = 10;
    config.hidden_dim = 16;
    
    GAN gan(config);
    
    float score = gan.discriminate(test_data_, {});
    
    // Should return a single float score
    EXPECT_TRUE(std::isfinite(score));
}

TEST_F(GenerativeModelsTest, GANBatchGeneration) {
    GAN::Config config;
    config.latent_dim = 2;
    config.input_dim = 10;
    config.hidden_dim = 16;
    
    GAN gan(config);
    
    std::vector<Tensor> outputs;
    gan.generate_batch(5, {}, outputs);
    
    EXPECT_EQ(outputs.size(), 5);
    for (const auto& output : outputs) {
        EXPECT_EQ(output.size(), 10);
    }
}

TEST_F(GenerativeModelsTest, GANTraining) {
    GAN::Config config;
    config.latent_dim = 2;
    config.input_dim = 10;
    config.hidden_dim = 16;
    
    GAN gan(config);
    
    std::vector<Tensor> real_data = {test_data_, test_data_};
    
    gan.train_step(real_data, {});
    
    EXPECT_GT(gan.get_generator_loss(), 0.0f);
    EXPECT_GT(gan.get_discriminator_loss(), 0.0f);
}

// =============================================================================
// DIFFUSION MODEL TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, DiffusionModelConstruction) {
    DiffusionModel::Config config;
    config.input_dim = 10;
    config.hidden_dim = 16;
    config.timesteps = 100;
    config.conditional = false;
    
    DiffusionModel diffusion(config);
    
    EXPECT_EQ(diffusion.get_loss(), 0.0f);
    EXPECT_EQ(diffusion.get_betas().size(), 100);
    EXPECT_EQ(diffusion.get_alphas().size(), 100);
}

TEST_F(GenerativeModelsTest, DiffusionForwardProcess) {
    DiffusionModel::Config config;
    config.input_dim = 10;
    config.hidden_dim = 16;
    config.timesteps = 100;
    
    DiffusionModel diffusion(config);
    
    Tensor xt, epsilon;
    diffusion.forward_diffusion(test_data_, 50, xt, epsilon);
    
    EXPECT_EQ(xt.size(), 10);
    EXPECT_EQ(epsilon.size(), 10);
}

TEST_F(GenerativeModelsTest, DiffusionNoisePrediction) {
    DiffusionModel::Config config;
    config.input_dim = 10;
    config.hidden_dim = 16;
    config.timesteps = 100;
    
    DiffusionModel diffusion(config);
    
    Tensor xt = test_data_;
    Tensor epsilon_pred;
    
    diffusion.predict_noise(xt, 50, {}, epsilon_pred);
    
    EXPECT_EQ(epsilon_pred.size(), 10);
}

TEST_F(GenerativeModelsTest, DiffusionTraining) {
    DiffusionModel::Config config;
    config.input_dim = 10;
    config.hidden_dim = 16;
    config.timesteps = 100;
    
    DiffusionModel diffusion(config);
    
    float loss = diffusion.train_step(test_data_, {});
    
    EXPECT_GT(loss, 0.0f);
    EXPECT_EQ(diffusion.get_loss(), loss);
}

TEST_F(GenerativeModelsTest, DiffusionSampling) {
    DiffusionModel::Config config;
    config.input_dim = 10;
    config.hidden_dim = 16;
    config.timesteps = 10;  // Small for testing
    
    DiffusionModel diffusion(config);
    
    Tensor output;
    diffusion.reverse_diffusion({}, output);
    
    EXPECT_EQ(output.size(), 10);
}

// =============================================================================
// FACTORY TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, FactoryCreateVAE) {
    auto vae = GenerativeModelFactory::create_lightweight_vae(10, 2);
    
    EXPECT_NE(vae, nullptr);
    
    Tensor output, mu, logvar;
    vae->forward(test_data_, output, mu, logvar);
    
    EXPECT_EQ(output.size(), 10);
    EXPECT_EQ(mu.size(), 2);
    EXPECT_EQ(logvar.size(), 2);
}

TEST_F(GenerativeModelsTest, FactoryCreateGAN) {
    auto gan = GenerativeModelFactory::create_lightweight_gan(2, 10);
    
    EXPECT_NE(gan, nullptr);
    
    Tensor output;
    Tensor noise = {0.5f, -0.5f};
    
    gan->generate(noise, {}, output);
    
    EXPECT_EQ(output.size(), 10);
}

TEST_F(GenerativeModelsTest, FactoryCreateDiffusion) {
    auto diffusion = GenerativeModelFactory::create_lightweight_diffusion(10);
    
    EXPECT_NE(diffusion, nullptr);
    
    float loss = diffusion->train_step(test_data_, {});
    
    EXPECT_GT(loss, 0.0f);
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, VAEPerformance) {
    VAE::Config config;
    config.input_dim = 100;
    config.latent_dim = 16;
    config.hidden_dim = 64;
    
    VAE vae(config);
    
    Tensor large_data(100, 0.5f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        vae.train_step(large_data);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete within reasonable time (adjust threshold as needed)
    EXPECT_LT(duration.count(), 1000);  // Less than 1 second for 100 training steps
}

TEST_F(GenerativeModelsTest, GANPerformance) {
    GAN::Config config;
    config.latent_dim = 16;
    config.input_dim = 100;
    config.hidden_dim = 64;
    
    GAN gan(config);
    
    std::vector<Tensor> real_data = {Tensor(100, 0.5f), Tensor(100, 0.5f)};
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 50; ++i) {
        gan.train_step(real_data, {});
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_LT(duration.count(), 1000);  // Less than 1 second for 50 training steps
}

TEST_F(GenerativeModelsTest, DiffusionPerformance) {
    DiffusionModel::Config config;
    config.input_dim = 50;
    config.hidden_dim = 32;
    config.timesteps = 50;  // Reduced for performance testing
    
    DiffusionModel diffusion(config);
    
    Tensor data(50, 0.5f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 20; ++i) {
        diffusion.train_step(data, {});
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_LT(duration.count(), 2000);  // Less than 2 seconds for 20 training steps
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, VAEConditionalGeneration) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 2;
    config.hidden_dim = 16;
    config.conditional = true;
    config.condition_dim = 3;
    
    VAE vae(config);
    
    Tensor output;
    vae.generate(condition_, output);
    
    EXPECT_EQ(output.size(), 10);
    
    // Test conditional training
    float loss = vae.train_step(test_data_, condition_);
    EXPECT_GT(loss, 0.0f);
}

TEST_F(GenerativeModelsTest, GANConditionalGeneration) {
    GAN::Config config;
    config.latent_dim = 2;
    config.input_dim = 10;
    config.hidden_dim = 16;
    config.conditional = true;
    config.condition_dim = 3;
    
    GAN gan(config);
    
    Tensor output;
    Tensor noise = {0.5f, -0.5f};
    
    gan.generate(noise, condition_, output);
    
    EXPECT_EQ(output.size(), 10);
    
    float score = gan.discriminate(test_data_, condition_);
    EXPECT_TRUE(std::isfinite(score));
}

TEST_F(GenerativeModelsTest, DiffusionConditionalGeneration) {
    DiffusionModel::Config config;
    config.input_dim = 10;
    config.hidden_dim = 16;
    config.timesteps = 20;
    config.conditional = true;
    config.condition_dim = 3;
    
    DiffusionModel diffusion(config);
    
    Tensor output;
    diffusion.reverse_diffusion(condition_, output);
    
    EXPECT_EQ(output.size(), 10);
    
    float loss = diffusion.train_step(test_data_, condition_);
    EXPECT_GT(loss, 0.0f);
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

TEST_F(GenerativeModelsTest, EmptyInputHandling) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 2;
    config.hidden_dim = 16;
    
    VAE vae(config);
    
    Tensor empty_input;
    Tensor output, mu, logvar;
    
    // Should handle empty input gracefully
    EXPECT_NO_THROW(vae.encode(empty_input, mu, logvar));
}

TEST_F(GenerativeModelsTest, SingleDimensionHandling) {
    VAE::Config config;
    config.input_dim = 1;
    config.latent_dim = 1;
    config.hidden_dim = 4;
    
    VAE vae(config);
    
    Tensor single_data = {0.5f};
    Tensor output, mu, logvar;
    
    vae.forward(single_data, output, mu, logvar);
    
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(mu.size(), 1);
    EXPECT_EQ(logvar.size(), 1);
}

TEST_F(GenerativeModelsTest, LargeLatentSpace) {
    VAE::Config config;
    config.input_dim = 10;
    config.latent_dim = 100;  // Larger latent space
    config.hidden_dim = 64;
    
    VAE vae(config);
    
    Tensor output, mu, logvar;
    vae.forward(test_data_, output, mu, logvar);
    
    EXPECT_EQ(output.size(), 10);
    EXPECT_EQ(mu.size(), 100);
    EXPECT_EQ(logvar.size(), 100);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
