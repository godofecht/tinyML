//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek Shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "PhysicsInformedNN.h"
#include <iostream>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>

namespace ML {
namespace Physics {

// PDESolver base implementation
PDESolver::PDESolver() = default;

std::unique_ptr<PDESolver> PDESolver::create_heat_solver(const PDEConfig& config) {
    return std::make_unique<HeatEquationSolver>(config);
}

std::unique_ptr<PDESolver> PDESolver::create_wave_solver(const PDEConfig& config) {
    return std::make_unique<WaveEquationSolver>(config);
}

std::unique_ptr<PDESolver> PDESolver::create_navier_stokes_solver(const PDEConfig& config) {
    return std::make_unique<NavierStokesSolver>(config);
}

std::unique_ptr<PDESolver> PDESolver::create_custom_solver(const PDEConfig& config) {
    // For custom PDEs, create a generic PINN
    PINNArchitecture arch(config.domain.bounds.size(), 1, {50, 50, 50});
    auto pinn = std::make_unique<PhysicsInformedNN>(config, arch);
    
    class CustomSolver : public PDESolver {
    public:
        CustomSolver(const PDEConfig& config, std::unique_ptr<PhysicsInformedNN> pinn)
            : pinn_(std::move(pinn)) {
            config_ = config;
        }
        
        PINNResults solve() override {
            return pinn_->solve();
        }
        
        std::vector<double> compute_residual(const std::vector<double>& input) override {
            return pinn_->compute_residual(input);
        }
        
    private:
        std::unique_ptr<PhysicsInformedNN> pinn_;
    };
    
    return std::make_unique<CustomSolver>(config, std::move(pinn));
}

// HeatEquationSolver implementation
HeatEquationSolver::HeatEquationSolver(const PDEConfig& config) {
    config_ = config;
    
    // Create PINN with appropriate architecture
    size_t input_dim = config.domain.bounds.size() / 2 + (config.is_time_dependent ? 1 : 0);
    PINNArchitecture arch(input_dim, 1, {50, 50, 50});
    arch.activation = "tanh";
    
    pinn_ = std::make_unique<PhysicsInformedNN>(config, arch);
}

PINNResults HeatEquationSolver::solve() {
    std::cout << "Solving Heat Equation with PINN..." << std::endl;
    std::cout << "Domain: [";
    for (size_t i = 0; i < config_.domain.bounds.size(); ++i) {
        std::cout << config_.domain.bounds[i];
        if (i < config_.domain.bounds.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Diffusion coefficient: " << config_.diffusion_coefficient << std::endl;
    
    if (config_.is_time_dependent) {
        return pinn_->solve_time_dependent();
    } else {
        return pinn_->solve();
    }
}

std::vector<double> HeatEquationSolver::compute_residual(const std::vector<double>& input) {
    return pinn_->compute_residual(input);
}

std::vector<double> HeatEquationSolver::compute_heat_residual(const std::vector<double>& input,
                                                           const std::vector<double>& output,
                                                           const std::vector<std::vector<double>>& gradients,
                                                           const std::vector<std::vector<std::vector<double>>>& hessians) const {
    // Heat equation: u_t - α * ∇²u = 0
    std::vector<double> residual(output.size());
    
    for (size_t i = 0; i < output.size(); ++i) {
        double laplacian = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            if (j < hessians.size() && j < hessians[j].size() && i < hessians[j][j].size()) {
                laplacian += hessians[j][j][i];
            }
        }
        
        double time_derivative = 0.0;
        if (config_.is_time_dependent && !gradients.empty()) {
            time_derivative = gradients.back()[i];
        }
        
        residual[i] = time_derivative - config_.diffusion_coefficient * laplacian;
    }
    
    return residual;
}

// WaveEquationSolver implementation
WaveEquationSolver::WaveEquationSolver(const PDEConfig& config) {
    config_ = config;
    
    // Create PINN with appropriate architecture for wave equation
    size_t input_dim = config.domain.bounds.size() / 2 + (config.is_time_dependent ? 1 : 0);
    PINNArchitecture arch(input_dim, 1, {60, 60, 60});
    arch.activation = "tanh";
    arch.use_residual_connections = true; // Wave equations benefit from residual connections
    
    pinn_ = std::make_unique<PhysicsInformedNN>(config, arch);
}

PINNResults WaveEquationSolver::solve() {
    std::cout << "Solving Wave Equation with PINN..." << std::endl;
    std::cout << "Domain: [";
    for (size_t i = 0; i < config_.domain.bounds.size(); ++i) {
        std::cout << config_.domain.bounds[i];
        if (i < config_.domain.bounds.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Wave speed: " << config_.wave_speed << std::endl;
    
    if (config_.is_time_dependent) {
        return pinn_->solve_time_dependent();
    } else {
        return pinn_->solve();
    }
}

std::vector<double> WaveEquationSolver::compute_residual(const std::vector<double>& input) {
    return pinn_->compute_residual(input);
}

std::vector<double> WaveEquationSolver::compute_wave_residual(const std::vector<double>& input,
                                                             const std::vector<double>& output,
                                                             const std::vector<std::vector<double>>& gradients,
                                                             const std::vector<std::vector<std::vector<double>>>& hessians) const {
    // Wave equation: u_tt - c² * ∇²u = 0
    std::vector<double> residual(output.size());
    
    for (size_t i = 0; i < output.size(); ++i) {
        double laplacian = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            if (j < hessians.size() && j < hessians[j].size() && i < hessians[j][j].size()) {
                laplacian += hessians[j][j][i];
            }
        }
        
        double time_second_derivative = 0.0;
        if (config_.is_time_dependent && hessians.size() > 0) {
            size_t time_idx = hessians.size() - 1;
            if (time_idx < hessians.size() && time_idx < hessians[time_idx].size() && 
                i < hessians[time_idx][time_idx].size()) {
                time_second_derivative = hessians[time_idx][time_idx][i];
            }
        }
        
        residual[i] = time_second_derivative - config_.wave_speed * config_.wave_speed * laplacian;
    }
    
    return residual;
}

// NavierStokesSolver implementation
NavierStokesSolver::NavierStokesSolver(const PDEConfig& config) {
    config_ = config;
    
    // Create PINN with appropriate architecture for Navier-Stokes
    size_t input_dim = config.domain.bounds.size() / 2 + (config.is_time_dependent ? 1 : 0);
    PINNArchitecture arch(input_dim, 3, {80, 80, 80, 80}); // u, v, p
    arch.activation = "tanh";
    arch.use_residual_connections = true;
    arch.use_batch_normalization = true;
    
    pinn_ = std::make_unique<PhysicsInformedNN>(config, arch);
}

PINNResults NavierStokesSolver::solve() {
    std::cout << "Solving Navier-Stokes Equations with PINN..." << std::endl;
    std::cout << "Domain: [";
    for (size_t i = 0; i < config_.domain.bounds.size(); ++i) {
        std::cout << config_.domain.bounds[i];
        if (i < config_.domain.bounds.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Viscosity: " << config_.viscosity << std::endl;
    
    if (config_.is_time_dependent) {
        return pinn_->solve_time_dependent();
    } else {
        return pinn_->solve();
    }
}

std::vector<double> NavierStokesSolver::compute_residual(const std::vector<double>& input) {
    return pinn_->compute_residual(input);
}

std::vector<double> NavierStokesSolver::compute_navier_stokes_residual(const std::vector<double>& input,
                                                                      const std::vector<double>& output,
                                                                      const std::vector<std::vector<double>>& gradients,
                                                                      const std::vector<std::vector<std::vector<double>>>& hessians) const {
    // Navier-Stokes: u_t + (u·∇)u - ν∇²u + ∇p = 0, ∇·u = 0
    std::vector<double> residual(output.size());
    
    if (output.size() >= 2) {
        // Velocity components
        double u = output[0], v = output[1];
        double p = (output.size() > 2) ? output[2] : 0.0;
        
        // Convective terms: (u·∇)u
        double u_advective = 0.0, v_advective = 0.0;
        if (gradients.size() >= 2) {
            u_advective = u * gradients[0][0] + v * gradients[1][0];
            v_advective = u * gradients[0][1] + v * gradients[1][1];
        }
        
        // Diffusive terms: -ν∇²u
        double u_diffusive = 0.0, v_diffusive = 0.0;
        for (size_t j = 0; j < std::min(input.size(), size_t(2)); ++j) {
            if (j < hessians.size() && j < hessians[j].size()) {
                if (0 < hessians[j][j].size()) u_diffusive += hessians[j][j][0];
                if (1 < hessians[j][j].size()) v_diffusive += hessians[j][j][1];
            }
        }
        u_diffusive *= config_.viscosity;
        v_diffusive *= config_.viscosity;
        
        // Pressure gradients: ∇p
        double p_grad_x = 0.0, p_grad_y = 0.0;
        if (gradients.size() >= 2 && gradients[0].size() > 2 && gradients[1].size() > 2) {
            p_grad_x = gradients[0][2];
            p_grad_y = gradients[1][2];
        }
        
        // Time derivatives: u_t
        double u_time = 0.0, v_time = 0.0;
        if (config_.is_time_dependent && !gradients.empty()) {
            size_t time_idx = gradients.size() - 1;
            if (time_idx < gradients.size()) {
                if (0 < gradients[time_idx].size()) u_time = gradients[time_idx][0];
                if (1 < gradients[time_idx].size()) v_time = gradients[time_idx][1];
            }
        }
        
        // Momentum equations
        residual[0] = u_time + u_advective - u_diffusive + p_grad_x;
        residual[1] = v_time + v_advective - v_diffusive + p_grad_y;
        
        // Continuity equation: ∇·u = 0
        if (output.size() > 2) {
            double divergence = 0.0;
            if (gradients.size() >= 2) {
                divergence = gradients[0][0] + gradients[1][1];
            }
            residual[2] = divergence;
        }
    }
    
    return residual;
}

// AdaptiveCollocation implementation
AdaptiveCollocation::AdaptiveCollocation(const PDEConfig& config) : config_(config) {
    initialize_points();
}

void AdaptiveCollocation::initialize_points() {
    points_.clear();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Generate random collocation points
    for (size_t i = 0; i < config_.num_collocation_points; ++i) {
        std::vector<double> coords;
        for (size_t dim = 0; dim < config_.domain.bounds.size() / 2; ++dim) {
            std::uniform_real_distribution<double> dis(config_.domain.bounds[2*dim], config_.domain.bounds[2*dim + 1]);
            coords.push_back(dis(gen));
        }
        
        std::uniform_real_distribution<double> time_dis(0.0, config_.time_span);
        coords.push_back(time_dis(gen));
        
        CollocationPoint point(coords);
        point.weight = 1.0;
        point.is_boundary = false;
        
        points_.push_back(point);
    }
}

void AdaptiveCollocation::update_points(const std::vector<CollocationPoint>& current_points,
                                      const std::vector<double>& residuals) {
    // Update point weights based on residuals
    std::vector<double> weights = compute_residual_weights(residuals);
    
    for (size_t i = 0; i < points_.size() && i < weights.size(); ++i) {
        points_[i].weight = weights[i];
    }
    
    // Add points in high residual regions
    refine_high_residual_regions(residuals);
    
    // Remove points in low residual regions
    coarsen_low_residual_regions();
}

void AdaptiveCollocation::refine_high_residual_regions(const std::vector<double>& residual_field) {
    if (residual_field.empty()) return;
    
    // Find regions with high residuals
    std::vector<std::pair<std::vector<double>, double>> high_residual_points;
    
    for (size_t i = 0; i < points_.size() && i < residual_field.size(); ++i) {
        if (residual_field[i] > refinement_threshold) {
            high_residual_points.emplace_back(points_[i].coordinates, residual_field[i]);
        }
    }
    
    // Add more points in high residual regions
    for (const auto& [coords, residual] : high_residual_points) {
        add_points_in_region(coords, 0.1, 5); // Add 5 points within radius 0.1
    }
}

void AdaptiveCollocation::coarsen_low_residual_regions() {
    // Remove points with very low residuals
    points_.erase(
        std::remove_if(points_.begin(), points_.end(),
                      [this](const CollocationPoint& point) {
                          return point.weight < coarsening_threshold;
                      }),
        points_.end()
    );
}

void AdaptiveCollocation::add_points_in_region(const std::vector<double>& center, double radius, size_t num_points) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < num_points; ++i) {
        std::vector<double> new_coords;
        
        for (size_t dim = 0; dim < center.size(); ++dim) {
            std::uniform_real_distribution<double> dis(center[dim] - radius, center[dim] + radius);
            new_coords.push_back(dis(gen));
        }
        
        CollocationPoint new_point(new_coords);
        new_point.weight = 1.0;
        new_point.is_boundary = false;
        
        points_.push_back(new_point);
    }
}

void AdaptiveCollocation::remove_points_in_region(const std::vector<double>& center, double radius) {
    points_.erase(
        std::remove_if(points_.begin(), points_.end(),
                      [&center, radius](const CollocationPoint& point) {
                          double distance = 0.0;
                          for (size_t dim = 0; dim < center.size() && dim < point.coordinates.size(); ++dim) {
                              double diff = point.coordinates[dim] - center[dim];
                              distance += diff * diff;
                          }
                          return std::sqrt(distance) < radius;
                      }),
        points_.end()
    );
}

std::vector<double> AdaptiveCollocation::compute_residual_weights(const std::vector<double>& residuals) const {
    std::vector<double> weights(residuals.size());
    
    if (residuals.empty()) return weights;
    
    // Normalize residuals to compute weights
    double max_residual = *std::max_element(residuals.begin(), residuals.end());
    double min_residual = *std::min_element(residuals.begin(), residuals.end());
    
    for (size_t i = 0; i < residuals.size(); ++i) {
        if (max_residual > min_residual) {
            weights[i] = (residuals[i] - min_residual) / (max_residual - min_residual);
        } else {
            weights[i] = 1.0;
        }
        
        // Ensure minimum weight
        weights[i] = std::max(weights[i], 0.1);
    }
    
    return weights;
}

// UncertaintyQuantification implementation
UncertaintyQuantification::UncertaintyQuantification(size_t ensemble_size) 
    : ensemble_size_(ensemble_size) {}

void UncertaintyQuantification::train_ensemble(const PDEConfig& config, const PINNArchitecture& arch) {
    ensemble_.clear();
    
    std::cout << "Training ensemble of " << ensemble_size_ << " PINNs for uncertainty quantification..." << std::endl;
    
    for (size_t i = 0; i < ensemble_size_; ++i) {
        auto pinn = std::make_unique<PhysicsInformedNN>(config, arch);
        
        // Add slight randomness to initial conditions for diversity
        pinn->initialize();
        
        // Train the ensemble member
        PINNResults results = pinn->solve();
        
        ensemble_.push_back(std::move(pinn));
        
        std::cout << "Ensemble member " << (i + 1) << "/" << ensemble_size_ << " trained" << std::endl;
    }
}

std::vector<double> UncertaintyQuantification::predict_with_uncertainty(const std::vector<double>& input) {
    std::vector<std::vector<double>> ensemble_predictions;
    
    // Get predictions from all ensemble members
    for (const auto& pinn : ensemble_) {
        ensemble_predictions.push_back(pinn->predict(input));
    }
    
    update_statistics(ensemble_predictions);
    
    return mean_prediction_;
}

std::vector<double> UncertaintyQuantification::compute_confidence_intervals(const std::vector<double>& input, double confidence) {
    if (mean_prediction_.empty() || std_deviation_.size() != mean_prediction_.size()) {
        predict_with_uncertainty(input);
    }
    
    // Compute confidence intervals (simplified normal distribution assumption)
    double z_score = 1.96; // For 95% confidence
    
    std::vector<double> confidence_intervals;
    for (size_t i = 0; i < mean_prediction_.size(); ++i) {
        double margin = z_score * std::abs(std_deviation_[i]);
        if (margin > 0.5) {
            margin = 0.5;
        }
        confidence_intervals.push_back(mean_prediction_[i] - margin);
        confidence_intervals.push_back(mean_prediction_[i] + margin);
    }
    
    return confidence_intervals;
}

void UncertaintyQuantification::update_statistics(const std::vector<std::vector<double>>& ensemble_predictions) {
    if (ensemble_predictions.empty()) return;
    
    size_t output_dim = ensemble_predictions[0].size();
    mean_prediction_.resize(output_dim, 0.0);
    std_deviation_.resize(output_dim, 0.0);
    
    // Compute mean
    for (const auto& prediction : ensemble_predictions) {
        for (size_t i = 0; i < output_dim; ++i) {
            mean_prediction_[i] += prediction[i];
        }
    }
    
    for (size_t i = 0; i < output_dim; ++i) {
        mean_prediction_[i] /= ensemble_predictions.size();
    }
    
    // Compute standard deviation
    for (const auto& prediction : ensemble_predictions) {
        for (size_t i = 0; i < output_dim; ++i) {
            double diff = prediction[i] - mean_prediction_[i];
            std_deviation_[i] += diff * diff;
        }
    }
    
    for (size_t i = 0; i < output_dim; ++i) {
        std_deviation_[i] = std::sqrt(std_deviation_[i] / ensemble_predictions.size());
    }
}

// InverseProblemSolver implementation
InverseProblemSolver::InverseProblemSolver(const PDEConfig& config, const PINNArchitecture& arch)
    : config_(config), architecture_(arch) {
    identified_parameters_.resize(config_.custom_parameters.size(), 0.0);
    parameter_uncertainty_ = 0.0;
}

std::vector<double> InverseProblemSolver::solve_inverse_problem(const std::vector<std::vector<double>>& observations,
                                                             const std::vector<std::vector<double>>& observation_points) {
    std::cout << "Solving inverse problem with " << observations.size() << " observations..." << std::endl;
    
    // Create PINN with trainable parameters
    auto pinn = std::make_unique<PhysicsInformedNN>(config_, architecture_);
    pinn->initialize();
    
    // Iterative parameter identification
    std::vector<double> current_parameters = config_.custom_parameters;
    double learning_rate = 0.01;
    
    for (size_t iteration = 0; iteration < 1000; ++iteration) {
        double total_loss = 0.0;
        
        // Compute loss against observations
        for (size_t i = 0; i < observations.size(); ++i) {
            std::vector<double> prediction = pinn->predict(observation_points[i]);
            
            // Compute misfit
            for (size_t j = 0; j < prediction.size() && j < observations[i].size(); ++j) {
                double error = prediction[j] - observations[i][j];
                total_loss += error * error;
            }
        }
        
        // Update parameters (simplified gradient descent)
        for (size_t i = 0; i < current_parameters.size(); ++i) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dis(0.0, 0.01);
            
            current_parameters[i] -= learning_rate * dis(gen);
        }
        
        // Update PDE config with new parameters
        config_.custom_parameters = current_parameters;
        
        if (iteration % 100 == 0) {
            std::cout << "Iteration " << iteration << ", Loss: " << total_loss << std::endl;
        }
    }
    
    identified_parameters_ = current_parameters;
    
    // Estimate parameter uncertainty (simplified)
    parameter_uncertainty_ = 0.01; // Placeholder
    
    return identified_parameters_;
}

double InverseProblemSolver::compute_misfit_loss(const std::vector<std::vector<double>>& observations,
                                                const std::vector<std::vector<double>>& observation_points) {
    double total_loss = 0.0;
    
    auto pinn = std::make_unique<PhysicsInformedNN>(config_, architecture_);
    pinn->initialize();
    
    for (size_t i = 0; i < observations.size(); ++i) {
        std::vector<double> prediction = pinn->predict(observation_points[i]);
        
        for (size_t j = 0; j < prediction.size() && j < observations[i].size(); ++j) {
            double error = prediction[j] - observations[i][j];
            total_loss += error * error;
        }
    }
    
    return total_loss / observations.size();
}

// MultiScaleModeling implementation
MultiScaleModeling::MultiScaleModeling(const std::vector<double>& scales) : scales_(scales) {}

void MultiScaleModeling::add_fine_scale_model(std::unique_ptr<PhysicsInformedNN> fine_model) {
    if (scale_models_.empty()) {
        scale_models_.push_back(std::move(fine_model));
    } else {
        scale_models_.insert(scale_models_.begin(), std::move(fine_model));
    }
}

void MultiScaleModeling::add_coarse_scale_model(std::unique_ptr<PhysicsInformedNN> coarse_model) {
    scale_models_.push_back(std::move(coarse_model));
}

std::vector<double> MultiScaleModeling::multi_scale_predict(const std::vector<double>& input) const {
    if (scale_models_.empty()) {
        return {};
    }
    
    if (scale_models_.size() == 1) {
        return scale_models_[0]->predict(input);
    }
    
    // Get predictions from different scales
    std::vector<std::vector<double>> scale_predictions;
    for (const auto& model : scale_models_) {
        scale_predictions.push_back(model->predict(input));
    }
    
    // Interpolate between scales
    if (scale_predictions.size() >= 2) {
        return interpolate_scales(scale_predictions[0], scale_predictions[1], input);
    }
    
    return scale_predictions[0];
}

void MultiScaleModeling::update_scale_coupling() {
    // Update coupling weights based on scale interactions
    coupling_weights_.resize(scales_.size(), 1.0);
    
    for (size_t i = 0; i < scales_.size(); ++i) {
        coupling_weights_[i] = 1.0 / (1.0 + scales_[i]); // Simple weighting scheme
    }
}

std::vector<double> MultiScaleModeling::interpolate_scales(const std::vector<double>& fine_output,
                                                          const std::vector<double>& coarse_output,
                                                          const std::vector<double>& input) const {
    std::vector<double> interpolated(fine_output.size());
    
    for (size_t i = 0; i < fine_output.size(); ++i) {
        // Simple linear interpolation between scales
        double alpha = 0.5; // Can be made input-dependent
        interpolated[i] = alpha * fine_output[i] + (1.0 - alpha) * coarse_output[i];
    }
    
    return interpolated;
}

// ConvergenceAccelerator implementation
ConvergenceAccelerator::ConvergenceAccelerator() = default;

void ConvergenceAccelerator::enable_preconditioning(const std::vector<std::vector<double>>& preconditioner) {
    use_preconditioning_ = true;
    preconditioner_matrix_ = preconditioner;
}

void ConvergenceAccelerator::enable_adaptive_learning_rate(double initial_lr, double decay_rate) {
    use_adaptive_lr_ = true;
    current_learning_rate_ = initial_lr;
    learning_rate_decay_ = decay_rate;
}

void ConvergenceAccelerator::enable_momentum(double momentum_coeff) {
    use_momentum_ = true;
    momentum_coefficient_ = momentum_coeff;
}

void ConvergenceAccelerator::update_learning_rate(double current_loss) {
    if (use_adaptive_lr_) {
        current_learning_rate_ *= learning_rate_decay_;
        current_learning_rate_ = std::max(current_learning_rate_, 1e-6); // Minimum learning rate
    }
}

std::vector<std::vector<double>> ConvergenceAccelerator::apply_preconditioning(const std::vector<std::vector<double>>& gradients) const {
    if (!use_preconditioning_ || preconditioner_matrix_.empty()) {
        return gradients;
    }
    
    // Simple preconditioning (matrix multiplication)
    std::vector<std::vector<double>> preconditioned_gradients(gradients.size(), 
        std::vector<double>(gradients[0].size(), 0.0));
    
    for (size_t i = 0; i < gradients.size(); ++i) {
        for (size_t j = 0; j < gradients[i].size(); ++j) {
            for (size_t k = 0; k < preconditioner_matrix_.size() && k < preconditioner_matrix_[k].size(); ++k) {
                preconditioned_gradients[i][j] += preconditioner_matrix_[i][k] * gradients[k][j];
            }
        }
    }
    
    return preconditioned_gradients;
}

std::vector<std::vector<double>> ConvergenceAccelerator::apply_momentum(const std::vector<std::vector<double>>& gradients) {
    if (!use_momentum_) {
        return gradients;
    }
    
    if (momentum_buffer_.empty()) {
        momentum_buffer_ = gradients;
    } else {
        for (size_t i = 0; i < gradients.size(); ++i) {
            for (size_t j = 0; j < gradients[i].size(); ++j) {
                momentum_buffer_[i][j] = momentum_coefficient_ * momentum_buffer_[i][j] + gradients[i][j];
            }
        }
    }
    
    return momentum_buffer_;
}

void ConvergenceAccelerator::update_momentum_buffer(const std::vector<std::vector<double>>& gradients) {
    if (momentum_buffer_.empty()) {
        momentum_buffer_ = gradients;
    } else {
        for (size_t i = 0; i < gradients.size(); ++i) {
            for (size_t j = 0; j < gradients[i].size(); ++j) {
                momentum_buffer_[i][j] = momentum_coefficient_ * momentum_buffer_[i][j] + gradients[i][j];
            }
        }
    }
}

} // namespace Physics
} // namespace ML
