//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "PhysicsInformedNN.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace ML {
namespace Physics {

// PhysicsInformedNN implementation
PhysicsInformedNN::PhysicsInformedNN(const PDEConfig& config, const PINNArchitecture& arch)
    : config_(config), architecture_(arch) {
    initialize_network();
    generate_collocation_points();
    generate_boundary_points();
}

bool PhysicsInformedNN::initialize() {
    // Initialize weights with Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        size_t fan_in = (layer == 0) ? architecture_.input_dim : architecture_.hidden_layers[layer - 1];
        size_t fan_out = (layer < architecture_.hidden_layers.size()) ? architecture_.hidden_layers[layer] : architecture_.output_dim;
        
        double limit = std::sqrt(6.0 / (fan_in + fan_out));
        std::uniform_real_distribution<double> dis(-limit, limit);
        
        for (size_t i = 0; i < weights_[layer].size(); ++i) {
            for (size_t j = 0; j < weights_[layer][i].size(); ++j) {
                weights_[layer][i][j] = dis(gen);
            }
        }
        
        // Initialize biases to zero
        std::fill(biases_[layer].begin(), biases_[layer].end(), 0.0);
    }
    
    return true;
}

PINNResults PhysicsInformedNN::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    results_.loss_history.clear();
    results_.residual_norms.clear();
    results_.converged = false;
    
    double prev_loss = std::numeric_limits<double>::infinity();
    
    for (size_t epoch = 0; epoch < config_.max_epochs; ++epoch) {
        // Combine collocation and boundary points
        std::vector<CollocationPoint> all_points = collocation_points_;
        all_points.insert(all_points.end(), boundary_points_.begin(), boundary_points_.end());
        
        // Shuffle points for stochastic training
        std::shuffle(all_points.begin(), all_points.end(), std::mt19937(std::random_device{}()));
        
        // Compute total loss
        double total_loss = compute_loss(all_points);
        results_.loss_history.push_back(total_loss);
        
        // Check convergence
        if (epoch > 0 && std::abs(prev_loss - total_loss) < config_.tolerance &&
            total_loss < config_.tolerance) {
            results_.converged = true;
            results_.epochs_trained = epoch;
            break;
        }
        
        prev_loss = total_loss;
        
        // Update weights using gradient descent
        update_weights(config_.learning_rate);
        
        // Adaptive collocation point update every 100 epochs
        if (epoch % 100 == 0 && epoch > 0) {
            update_collocation_points();
        }
        
        // Progress reporting
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results_.training_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    results_.final_loss = results_.loss_history.back();

    results_.solution.clear();
    results_.solution.reserve(collocation_points_.size());
    for (const auto& point : collocation_points_) {
        results_.solution.push_back(forward_pass(point.coordinates));
    }

    if (!results_.converged) {
        results_.epochs_trained = config_.max_epochs;
    }
    
    // Compute convergence rate
    if (results_.loss_history.size() > 10 && results_.epochs_trained > 0) {
        double initial_loss = results_.loss_history[0];
        double final_loss = results_.loss_history.back();
        if (initial_loss > 0.0 && final_loss > 0.0) {
            results_.convergence_rate = -std::log(final_loss / initial_loss) / results_.epochs_trained;
        }
    }
    
    return results_;
}

PINNResults PhysicsInformedNN::solve_time_dependent() {
    // For time-dependent problems, solve at each time step
    PINNResults results;
    
    PINNResults final_results;
    
    for (size_t time_step = 0; time_step < config_.num_time_steps; ++time_step) {
        double current_time = time_step * config_.time_span / config_.num_time_steps;
        
        // Update time in collocation points
        for (auto& point : collocation_points_) {
            if (point.coordinates.size() > architecture_.input_dim - 1) {
                point.coordinates.back() = current_time;
            }
        }
        
        PINNResults time_results = solve();
        
        if (time_step == 0) {
            final_results = time_results;
        } else {
            final_results.solution.insert(final_results.solution.end(), time_results.solution.begin(), time_results.solution.end());
        }
    }
    
    return final_results;
}

std::vector<double> PhysicsInformedNN::predict(const std::vector<double>& input) const {
    return forward_pass(input);
}

std::vector<double> PhysicsInformedNN::compute_residual(const std::vector<double>& input) const {
    std::vector<double> output = forward_pass(input);
    return compute_physics_residual(input, output);
}

double PhysicsInformedNN::compute_loss(const std::vector<CollocationPoint>& points) const {
    if (points.empty()) {
        return 0.0;
    }
    double total_loss = 0.0;
    double residual_norm = 0.0;
    
    for (const auto& point : points) {
        std::vector<double> output = forward_pass(point.coordinates);
        std::vector<double> residual = compute_physics_residual(point.coordinates, output);
        
        if (point.is_boundary) {
            // Apply boundary conditions
            residual = apply_boundary_conditions(point.coordinates, output);
        }
        
        // Weighted residual
        double point_loss = 0.0;
        for (double r : residual) {
            point_loss += r * r;
        }
        point_loss *= point.weight;
        
        total_loss += point_loss;
        residual_norm += std::sqrt(point_loss);
    }
    
    results_.residual_norms.push_back(residual_norm / points.size());
    return total_loss / points.size();
}

void PhysicsInformedNN::update_collocation_points() {
    AdaptiveCollocation adaptive(config_);
    adaptive.update_points(collocation_points_, results_.residual_norms);
    collocation_points_ = adaptive.get_collocation_points();
}

void PhysicsInformedNN::refine_mesh_high_residual() {
    if (!results_.residual_norms.empty()) {
        AdaptiveCollocation adaptive(config_);
        adaptive.refine_high_residual_regions(results_.residual_norms);
        collocation_points_ = adaptive.get_collocation_points();
    }
}

void PhysicsInformedNN::enable_uncertainty_quantification() {
    uncertainty_quantifier_ = std::make_unique<UncertaintyQuantification>(10);
    uncertainty_quantifier_->train_ensemble(config_, architecture_);
}

std::vector<double> PhysicsInformedNN::predict_with_uncertainty(const std::vector<double>& input) const {
    if (uncertainty_quantifier_) {
        return uncertainty_quantifier_->predict_with_uncertainty(input);
    }
    return predict(input);
}

void PhysicsInformedNN::enable_multi_scale_modeling(const std::vector<double>& scales) {
    multi_scale_scales_ = scales;
}

std::vector<double> PhysicsInformedNN::multi_scale_predict(const std::vector<double>& input) const {
    if (multi_scale_scales_.empty()) {
        return predict(input);
    }
    
    // Multi-scale prediction (simplified implementation)
    std::vector<double> prediction = predict(input);
    
    // Apply scale-dependent modifications
    for (size_t i = 0; i < prediction.size(); ++i) {
        for (double scale : multi_scale_scales_) {
            prediction[i] *= (1.0 + 0.1 * std::sin(scale * input[i % input.size()]));
        }
    }
    
    return prediction;
}

void PhysicsInformedNN::enable_preconditioning() {
    use_preconditioning_ = true;
}

void PhysicsInformedNN::enable_adaptive_learning_rate() {
    use_adaptive_lr_ = true;
}

double PhysicsInformedNN::get_training_time() const {
    return results_.training_time_ms;
}

double PhysicsInformedNN::get_convergence_rate() const {
    return results_.convergence_rate;
}

bool PhysicsInformedNN::has_converged() const {
    return results_.converged;
}

// Private methods
void PhysicsInformedNN::initialize_network() {
    size_t num_layers = architecture_.hidden_layers.size() + 1;
    
    weights_.resize(num_layers);
    biases_.resize(num_layers);
    
    // Initialize weights for each layer
    for (size_t i = 0; i < architecture_.hidden_layers.size(); ++i) {
        size_t input_size = (i == 0) ? architecture_.input_dim : architecture_.hidden_layers[i - 1];
        size_t output_size = architecture_.hidden_layers[i];
        
        weights_[i].resize(output_size, std::vector<double>(input_size, 0.0));
        biases_[i].resize(output_size, 0.0);
    }
    
    // Output layer
    size_t last_hidden_size = architecture_.hidden_layers.back();
    weights_.back().resize(architecture_.output_dim, std::vector<double>(last_hidden_size, 0.0));
    biases_.back().resize(architecture_.output_dim, 0.0);
}

void PhysicsInformedNN::generate_collocation_points() {
    collocation_points_.clear();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Generate random points in the domain
    for (size_t i = 0; i < config_.num_collocation_points; ++i) {
        std::vector<double> coords;
        
        for (size_t dim = 0; dim < config_.domain.bounds.size() / 2; ++dim) {
            std::uniform_real_distribution<double> dis(config_.domain.bounds[2*dim], config_.domain.bounds[2*dim + 1]);
            coords.push_back(dis(gen));
        }
        
        // Add time dimension if time-dependent
        if (config_.is_time_dependent) {
            std::uniform_real_distribution<double> time_dis(0.0, config_.time_span);
            coords.push_back(time_dis(gen));
        }
        
        CollocationPoint point(coords);
        point.weight = 1.0;
        point.is_boundary = false;
        
        collocation_points_.push_back(point);
    }
}

void PhysicsInformedNN::generate_boundary_points() {
    boundary_points_.clear();
    
    std::random_device rd;
    std::mt19937 gen(rd());

    const bool has_boundary_conditions = !config_.boundary_conditions.empty();
    
    // Generate points on domain boundaries
    for (size_t i = 0; i < config_.num_boundary_points; ++i) {
        std::vector<double> coords;
        
        // Place points on boundaries (simplified for rectangular domains)
        size_t dim = config_.domain.bounds.size() / 2;
        
        for (size_t d = 0; d < dim; ++d) {
            std::uniform_real_distribution<double> dis(config_.domain.bounds[2*d], config_.domain.bounds[2*d + 1]);
            
            // Randomly choose which boundary to place point on
            if (i % 2 == 0) {
                coords.push_back(config_.domain.bounds[2*d]); // Lower boundary
            } else {
                coords.push_back(config_.domain.bounds[2*d + 1]); // Upper boundary
            }
        }
        
        // Add time dimension if time-dependent
        if (config_.is_time_dependent) {
            std::uniform_real_distribution<double> time_dis(0.0, config_.time_span);
            coords.push_back(time_dis(gen));
        }
        
        CollocationPoint point(coords);
        point.weight = 10.0; // Higher weight for boundary conditions
        point.is_boundary = true;
        point.bc_type = has_boundary_conditions
            ? config_.boundary_conditions[i % config_.boundary_conditions.size()]
            : BoundaryConditionType::DIRICHLET;
        
        boundary_points_.push_back(point);
    }
}

void PhysicsInformedNN::train_network() {
    // Training logic is handled in solve() method
}

void PhysicsInformedNN::update_weights(double learning_rate) {
    // Simplified weight update (in practice, this would use automatic differentiation)
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        for (size_t i = 0; i < weights_[layer].size(); ++i) {
            for (size_t j = 0; j < weights_[layer][i].size(); ++j) {
                // Add small random perturbation (gradient descent approximation)
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<double> dis(0.0, 0.01);
                
                weights_[layer][i][j] -= learning_rate * dis(gen);
            }
            
            biases_[layer][i] -= learning_rate * 0.01; // Simplified bias update
        }
    }
}

std::vector<double> PhysicsInformedNN::compute_physics_residual(const std::vector<double>& input, 
                                                             const std::vector<double>& output) const {
    // Compute physics residual based on PDE type
    switch (config_.type) {
        case PDEType::HEAT_EQUATION:
            return compute_heat_residual(input, output);
        case PDEType::WAVE_EQUATION:
            return compute_wave_residual(input, output);
        case PDEType::NAVIER_STOKES:
            return compute_navier_stokes_residual(input, output);
        case PDEType::POISSON_EQUATION:
            return compute_poisson_residual(input, output);
        default:
            return std::vector<double>(output.size(), 0.0);
    }
}

std::vector<double> PhysicsInformedNN::apply_boundary_conditions(const std::vector<double>& input, 
                                                               const std::vector<double>& output) const {
    if (config_.boundary_conditions.empty()) {
        return output;
    }
    // Apply boundary conditions based on type
    switch (config_.boundary_conditions[0]) {
        case BoundaryConditionType::DIRICHLET:
            return Utils::apply_dirichlet_bc(input, 0.0);
        case BoundaryConditionType::NEUMANN:
            return Utils::apply_neumann_bc(input, 0.0);
        case BoundaryConditionType::PERIODIC:
            return Utils::apply_periodic_bc(input);
        default:
            return output;
    }
}

std::vector<double> PhysicsInformedNN::forward_pass(const std::vector<double>& input) const {
    std::vector<double> current = input;
    
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        std::vector<double> next(weights_[layer].size(), 0.0);
        
        // Matrix multiplication
        for (size_t i = 0; i < weights_[layer].size(); ++i) {
            for (size_t j = 0; j < current.size(); ++j) {
                next[i] += weights_[layer][i][j] * current[j];
            }
            next[i] += biases_[layer][i];
        }
        
        // Activation function (except for output layer)
        if (layer < weights_.size() - 1) {
            next = activate(next, architecture_.activation);
        }
        
        current = next;
    }
    
    return current;
}

std::vector<double> PhysicsInformedNN::activate(const std::vector<double>& x, const std::string& activation) const {
    std::vector<double> result(x.size());
    
    if (activation == "tanh") {
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::tanh(x[i]);
        }
    } else if (activation == "relu") {
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::max(0.0, x[i]);
        }
    } else if (activation == "sigmoid") {
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = 1.0 / (1.0 + std::exp(-x[i]));
        }
    } else {
        result = x; // Linear activation
    }
    
    return result;
}

std::vector<std::vector<double>> PhysicsInformedNN::compute_gradients(const std::vector<double>& input) const {
    // Simplified automatic differentiation (finite differences)
    std::vector<std::vector<double>> gradients(input.size(), std::vector<double>(architecture_.output_dim, 0.0));
    
    double epsilon = 1e-6;
    std::vector<double> base_output = forward_pass(input);
    
    for (size_t i = 0; i < input.size(); ++i) {
        std::vector<double> perturbed_input = input;
        perturbed_input[i] += epsilon;
        std::vector<double> perturbed_output = forward_pass(perturbed_input);
        
        for (size_t j = 0; j < architecture_.output_dim; ++j) {
            gradients[i][j] = (perturbed_output[j] - base_output[j]) / epsilon;
        }
    }
    
    return gradients;
}

std::vector<std::vector<std::vector<double>>> PhysicsInformedNN::compute_hessians(const std::vector<double>& input) const {
    // Simplified Hessian computation (second-order finite differences)
    std::vector<std::vector<std::vector<double>>> hessians(input.size(), 
        std::vector<std::vector<double>>(input.size(), std::vector<double>(architecture_.output_dim, 0.0)));
    
    double epsilon = 1e-5;
    
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            std::vector<double> input_pp = input;
            std::vector<double> input_pm = input;
            std::vector<double> input_mp = input;
            std::vector<double> input_mm = input;
            
            input_pp[i] += epsilon; input_pp[j] += epsilon;
            input_pm[i] += epsilon; input_pm[j] -= epsilon;
            input_mp[i] -= epsilon; input_mp[j] += epsilon;
            input_mm[i] -= epsilon; input_mm[j] -= epsilon;
            
            std::vector<double> output_pp = forward_pass(input_pp);
            std::vector<double> output_pm = forward_pass(input_pm);
            std::vector<double> output_mp = forward_pass(input_mp);
            std::vector<double> output_mm = forward_pass(input_mm);
            
            for (size_t k = 0; k < architecture_.output_dim; ++k) {
                hessians[i][j][k] = (output_pp[k] - output_pm[k] - output_mp[k] + output_mm[k]) / (4 * epsilon * epsilon);
            }
        }
    }
    
    return hessians;
}

// PDE-specific residual computations
std::vector<double> PhysicsInformedNN::compute_heat_residual(const std::vector<double>& input,
                                                            const std::vector<double>& output) const {
    // Heat equation: u_t - α * ∇²u = 0
    auto gradients = compute_gradients(input);
    auto hessians = compute_hessians(input);
    
    std::vector<double> residual(output.size());
    
    for (size_t i = 0; i < output.size(); ++i) {
        double laplacian = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            laplacian += hessians[j][j][i];
        }
        
        double time_derivative = (config_.is_time_dependent && !gradients.empty()) ? 
                                gradients.back()[i] : 0.0;
        
        residual[i] = time_derivative - config_.diffusion_coefficient * laplacian;
    }
    
    return residual;
}

std::vector<double> PhysicsInformedNN::compute_wave_residual(const std::vector<double>& input,
                                                           const std::vector<double>& output) const {
    // Wave equation: u_tt - c² * ∇²u = 0
    auto gradients = compute_gradients(input);
    auto hessians = compute_hessians(input);
    
    std::vector<double> residual(output.size());
    
    for (size_t i = 0; i < output.size(); ++i) {
        double laplacian = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            laplacian += hessians[j][j][i];
        }
        
        double time_second_derivative = 0.0;
        if (config_.is_time_dependent && hessians.size() > 0) {
            time_second_derivative = hessians[hessians.size()-1][i][i];
        }
        
        residual[i] = time_second_derivative - config_.wave_speed * config_.wave_speed * laplacian;
    }
    
    return residual;
}

std::vector<double> PhysicsInformedNN::compute_navier_stokes_residual(const std::vector<double>& input,
                                                                    const std::vector<double>& output) const {
    // Navier-Stokes: u_t + (u·∇)u - ν∇²u + ∇p = 0, ∇·u = 0
    auto gradients = compute_gradients(input);
    auto hessians = compute_hessians(input);
    
    std::vector<double> residual(output.size());
    
    // Simplified 2D implementation
    if (output.size() >= 2) {
        // Velocity components
        double u = output[0], v = output[1];
        
        // Convective terms
        double u_advective = u * gradients[0][0] + v * gradients[1][0];
        double v_advective = u * gradients[0][1] + v * gradients[1][1];
        
        // Diffusive terms
        double u_diffusive = 0.0, v_diffusive = 0.0;
        for (size_t j = 0; j < std::min(input.size(), size_t(2)); ++j) {
            u_diffusive += hessians[j][j][0];
            v_diffusive += hessians[j][j][1];
        }
        u_diffusive *= config_.viscosity;
        v_diffusive *= config_.viscosity;
        
        // Time derivatives
        double u_time = (config_.is_time_dependent && !gradients.empty()) ? gradients.back()[0] : 0.0;
        double v_time = (config_.is_time_dependent && !gradients.empty()) ? gradients.back()[1] : 0.0;
        
        residual[0] = u_time + u_advective - u_diffusive;
        residual[1] = v_time + v_advective - v_diffusive;
        
        // Continuity equation (∇·u = 0)
        if (output.size() > 2) {
            residual[2] = gradients[0][0] + gradients[1][1];
        }
    }
    
    return residual;
}

std::vector<double> PhysicsInformedNN::compute_poisson_residual(const std::vector<double>& input,
                                                              const std::vector<double>& output) const {
    // Poisson equation: -∇²u = f
    auto hessians = compute_hessians(input);
    
    std::vector<double> residual(output.size());
    
    for (size_t i = 0; i < output.size(); ++i) {
        double laplacian = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            laplacian += hessians[j][j][i];
        }
        
        // For simplicity, assume f = 0 (homogeneous equation)
        residual[i] = -laplacian;
    }
    
    return residual;
}

// Utility functions implementation
namespace Utils {

std::vector<double> apply_dirichlet_bc(const std::vector<double>& input, double value) {
    std::vector<double> result(input.size(), value);
    return result;
}

std::vector<double> apply_neumann_bc(const std::vector<double>& input, double derivative) {
    // Simplified Neumann BC implementation
    std::vector<double> result(input.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = derivative * input[i];
    }
    return result;
}

std::vector<double> apply_periodic_bc(const std::vector<double>& input) {
    // Periodic boundary conditions
    return input; // Simplified implementation
}

double compute_l2_error(const std::vector<double>& numerical, const std::vector<double>& analytical) {
    if (numerical.size() != analytical.size()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double error = 0.0;
    for (size_t i = 0; i < numerical.size(); ++i) {
        double diff = numerical[i] - analytical[i];
        error += diff * diff;
    }
    
    return std::sqrt(error / numerical.size());
}

double compute_linf_error(const std::vector<double>& numerical, const std::vector<double>& analytical) {
    if (numerical.size() != analytical.size()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double max_error = 0.0;
    for (size_t i = 0; i < numerical.size(); ++i) {
        double error = std::abs(numerical[i] - analytical[i]);
        max_error = std::max(max_error, error);
    }
    
    return max_error;
}

} // namespace Utils

} // namespace Physics
} // namespace ML
