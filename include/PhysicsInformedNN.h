//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek Shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef PHYSICS_INFORMED_NN_H
#define PHYSICS_INFORMED_NN_H

#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include "XSIMDOperations.h"

namespace ML {
namespace Physics {

// Forward declarations
class PhysicsInformedNN;
class PDESolver;
class AdaptiveCollocation;
class UncertaintyQuantification;

// PDE types and boundary conditions
enum class PDEType {
    HEAT_EQUATION,
    WAVE_EQUATION,
    NAVIER_STOKES,
    POISSON_EQUATION,
    BURGERS_EQUATION,
    SCHRÖDINGER_EQUATION,
    CUSTOM_PDE
};

enum class BoundaryConditionType {
    DIRICHLET,
    NEUMANN,
    ROBIN,
    PERIODIC,
    MIXED
};

// Domain and mesh structures
struct Domain {
    std::vector<double> bounds;  // [x_min, x_max, y_min, y_max, z_min, z_max]
    std::vector<size_t> dimensions;  // Number of points in each dimension
    bool is_periodic = false;
    
    Domain() = default;
    Domain(std::vector<double> b, std::vector<size_t> d) 
        : bounds(std::move(b)), dimensions(std::move(d)) {}
};

struct CollocationPoint {
    std::vector<double> coordinates;
    std::vector<double> physics_residuals;
    double weight = 1.0;
    bool is_boundary = false;
    BoundaryConditionType bc_type = BoundaryConditionType::DIRICHLET;
    
    CollocationPoint(std::vector<double> coords) : coordinates(std::move(coords)) {}
};

// PDE configuration
struct PDEConfig {
    PDEType type;
    Domain domain;
    std::vector<BoundaryConditionType> boundary_conditions;
    double time_span = 1.0;
    size_t num_time_steps = 100;
    bool is_time_dependent = false;
    bool is_nonlinear = false;
    
    // Physics parameters
    double diffusion_coefficient = 1.0;
    double wave_speed = 1.0;
    double viscosity = 0.01;
    std::vector<double> custom_parameters;
    
    // Training parameters
    size_t num_collocation_points = 1000;
    size_t num_boundary_points = 200;
    double learning_rate = 0.001;
    size_t max_epochs = 10000;
    double tolerance = 1e-6;
};

// Neural network architecture for PINNs
struct PINNArchitecture {
    size_t input_dim;
    size_t output_dim;
    std::vector<size_t> hidden_layers;
    std::string activation = "tanh";
    bool use_residual_connections = false;
    bool use_batch_normalization = false;
    
    PINNArchitecture(size_t in_dim, size_t out_dim, std::vector<size_t> hidden)
        : input_dim(in_dim), output_dim(out_dim), hidden_layers(std::move(hidden)) {}
};

// Training results and metrics
struct PINNResults {
    std::vector<std::vector<double>> solution;
    std::vector<double> loss_history;
    std::vector<double> residual_norms;
    double final_loss = 0.0;
    double convergence_rate = 0.0;
    size_t epochs_trained = 0;
    double training_time_ms = 0.0;
    bool converged = false;
    
    // Uncertainty quantification
    std::vector<double> mean_solution;
    std::vector<double> std_deviation;
    std::vector<double> confidence_intervals;
};

// Main Physics-Informed Neural Network class
class PhysicsInformedNN {
public:
    PhysicsInformedNN(const PDEConfig& config, const PINNArchitecture& arch);
    ~PhysicsInformedNN() = default;
    
    // Core PINN functionality
    bool initialize();
    PINNResults solve();
    PINNResults solve_time_dependent();
    
    // Prediction and evaluation
    std::vector<double> predict(const std::vector<double>& input) const;
    std::vector<double> compute_residual(const std::vector<double>& input) const;
    double compute_loss(const std::vector<CollocationPoint>& points) const;
    
    // Adaptive collocation
    void update_collocation_points();
    void refine_mesh_high_residual();
    
    // Uncertainty quantification
    void enable_uncertainty_quantification();
    std::vector<double> predict_with_uncertainty(const std::vector<double>& input) const;
    
    // Multi-scale modeling
    void enable_multi_scale_modeling(const std::vector<double>& scales);
    std::vector<double> multi_scale_predict(const std::vector<double>& input) const;
    
    // Convergence acceleration
    void enable_preconditioning();
    void enable_adaptive_learning_rate();
    
    // Getters and setters
    PINNResults get_results() const { return results_; }
    std::vector<CollocationPoint> get_collocation_points() const { return collocation_points_; }
    PDEConfig get_config() const { return config_; }
    
    // Performance monitoring
    double get_training_time() const;
    double get_convergence_rate() const;
    bool has_converged() const;

private:
    PDEConfig config_;
    PINNArchitecture architecture_;
    
    // Neural network weights and biases
    std::vector<std::vector<std::vector<double>>> weights_;
    std::vector<std::vector<double>> biases_;
    
    // Training data
    std::vector<CollocationPoint> collocation_points_;
    std::vector<CollocationPoint> boundary_points_;
    
    // Results and metrics
    mutable PINNResults results_;
    
    // Advanced features
    std::unique_ptr<UncertaintyQuantification> uncertainty_quantifier_;
    std::vector<double> multi_scale_scales_;
    bool use_preconditioning_ = false;
    bool use_adaptive_lr_ = false;
    
    // Internal methods
    void initialize_network();
    void generate_collocation_points();
    void generate_boundary_points();
    void train_network();
    void update_weights(double learning_rate);
    
    // PDE-specific physics computations
    std::vector<double> compute_physics_residual(const std::vector<double>& input, 
                                               const std::vector<double>& output) const;
    std::vector<double> apply_boundary_conditions(const std::vector<double>& input, 
                                                 const std::vector<double>& output) const;
    
    // PDE-specific residual computations
    std::vector<double> compute_heat_residual(const std::vector<double>& input, 
                                            const std::vector<double>& output) const;
    std::vector<double> compute_wave_residual(const std::vector<double>& input, 
                                            const std::vector<double>& output) const;
    std::vector<double> compute_navier_stokes_residual(const std::vector<double>& input, 
                                                     const std::vector<double>& output) const;
    std::vector<double> compute_poisson_residual(const std::vector<double>& input, 
                                               const std::vector<double>& output) const;
    
    // Boundary condition applications
    std::vector<double> apply_dirichlet_bc(const std::vector<double>& input, double value) const;
    std::vector<double> apply_neumann_bc(const std::vector<double>& input, double derivative) const;
    std::vector<double> apply_periodic_bc(const std::vector<double>& input) const;
    
    // Neural network forward pass
    std::vector<double> forward_pass(const std::vector<double>& input) const;
    std::vector<double> activate(const std::vector<double>& x, const std::string& activation) const;
    
    // Automatic differentiation for gradients
    std::vector<std::vector<double>> compute_gradients(const std::vector<double>& input) const;
    std::vector<std::vector<std::vector<double>>> compute_hessians(const std::vector<double>& input) const;
};

// Specialized PDE solver class
class PDESolver {
public:
    PDESolver();
    virtual ~PDESolver() = default;
    
    // Factory methods for different PDE types
    static std::unique_ptr<PDESolver> create_heat_solver(const PDEConfig& config);
    static std::unique_ptr<PDESolver> create_wave_solver(const PDEConfig& config);
    static std::unique_ptr<PDESolver> create_navier_stokes_solver(const PDEConfig& config);
    static std::unique_ptr<PDESolver> create_custom_solver(const PDEConfig& config);
    
    virtual PINNResults solve() = 0;
    virtual std::vector<double> compute_residual(const std::vector<double>& input) = 0;
    
protected:
    PDEConfig config_;
    std::unique_ptr<PhysicsInformedNN> pinn_;
};

// Heat equation solver
class HeatEquationSolver : public PDESolver {
public:
    explicit HeatEquationSolver(const PDEConfig& config);
    PINNResults solve() override;
    std::vector<double> compute_residual(const std::vector<double>& input) override;
    
private:
    std::vector<double> compute_heat_residual(const std::vector<double>& input,
                                             const std::vector<double>& output,
                                             const std::vector<std::vector<double>>& gradients,
                                             const std::vector<std::vector<std::vector<double>>>& hessians) const;
};

// Wave equation solver
class WaveEquationSolver : public PDESolver {
public:
    explicit WaveEquationSolver(const PDEConfig& config);
    PINNResults solve() override;
    std::vector<double> compute_residual(const std::vector<double>& input) override;
    
private:
    std::vector<double> compute_wave_residual(const std::vector<double>& input,
                                             const std::vector<double>& output,
                                             const std::vector<std::vector<double>>& gradients,
                                             const std::vector<std::vector<std::vector<double>>>& hessians) const;
};

// Navier-Stokes solver
class NavierStokesSolver : public PDESolver {
public:
    explicit NavierStokesSolver(const PDEConfig& config);
    PINNResults solve() override;
    std::vector<double> compute_residual(const std::vector<double>& input) override;
    
private:
    std::vector<double> compute_navier_stokes_residual(const std::vector<double>& input,
                                                      const std::vector<double>& output,
                                                      const std::vector<std::vector<double>>& gradients,
                                                      const std::vector<std::vector<std::vector<double>>>& hessians) const;
};

// Adaptive collocation point management
class AdaptiveCollocation {
public:
    explicit AdaptiveCollocation(const PDEConfig& config);
    
    void initialize_points();
    void update_points(const std::vector<CollocationPoint>& current_points,
                      const std::vector<double>& residuals);
    void refine_high_residual_regions(const std::vector<double>& residual_field);
    void coarsen_low_residual_regions();
    
    std::vector<CollocationPoint> get_collocation_points() const { return points_; }
    
private:
    PDEConfig config_;
    std::vector<CollocationPoint> points_;
    double refinement_threshold = 0.1;
    double coarsening_threshold = 0.01;
    
    void add_points_in_region(const std::vector<double>& center, double radius, size_t num_points);
    void remove_points_in_region(const std::vector<double>& center, double radius);
    std::vector<double> compute_residual_weights(const std::vector<double>& residuals) const;
};

// Uncertainty quantification for PINNs
class UncertaintyQuantification {
public:
    explicit UncertaintyQuantification(size_t ensemble_size = 10);
    
    void train_ensemble(const PDEConfig& config, const PINNArchitecture& arch);
    std::vector<double> predict_with_uncertainty(const std::vector<double>& input);
    std::vector<double> compute_confidence_intervals(const std::vector<double>& input, double confidence = 0.95);
    
    std::vector<double> get_mean_prediction() const { return mean_prediction_; }
    std::vector<double> get_std_deviation() const { return std_deviation_; }
    
private:
    size_t ensemble_size_;
    std::vector<std::unique_ptr<PhysicsInformedNN>> ensemble_;
    std::vector<double> mean_prediction_;
    std::vector<double> std_deviation_;
    
    void update_statistics(const std::vector<std::vector<double>>& ensemble_predictions);
};

// Inverse problem solver
class InverseProblemSolver {
public:
    InverseProblemSolver(const PDEConfig& config, const PINNArchitecture& arch);
    
    std::vector<double> solve_inverse_problem(const std::vector<std::vector<double>>& observations,
                                             const std::vector<std::vector<double>>& observation_points);
    std::vector<double> get_identified_parameters() const { return identified_parameters_; }
    double get_parameter_uncertainty() const { return parameter_uncertainty_; }
    
private:
    PDEConfig config_;
    PINNArchitecture architecture_;
    std::vector<double> identified_parameters_;
    double parameter_uncertainty_;
    
    double compute_misfit_loss(const std::vector<std::vector<double>>& observations,
                               const std::vector<std::vector<double>>& observation_points);
};

// Multi-scale modeling support
class MultiScaleModeling {
public:
    explicit MultiScaleModeling(const std::vector<double>& scales);
    
    void add_fine_scale_model(std::unique_ptr<PhysicsInformedNN> fine_model);
    void add_coarse_scale_model(std::unique_ptr<PhysicsInformedNN> coarse_model);
    
    std::vector<double> multi_scale_predict(const std::vector<double>& input) const;
    void update_scale_coupling();
    
private:
    std::vector<double> scales_;
    std::vector<std::unique_ptr<PhysicsInformedNN>> scale_models_;
    std::vector<double> coupling_weights_;
    
    std::vector<double> interpolate_scales(const std::vector<double>& fine_output,
                                         const std::vector<double>& coarse_output,
                                         const std::vector<double>& input) const;
};

// Convergence acceleration methods
class ConvergenceAccelerator {
public:
    ConvergenceAccelerator();
    
    void enable_preconditioning(const std::vector<std::vector<double>>& preconditioner);
    void enable_adaptive_learning_rate(double initial_lr, double decay_rate);
    void enable_momentum(double momentum_coeff);
    
    void update_learning_rate(double current_loss);
    double get_current_learning_rate() const { return current_learning_rate_; }
    
    std::vector<std::vector<double>> apply_preconditioning(const std::vector<std::vector<double>>& gradients) const;
    std::vector<std::vector<double>> apply_momentum(const std::vector<std::vector<double>>& gradients);
    
private:
    bool use_preconditioning_ = false;
    bool use_adaptive_lr_ = false;
    bool use_momentum_ = false;
    
    double current_learning_rate_ = 0.001;
    double learning_rate_decay_ = 0.95;
    double momentum_coefficient_ = 0.9;
    
    std::vector<std::vector<double>> preconditioner_matrix_;
    std::vector<std::vector<double>> momentum_buffer_;
    
    void update_momentum_buffer(const std::vector<std::vector<double>>& gradients);
};

// Utility functions
namespace Utils {
    // Domain generation
    Domain generate_rectangular_domain(const std::vector<double>& bounds, const std::vector<size_t>& resolution);
    Domain generate_circular_domain(double radius, size_t num_points);
    Domain generate_irregular_domain(const std::vector<std::vector<double>>& boundary_points);
    
    // Mesh generation
    std::vector<std::vector<double>> generate_mesh(const Domain& domain);
    std::vector<std::vector<double>> refine_mesh(const std::vector<std::vector<double>>& mesh,
                                                const std::vector<double>& error_indicator);
    
    // Boundary condition handling
    std::vector<double> apply_dirichlet_bc(const std::vector<double>& input, double value);
    std::vector<double> apply_neumann_bc(const std::vector<double>& input, double derivative);
    std::vector<double> apply_periodic_bc(const std::vector<double>& input);
    
    // Error analysis
    double compute_l2_error(const std::vector<double>& numerical, const std::vector<double>& analytical);
    double compute_linf_error(const std::vector<double>& numerical, const std::vector<double>& analytical);
    std::vector<double> compute_residual_norms(const std::vector<std::vector<double>>& residuals);
    
    // Visualization helpers
    void export_solution_to_vtk(const std::vector<std::vector<double>>& solution, 
                                const Domain& domain, const std::string& filename);
    void plot_convergence_history(const std::vector<double>& loss_history, const std::string& filename);
}

} // namespace Physics
} // namespace ML

#endif // PHYSICS_INFORMED_NN_H
