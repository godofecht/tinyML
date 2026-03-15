//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek Shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "PhysicsInformedNN.h"
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <random>

using namespace ML::Physics;

namespace {
PDEConfig create_basic_heat_config() {
    PDEConfig config;
    config.type = PDEType::HEAT_EQUATION;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    config.diffusion_coefficient = 0.1;
    config.num_collocation_points = 100;
    config.learning_rate = 0.001;
    config.max_epochs = 100;
    config.boundary_conditions = {BoundaryConditionType::DIRICHLET};
    return config;
}
} // namespace

class ComprehensivePhysicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup comprehensive test configurations
    }
};

// ========== CORE PINN FUNCTIONALITY TESTS ==========

TEST_F(ComprehensivePhysicsTest, PINNInitialization) {
    PDEConfig config;
    config.type = PDEType::HEAT_EQUATION;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    config.diffusion_coefficient = 0.1;
    config.num_collocation_points = 100;
    config.learning_rate = 0.001;
    config.max_epochs = 100;
    
    PINNArchitecture arch(2, 1, {20, 20});
    
    auto pinn = std::make_unique<PhysicsInformedNN>(config, arch);
    
    EXPECT_TRUE(pinn->initialize());
    EXPECT_EQ(pinn->get_collocation_points().size(), 100);
    
    // Test prediction
    std::vector<double> input = {0.5, 0.5};
    auto output = pinn->predict(input);
    
    EXPECT_EQ(output.size(), 1);
    EXPECT_FALSE(std::isnan(output[0]));
    EXPECT_FALSE(std::isinf(output[0]));
    
    // Test residual computation
    auto residual = pinn->compute_residual(input);
    EXPECT_EQ(residual.size(), 1);
}

TEST_F(ComprehensivePhysicsTest, NetworkArchitectureVariations) {
    PDEConfig config = create_basic_heat_config();
    
    std::vector<PINNArchitecture> architectures = {
        PINNArchitecture(2, 1, {10}),           // Single layer
        PINNArchitecture(2, 1, {20, 20}),        // Two layers
        PINNArchitecture(2, 1, {30, 30, 30}),   // Three layers
        PINNArchitecture(2, 1, {40, 40, 40, 40}) // Four layers
    };
    
    for (size_t i = 0; i < architectures.size(); ++i) {
        auto pinn = std::make_unique<PhysicsInformedNN>(config, architectures[i]);
        EXPECT_TRUE(pinn->initialize()) << "Failed to initialize architecture " << i;
        
        std::vector<double> input = {0.5, 0.5};
        auto output = pinn->predict(input);
        EXPECT_EQ(output.size(), 1) << "Wrong output size for architecture " << i;
        EXPECT_FALSE(std::isnan(output[0])) << "NaN output for architecture " << i;
    }
}

TEST_F(ComprehensivePhysicsTest, ActivationFunctionComparison) {
    PDEConfig config = create_basic_heat_config();
    
    std::vector<std::string> activations = {"tanh", "relu", "sigmoid"};
    
    for (const auto& activation : activations) {
        PINNArchitecture arch(2, 1, {20, 20});
        arch.activation = activation;
        
        auto pinn = std::make_unique<PhysicsInformedNN>(config, arch);
        EXPECT_TRUE(pinn->initialize()) << "Failed to initialize with " << activation;
        
        std::vector<double> input = {0.5, 0.5};
        auto output = pinn->predict(input);
        
        EXPECT_FALSE(std::isnan(output[0])) << "NaN output with " << activation;
        EXPECT_FALSE(std::isinf(output[0])) << "Inf output with " << activation;
    }
}

// ========== PDE-SPECIFIC SOLVER TESTS ==========

TEST_F(ComprehensivePhysicsTest, HeatEquationSteadyState) {
    auto solver = PDESolver::create_heat_solver(create_basic_heat_config());
    ASSERT_NE(solver, nullptr);
    
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_GE(results.final_loss, 0.0);
    EXPECT_GT(results.training_time_ms, 0.0);
    
    // Test solution properties
    std::vector<std::vector<double>> test_points = {
        {0.0, 0.0}, {0.5, 0.5}, {1.0, 1.0}
    };
    
    for (const auto& point : test_points) {
        auto residual = solver->compute_residual(point);
        EXPECT_LT(std::abs(residual[0]), 1e-1) << "High residual at point (" 
                                                << point[0] << ", " << point[1] << ")";
    }
}

TEST_F(ComprehensivePhysicsTest, HeatEquationTimeDependent) {
    PDEConfig config = create_basic_heat_config();
    config.is_time_dependent = true;
    config.time_span = 1.0;
    config.num_time_steps = 10;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {10, 10, 10}); // Add time dimension
    
    auto solver = PDESolver::create_heat_solver(config);
    ASSERT_NE(solver, nullptr);
    
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_LT(results.final_loss, 1e-2);
    
    // Test time evolution
    std::vector<double> input = {0.5, 0.5, 0.5}; // x, y, t
    auto residual = solver->compute_residual(input);
    EXPECT_LT(std::abs(residual[0]), 1e-1);
}

TEST_F(ComprehensivePhysicsTest, WaveEquationProperties) {
    PDEConfig config;
    config.type = PDEType::WAVE_EQUATION;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0, 0.0, 2.0}, {5, 5, 5});
    config.wave_speed = 1.0;
    config.is_time_dependent = true;
    config.num_time_steps = 2;
    config.num_collocation_points = 50;
    config.max_epochs = 3;
    
    auto solver = PDESolver::create_wave_solver(config);
    ASSERT_NE(solver, nullptr);
    
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_LT(results.final_loss, 1e-2);
    
    // Test wave propagation
    std::vector<std::vector<double>> test_points = {
        {0.5, 0.5, 0.0}, // Initial time
        {0.5, 0.5, 0.5}, // Middle time
        {0.5, 0.5, 1.0}  // Final time
    };
    
    for (const auto& point : test_points) {
        auto residual = solver->compute_residual(point);
        EXPECT_LT(std::abs(residual[0]), 1e-1);
    }
}

TEST_F(ComprehensivePhysicsTest, NavierStokesIncompressibility) {
    PDEConfig config;
    config.type = PDEType::NAVIER_STOKES;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {5, 5});
    config.viscosity = 0.01;
    config.is_time_dependent = true;
    config.num_time_steps = 2;
    config.num_collocation_points = 75;
    config.max_epochs = 3;
    
    auto solver = PDESolver::create_navier_stokes_solver(config);
    ASSERT_NE(solver, nullptr);
    
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_LT(results.final_loss, 1e-1);
    
    // Test incompressibility (∇·u = 0)
    std::vector<double> input = {0.5, 0.5, 0.5};
    auto residual = solver->compute_residual(input);
    
    EXPECT_EQ(residual.size(), 3); // u, v, continuity equations
    EXPECT_LT(std::abs(residual[2]), 1e-1); // Continuity equation should be small
}

// ========== ADVANCED FEATURES TESTS ==========

TEST_F(ComprehensivePhysicsTest, AdaptiveCollocationEffectiveness) {
    PDEConfig config = create_basic_heat_config();
    config.num_collocation_points = 100;
    
    AdaptiveCollocation adaptive(config);
    
    auto initial_points = adaptive.get_collocation_points();
    EXPECT_EQ(initial_points.size(), 100);
    
    // Simulate multiple refinement cycles
    for (int cycle = 0; cycle < 5; ++cycle) {
        std::vector<double> residuals(initial_points.size());
        
        // Create higher residuals in certain regions
        for (size_t i = 0; i < residuals.size(); ++i) {
            double x = initial_points[i].coordinates[0];
            double y = initial_points[i].coordinates[1];
            
            // Higher residuals near boundaries
            double dist_to_boundary = std::min({x, y, 1.0-x, 1.0-y});
            residuals[i] = 0.2 * (1.0 - dist_to_boundary);
        }
        
        adaptive.update_points(initial_points, residuals);
        initial_points = adaptive.get_collocation_points();
    }
    
    // Should have refined in high residual regions
    EXPECT_GT(initial_points.size(), 100);
    
    // Check that points are concentrated in high residual regions
    size_t boundary_points = 0;
    for (const auto& point : initial_points) {
        double x = point.coordinates[0];
        double y = point.coordinates[1];
        double dist_to_boundary = std::min({x, y, 1.0-x, 1.0-y});
        
        if (dist_to_boundary < 0.1) {
            boundary_points++;
        }
    }
    
    // Should have more points near boundaries after refinement
    EXPECT_GT(boundary_points, initial_points.size() * 0.3);
}

TEST_F(ComprehensivePhysicsTest, UncertaintyQuantificationAccuracy) {
    PDEConfig config = create_basic_heat_config();
    config.max_epochs = 50; // Reduced for testing
    
    PINNArchitecture arch(2, 1, {15, 15});
    
    UncertaintyQuantification uq(3); // Small ensemble for testing
    uq.train_ensemble(config, arch);
    
    std::vector<double> input = {0.5, 0.5};
    auto prediction = uq.predict_with_uncertainty(input);
    auto confidence_intervals = uq.compute_confidence_intervals(input, 0.95);
    
    EXPECT_FALSE(prediction.empty());
    EXPECT_EQ(confidence_intervals.size(), 2); // Lower and upper bounds
    EXPECT_LT(confidence_intervals[0], confidence_intervals[1]); // Lower < upper
    
    // Check that confidence intervals are reasonable
    double width = confidence_intervals[1] - confidence_intervals[0];
    EXPECT_GT(width, 0.0);
    EXPECT_LT(width, 2.0); // Shouldn't be too wide
    
    // Test multiple points
    std::vector<std::vector<double>> test_points = {
        {0.25, 0.25}, {0.5, 0.5}, {0.75, 0.75}
    };
    
    for (const auto& point : test_points) {
        auto pred = uq.predict_with_uncertainty(point);
        auto ci = uq.compute_confidence_intervals(point, 0.95);
        
        EXPECT_FALSE(pred.empty());
        EXPECT_EQ(ci.size(), 2);
        EXPECT_LT(ci[0], ci[1]);
    }
}

TEST_F(ComprehensivePhysicsTest, InverseProblemConvergence) {
    PDEConfig config = create_basic_heat_config();
    config.custom_parameters = {0.1}; // Unknown diffusion coefficient
    config.max_epochs = 100;
    
    PINNArchitecture arch(2, 1, {20, 20});
    
    InverseProblemSolver inverse_solver(config, arch);
    
    // Create synthetic observations with known parameter
    double true_diffusion = 0.15;
    std::vector<std::vector<double>> observations;
    std::vector<std::vector<double>> observation_points;
    
    for (double x = 0.2; x <= 0.8; x += 0.2) {
        for (double y = 0.2; y <= 0.8; y += 0.2) {
            observation_points.push_back({x, y});
            
            // Simple analytical solution for steady-state heat equation
            double analytical_solution = true_diffusion * x * (1.0 - x) * y * (1.0 - y);
            observations.push_back({analytical_solution});
        }
    }
    
    auto identified_params = inverse_solver.solve_inverse_problem(observations, observation_points);
    
    EXPECT_EQ(identified_params.size(), 1);
    EXPECT_GT(identified_params[0], 0.0);
    EXPECT_LT(identified_params[0], 1.0);
    
    // Should be reasonably close to true value
    double relative_error = std::abs(identified_params[0] - true_diffusion) / true_diffusion;
    EXPECT_LT(relative_error, 0.5) << "Parameter identification error too large: " << relative_error;
    
    // Check uncertainty
    double uncertainty = inverse_solver.get_parameter_uncertainty();
    EXPECT_GT(uncertainty, 0.0);
    EXPECT_LT(uncertainty, 1.0);
}

TEST_F(ComprehensivePhysicsTest, MultiScaleModelingConsistency) {
    std::vector<double> scales = {0.1, 1.0, 10.0};
    MultiScaleModeling multi_scale(scales);
    
    PDEConfig config = create_basic_heat_config();
    PINNArchitecture arch(2, 1, {15, 15});
    
    // Add models for different scales
    auto fine_model = std::make_unique<PhysicsInformedNN>(config, arch);
    auto coarse_model = std::make_unique<PhysicsInformedNN>(config, arch);
    
    fine_model->initialize();
    coarse_model->initialize();
    
    multi_scale.add_fine_scale_model(std::move(fine_model));
    multi_scale.add_coarse_scale_model(std::move(coarse_model));
    
    std::vector<double> input = {0.5, 0.5};
    auto multi_scale_prediction = multi_scale.multi_scale_predict(input);
    
    EXPECT_FALSE(multi_scale_prediction.empty());
    EXPECT_EQ(multi_scale_prediction.size(), 1);
    
    // Test consistency across different inputs
    std::vector<std::vector<double>> test_inputs = {
        {0.25, 0.25}, {0.5, 0.5}, {0.75, 0.75}
    };
    
    for (const auto& test_input : test_inputs) {
        auto prediction = multi_scale.multi_scale_predict(test_input);
        EXPECT_FALSE(prediction.empty());
        EXPECT_EQ(prediction.size(), 1);
        EXPECT_FALSE(std::isnan(prediction[0]));
        EXPECT_FALSE(std::isinf(prediction[0]));
    }
}

TEST_F(ComprehensivePhysicsTest, ConvergenceAccelerationEffectiveness) {
    PDEConfig config = create_basic_heat_config();
    config.max_epochs = 10;
    
    PINNArchitecture arch(2, 1, {20, 20});
    
    // Test without acceleration
    auto pinn_baseline = std::make_unique<PhysicsInformedNN>(config, arch);
    pinn_baseline->initialize();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto results_baseline = pinn_baseline->solve();
    auto baseline_time = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    
    // Test with acceleration
    auto pinn_accelerated = std::make_unique<PhysicsInformedNN>(config, arch);
    pinn_accelerated->initialize();
    pinn_accelerated->enable_preconditioning();
    pinn_accelerated->enable_adaptive_learning_rate();
    
    start_time = std::chrono::high_resolution_clock::now();
    auto results_accelerated = pinn_accelerated->solve();
    auto accelerated_time = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    
    // Both should converge
    EXPECT_GT(results_baseline.epochs_trained, 0);
    EXPECT_GT(results_accelerated.epochs_trained, 0);
    
    // Convergence metrics should be finite and acceleration should not be wildly slower.
    EXPECT_TRUE(std::isfinite(results_baseline.convergence_rate));
    EXPECT_TRUE(std::isfinite(results_accelerated.convergence_rate));
    if (baseline_time > 0.0) {
        EXPECT_LT(accelerated_time, baseline_time * 5.0)
            << "Acceleration path unexpectedly slow";
    }
    
    // Both should achieve similar final losses
    double loss_ratio = results_baseline.final_loss > 0.0
                            ? results_accelerated.final_loss / results_baseline.final_loss
                            : 1.0;
    EXPECT_GT(loss_ratio, 0.01) << "Accelerated version quality degraded too much";
    EXPECT_LT(loss_ratio, 10.0) << "Accelerated version quality diverged too much";
}

// ========== BOUNDARY CONDITION TESTS ==========

TEST_F(ComprehensivePhysicsTest, BoundaryConditionTypes) {
    std::vector<BoundaryConditionType> bc_types = {
        BoundaryConditionType::DIRICHLET,
        BoundaryConditionType::NEUMANN,
        BoundaryConditionType::PERIODIC
    };
    
    for (auto bc_type : bc_types) {
        PDEConfig config = create_basic_heat_config();
        config.boundary_conditions = {bc_type};
        
        auto solver = PDESolver::create_heat_solver(config);
        ASSERT_NE(solver, nullptr);
        
        auto results = solver->solve();
        EXPECT_GT(results.epochs_trained, 0) << "Failed to converge with BC type " << static_cast<int>(bc_type);
        
        // Test boundary points
        std::vector<std::vector<double>> boundary_test_points = {
            {0.0, 0.5}, {1.0, 0.5}, {0.5, 0.0}, {0.5, 1.0}
        };
        
        for (const auto& point : boundary_test_points) {
            auto residual = solver->compute_residual(point);
            EXPECT_LT(std::abs(residual[0]), 1e-1) << "High residual at boundary with BC type " 
                                                    << static_cast<int>(bc_type);
        }
    }
}

TEST_F(ComprehensivePhysicsTest, MixedBoundaryConditions) {
    PDEConfig config = create_basic_heat_config();
    config.boundary_conditions = {
        BoundaryConditionType::DIRICHLET,   // Left boundary
        BoundaryConditionType::NEUMANN,     // Right boundary
        BoundaryConditionType::DIRICHLET,   // Bottom boundary
        BoundaryConditionType::NEUMANN      // Top boundary
    };
    
    auto solver = PDESolver::create_heat_solver(config);
    ASSERT_NE(solver, nullptr);
    
    auto results = solver->solve();
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_LT(results.final_loss, 1e-2);
}

// ========== PERFORMANCE AND SCALING TESTS ==========

TEST_F(ComprehensivePhysicsTest, PerformanceScalingWithNetworkSize) {
    std::vector<std::vector<size_t>> network_sizes = {
        {10}, {20, 20}, {30, 30, 30}, {40, 40, 40, 40}
    };
    
    std::vector<double> inference_times;
    std::vector<double> memory_usage;
    
    for (const auto& net_size : network_sizes) {
        PDEConfig config = create_basic_heat_config();
        config.max_epochs = 10; // Reduced for performance testing
        
        PINNArchitecture arch(2, 1, net_size);
        auto pinn = std::make_unique<PhysicsInformedNN>(config, arch);
        pinn->initialize();
        
        // Measure inference time
        std::vector<double> input = {0.5, 0.5};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; ++i) {
            pinn->predict(input);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count() / 20.0;
        inference_times.push_back(inference_time);
        
        // Estimate memory usage
        size_t total_params = 0;
        size_t input_size = 2;
        for (size_t layer_size : net_size) {
            total_params += input_size * layer_size + layer_size;
            input_size = layer_size;
        }
        total_params += input_size + 1; // Output layer
        
        double memory_mb = total_params * 8 / (1024.0 * 1024.0);
        memory_usage.push_back(memory_mb);
    }
    
    // Check that performance scales reasonably
    for (size_t i = 1; i < inference_times.size(); ++i) {
        if (inference_times[i - 1] > 0.01) {
            EXPECT_LT(inference_times[i], inference_times[i-1] * 5.0)
                << "Inference time scaling too aggressive for network size " << i;
        }
        EXPECT_GE(memory_usage[i], memory_usage[i - 1])
            << "Memory usage should not decrease for larger network size " << i;
        EXPECT_LT(memory_usage[i], 1.0)
            << "Unexpected memory usage spike for network size " << i;
    }
    
    // All should meet performance target
    for (double time : inference_times) {
        EXPECT_LT(time, 2.0) << "Inference time exceeds target: " << time << "ms";
    }
}

TEST_F(ComprehensivePhysicsTest, PerformanceScalingWithDomainSize) {
    std::vector<size_t> point_counts = {50, 100, 200, 500};
    
    for (size_t num_points : point_counts) {
        PDEConfig config = create_basic_heat_config();
        config.num_collocation_points = num_points;
        config.max_epochs = 100; // Reduced for testing
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto solver = PDESolver::create_heat_solver(config);
        auto results = solver->solve();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto solving_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        EXPECT_GT(results.epochs_trained, 0) << "Failed to converge with " << num_points << " points";
        EXPECT_LT(solving_time, 30000.0) << "Solving time too long: " << solving_time << "ms";
        
        // Test inference time
        std::vector<double> input = {0.5, 0.5};
        
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 50; ++i) {
            solver->compute_residual(input);
        }
        end_time = std::chrono::high_resolution_clock::now();
        
        auto inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count() / 50.0;
        EXPECT_LT(inference_time, 2.0) << "Inference time too high: " << inference_time << "ms";
    }
}

// ========== ROBUSTNESS AND STABILITY TESTS ==========

TEST_F(ComprehensivePhysicsTest, NumericalStability) {
    PDEConfig config = create_basic_heat_config();
    config.max_epochs = 100;
    config.learning_rate = 0.01; // Higher learning rate
    
    PINNArchitecture arch(2, 1, {30, 30});
    auto pinn = std::make_unique<PhysicsInformedNN>(config, arch);
    
    EXPECT_TRUE(pinn->initialize());
    
    // Should not crash with higher learning rate
    auto results = pinn->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_FALSE(std::isnan(results.final_loss));
    EXPECT_FALSE(std::isinf(results.final_loss));
    
    // Predictions should be stable
    std::vector<double> input = {0.5, 0.5};
    auto output = pinn->predict(input);
    
    EXPECT_FALSE(std::isnan(output[0]));
    EXPECT_FALSE(std::isinf(output[0]));
    EXPECT_LT(std::abs(output[0]), 100.0); // Reasonable magnitude
}

TEST_F(ComprehensivePhysicsTest, ExtremeParameterValues) {
    // Test with extreme diffusion coefficients
    std::vector<double> diffusion_values = {1e-6, 1e-3, 1.0, 10.0, 100.0};
    
    for (double diffusion : diffusion_values) {
        PDEConfig config = create_basic_heat_config();
        config.diffusion_coefficient = diffusion;
        config.max_epochs = 50;
        
        auto solver = PDESolver::create_heat_solver(config);
        ASSERT_NE(solver, nullptr);
        
        auto results = solver->solve();
        
        EXPECT_GT(results.epochs_trained, 0) << "Failed to converge with diffusion " << diffusion;
        EXPECT_FALSE(std::isnan(results.final_loss)) << "NaN loss with diffusion " << diffusion;
        EXPECT_FALSE(std::isinf(results.final_loss)) << "Inf loss with diffusion " << diffusion;
    }
}

TEST_F(ComprehensivePhysicsTest, RandomInputRobustness) {
    PDEConfig config = create_basic_heat_config();
    auto solver = PDESolver::create_heat_solver(config);
    auto results = solver->solve();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.1, 1.1); // Slightly outside domain
    
    // Test with random inputs (including some outside domain)
    for (int i = 0; i < 100; ++i) {
        std::vector<double> input = {dis(gen), dis(gen)};
        auto residual = solver->compute_residual(input);
        
        // Should not crash
        EXPECT_EQ(residual.size(), 1);
        
        // Should be finite (even if outside domain)
        EXPECT_FALSE(std::isnan(residual[0]));
        EXPECT_FALSE(std::isinf(residual[0]));
    }
}

// ========== INTEGRATION TESTS ==========

TEST_F(ComprehensivePhysicsTest, MultiPhysicsCoupling) {
    // Create coupled heat and flow problem
    PDEConfig heat_config = create_basic_heat_config();
    heat_config.diffusion_coefficient = 0.1;
    
    PDEConfig flow_config;
    flow_config.type = PDEType::NAVIER_STOKES;
    flow_config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    flow_config.viscosity = 0.01;
    flow_config.num_collocation_points = 200;
    flow_config.max_epochs = 100;
    
    auto heat_solver = PDESolver::create_heat_solver(heat_config);
    auto flow_solver = PDESolver::create_navier_stokes_solver(flow_config);
    
    auto heat_results = heat_solver->solve();
    auto flow_results = flow_solver->solve();
    
    EXPECT_GT(heat_results.epochs_trained, 0);
    EXPECT_GT(flow_results.epochs_trained, 0);
    
    // Test that both solvers work on the same domain
    std::vector<double> test_point = {0.5, 0.5};
    
    auto heat_residual = heat_solver->compute_residual(test_point);
    auto flow_residual = flow_solver->compute_residual(test_point);
    
    EXPECT_EQ(heat_residual.size(), 1);
    EXPECT_EQ(flow_residual.size(), 3); // u, v, continuity
    
    EXPECT_LT(std::abs(heat_residual[0]), 1e-1);
    EXPECT_LT(std::abs(flow_residual[2]), 1e-1); // Continuity equation
}

TEST_F(ComprehensivePhysicsTest, EndToEndWorkflow) {
    // Complete workflow: problem setup -> solving -> validation -> uncertainty
    
    // 1. Setup problem
    PDEConfig config = create_basic_heat_config();
    config.is_time_dependent = true;
    config.time_span = 1.0;
    config.num_time_steps = 5;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {10, 10, 10});
    
    // 2. Solve
    auto solver = PDESolver::create_heat_solver(config);
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_LT(results.final_loss, 1e-2);
    
    // 3. Validate solution
    std::vector<std::vector<double>> validation_points = {
        {0.25, 0.25, 0.5}, {0.5, 0.5, 0.5}, {0.75, 0.75, 0.5}
    };
    
    for (const auto& point : validation_points) {
        auto residual = solver->compute_residual(point);
        EXPECT_LT(std::abs(residual[0]), 1e-1);
    }
    
    // 4. Uncertainty analysis
    UncertaintyQuantification uq(3);
    PINNArchitecture arch(3, 1, {20, 20}); // x, y, t -> temperature
    
    uq.train_ensemble(config, arch);
    
    std::vector<double> test_input = {0.5, 0.5, 0.5};
    auto prediction = uq.predict_with_uncertainty(test_input);
    auto confidence_intervals = uq.compute_confidence_intervals(test_input, 0.95);
    
    EXPECT_FALSE(prediction.empty());
    EXPECT_EQ(confidence_intervals.size(), 2);
    EXPECT_LT(confidence_intervals[0], confidence_intervals[1]);
}

// ========== MAIN TEST RUNNER ==========

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=== Comprehensive Physics-Informed Neural Network Test Suite ===" << std::endl;
    std::cout << "Testing PINN functionality, performance, and robustness..." << std::endl;
    
    auto result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << "\n🎉 All PINN tests passed!" << std::endl;
        std::cout << "✅ Core functionality verified" << std::endl;
        std::cout << "✅ Advanced features validated" << std::endl;
        std::cout << "✅ Performance targets met" << std::endl;
        std::cout << "✅ Robustness confirmed" << std::endl;
    } else {
        std::cout << "\n❌ Some PINN tests failed!" << std::endl;
    }
    
    return result;
}
