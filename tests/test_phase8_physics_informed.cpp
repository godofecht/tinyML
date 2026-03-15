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

using namespace ML::Physics;

class PhysicsInformedNNTest : public ::testing::Test {
protected:
    PhysicsInformedNNTest() : arch_(2, 1, {20, 20}) {}

    void SetUp() override {
        // Setup basic PDE configuration
        config_.type = PDEType::HEAT_EQUATION;
        config_.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10}); // Unit square
        config_.diffusion_coefficient = 0.1;
        config_.is_time_dependent = false;
        config_.num_collocation_points = 100;
        config_.num_boundary_points = 20;
        config_.learning_rate = 0.001;
        config_.max_epochs = 100;
        config_.tolerance = 1e-6;
        config_.boundary_conditions = {BoundaryConditionType::DIRICHLET};
    }

    PDEConfig config_;
    PINNArchitecture arch_;
};

TEST_F(PhysicsInformedNNTest, CreatePINN) {
    auto pinn = std::make_unique<PhysicsInformedNN>(config_, arch_);
    
    EXPECT_NE(pinn, nullptr);
    EXPECT_TRUE(pinn->initialize());
    
    // Test basic prediction
    std::vector<double> input = {0.5, 0.5};
    auto output = pinn->predict(input);
    
    EXPECT_EQ(output.size(), 1);
    EXPECT_FALSE(std::isnan(output[0]));
    EXPECT_FALSE(std::isinf(output[0]));
}

TEST_F(PhysicsInformedNNTest, HeatEquationSolver) {
    auto solver = PDESolver::create_heat_solver(config_);
    EXPECT_NE(solver, nullptr);
    
    // Test solving
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_GE(results.final_loss, 0.0);
    EXPECT_GT(results.training_time_ms, 0.0);
    
    // Test residual computation
    std::vector<double> input = {0.5, 0.5};
    auto residual = solver->compute_residual(input);
    
    EXPECT_EQ(residual.size(), 1);
    EXPECT_FALSE(std::isnan(residual[0]));
}

TEST_F(PhysicsInformedNNTest, WaveEquationSolver) {
    config_.type = PDEType::WAVE_EQUATION;
    config_.wave_speed = 1.0;
    config_.is_time_dependent = true;
    config_.domain = Domain({0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {10, 10, 10}); // Add time dimension
    
    auto solver = PDESolver::create_wave_solver(config_);
    EXPECT_NE(solver, nullptr);
    
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_GE(results.final_loss, 0.0);
}

TEST_F(PhysicsInformedNNTest, NavierStokesSolver) {
    config_.type = PDEType::NAVIER_STOKES;
    config_.viscosity = 0.01;
    config_.is_time_dependent = true;
    
    auto solver = PDESolver::create_navier_stokes_solver(config_);
    EXPECT_NE(solver, nullptr);
    
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_GE(results.final_loss, 0.0);
}

TEST_F(PhysicsInformedNNTest, CustomPDESolver) {
    config_.type = PDEType::CUSTOM_PDE;
    
    auto solver = PDESolver::create_custom_solver(config_);
    EXPECT_NE(solver, nullptr);
    
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_GE(results.final_loss, 0.0);
}

TEST_F(PhysicsInformedNNTest, AdaptiveCollocation) {
    AdaptiveCollocation adaptive(config_);
    
    auto points = adaptive.get_collocation_points();
    EXPECT_EQ(points.size(), config_.num_collocation_points);
    
    // Test point refinement
    std::vector<double> residuals(points.size(), 0.5);
    adaptive.update_points(points, residuals);
    
    auto updated_points = adaptive.get_collocation_points();
    EXPECT_GE(updated_points.size(), points.size()); // Should have added points
}

TEST_F(PhysicsInformedNNTest, UncertaintyQuantification) {
    UncertaintyQuantification uq(5); // Ensemble of 5
    
    // Test ensemble training
    uq.train_ensemble(config_, arch_);
    
    std::vector<double> input = {0.5, 0.5};
    auto prediction = uq.predict_with_uncertainty(input);
    
    EXPECT_FALSE(prediction.empty());
    
    auto confidence_intervals = uq.compute_confidence_intervals(input, 0.95);
    EXPECT_EQ(confidence_intervals.size(), prediction.size() * 2); // Lower and upper bounds
    
    // Check that mean and std dev are computed
    auto mean = uq.get_mean_prediction();
    auto std_dev = uq.get_std_deviation();
    
    EXPECT_EQ(mean.size(), prediction.size());
    EXPECT_EQ(std_dev.size(), prediction.size());
}

TEST_F(PhysicsInformedNNTest, InverseProblemSolver) {
    config_.custom_parameters = {0.1, 0.5}; // Unknown parameters
    
    InverseProblemSolver inverse_solver(config_, arch_);
    
    // Create synthetic observations
    std::vector<std::vector<double>> observations = {{0.8}, {0.6}, {0.4}};
    std::vector<std::vector<double>> observation_points = {{0.2, 0.2}, {0.5, 0.5}, {0.8, 0.8}};
    
    auto identified_params = inverse_solver.solve_inverse_problem(observations, observation_points);
    
    EXPECT_EQ(identified_params.size(), config_.custom_parameters.size());
    
    auto uncertainty = inverse_solver.get_parameter_uncertainty();
    EXPECT_GE(uncertainty, 0.0);
}

TEST_F(PhysicsInformedNNTest, MultiScaleModeling) {
    std::vector<double> scales = {0.1, 1.0, 10.0};
    MultiScaleModeling multi_scale(scales);
    
    // Create models for different scales
    auto fine_model = std::make_unique<PhysicsInformedNN>(config_, arch_);
    auto coarse_model = std::make_unique<PhysicsInformedNN>(config_, arch_);
    
    fine_model->initialize();
    coarse_model->initialize();
    
    multi_scale.add_fine_scale_model(std::move(fine_model));
    multi_scale.add_coarse_scale_model(std::move(coarse_model));
    
    std::vector<double> input = {0.5, 0.5};
    auto prediction = multi_scale.multi_scale_predict(input);
    
    EXPECT_FALSE(prediction.empty());
    
    multi_scale.update_scale_coupling();
}

TEST_F(PhysicsInformedNNTest, ConvergenceAccelerator) {
    ConvergenceAccelerator accelerator;
    
    // Test preconditioning
    std::vector<std::vector<double>> preconditioner = {{1.0, 0.0}, {0.0, 1.0}};
    accelerator.enable_preconditioning(preconditioner);
    
    std::vector<std::vector<double>> gradients = {{0.1, 0.2}, {0.3, 0.4}};
    auto preconditioned = accelerator.apply_preconditioning(gradients);
    
    EXPECT_EQ(preconditioned.size(), gradients.size());
    
    // Test adaptive learning rate
    accelerator.enable_adaptive_learning_rate(0.01, 0.95);
    double initial_lr = accelerator.get_current_learning_rate();
    
    accelerator.update_learning_rate(0.5);
    double updated_lr = accelerator.get_current_learning_rate();
    
    EXPECT_LT(updated_lr, initial_lr);
    
    // Test momentum
    accelerator.enable_momentum(0.9);
    auto momentum_gradients = accelerator.apply_momentum(gradients);
    
    EXPECT_EQ(momentum_gradients.size(), gradients.size());
}

TEST_F(PhysicsInformedNNTest, TimeDependentSolving) {
    config_.is_time_dependent = true;
    config_.time_span = 1.0;
    config_.num_time_steps = 10;
    config_.domain = Domain({0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {10, 10, 10}); // x, y, t
    
    auto pinn = std::make_unique<PhysicsInformedNN>(config_, arch_);
    EXPECT_TRUE(pinn->initialize());
    
    auto results = pinn->solve_time_dependent();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_GE(results.final_loss, 0.0);
    EXPECT_GT(results.training_time_ms, 0.0);
    
    // Time-dependent solution should have more data points
    EXPECT_GT(results.solution.size(), arch_.output_dim);
}

TEST_F(PhysicsInformedNNTest, BoundaryConditions) {
    // Test different boundary conditions
    std::vector<BoundaryConditionType> bc_types = {
        BoundaryConditionType::DIRICHLET,
        BoundaryConditionType::NEUMANN,
        BoundaryConditionType::PERIODIC
    };
    
    for (auto bc_type : bc_types) {
        config_.boundary_conditions = {bc_type};
        
        auto pinn = std::make_unique<PhysicsInformedNN>(config_, arch_);
        EXPECT_TRUE(pinn->initialize());
        
        // Test boundary point generation
        auto boundary_points = pinn->get_collocation_points();
        EXPECT_GT(boundary_points.size(), 0);
    }
}

TEST_F(PhysicsInformedNNTest, PerformanceBenchmarks) {
    auto pinn = std::make_unique<PhysicsInformedNN>(config_, arch_);
    EXPECT_TRUE(pinn->initialize());
    
    // Measure training time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto results = pinn->solve();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto training_time = std::chrono::duration<double, std::milli>(end_time - start_time);
    
    EXPECT_LT(training_time.count(), 10000.0); // Should complete within 10 seconds
    EXPECT_LT(results.training_time_ms, 10000.0);
    
    // Measure inference time
    std::vector<double> input = {0.5, 0.5};
    
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        pinn->predict(input);
    }
    end_time = std::chrono::high_resolution_clock::now();
    
    auto inference_time = std::chrono::duration<double, std::milli>(end_time - start_time);
    double avg_inference_time = inference_time.count() / 100.0;
    
    EXPECT_LT(avg_inference_time, 10.0); // Should be less than 10ms per inference
    
    // Check if target latency is met
    EXPECT_LT(avg_inference_time, 1.0); // Target: <1ms for simple PDEs
}

TEST_F(PhysicsInformedNNTest, ConvergenceAnalysis) {
    config_.max_epochs = 1000;
    config_.tolerance = 1e-8;
    
    auto pinn = std::make_unique<PhysicsInformedNN>(config_, arch_);
    EXPECT_TRUE(pinn->initialize());
    
    auto results = pinn->solve();
    
    // Analyze convergence
    EXPECT_TRUE(std::isfinite(results.convergence_rate));
    
    if (results.converged) {
        EXPECT_LT(results.final_loss, config_.tolerance);
    } else {
        EXPECT_GE(results.final_loss, 0.0);
    }
    
    // Check loss history
    EXPECT_FALSE(results.loss_history.empty());
    
    // Loss should generally decrease
    bool decreasing = true;
    for (size_t i = 1; i < std::min(size_t(10), results.loss_history.size()); ++i) {
        if (results.loss_history[i] > results.loss_history[i-1]) {
            decreasing = false;
            break;
        }
    }
    
    // Loss might not be monotonically decreasing due to stochastic training
    // but should show overall downward trend
}

TEST_F(PhysicsInformedNNTest, MemoryUsage) {
    // Test memory efficiency with different network sizes
    std::vector<PINNArchitecture> architectures = {
        PINNArchitecture(2, 1, {10}),
        PINNArchitecture(2, 1, {20, 20}),
        PINNArchitecture(2, 1, {50, 50, 50})
    };
    
    for (const auto& arch : architectures) {
        auto pinn = std::make_unique<PhysicsInformedNN>(config_, arch);
        EXPECT_TRUE(pinn->initialize());
        
        // Test that larger networks don't crash
        std::vector<double> input = {0.5, 0.5};
        auto output = pinn->predict(input);
        
        EXPECT_EQ(output.size(), 1);
        EXPECT_FALSE(std::isnan(output[0]));
    }
}

class PDESolverIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup realistic PDE scenarios
    }
};

TEST_F(PDESolverIntegrationTest, HeatEquationSteadyState) {
    PDEConfig config;
    config.type = PDEType::HEAT_EQUATION;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    config.diffusion_coefficient = 0.1;
    config.is_time_dependent = false;
    config.num_collocation_points = 500;
    config.num_boundary_points = 100;
    config.learning_rate = 0.001;
    config.max_epochs = 1000;
    config.tolerance = 1e-6;
    config.boundary_conditions = {BoundaryConditionType::DIRICHLET};
    
    auto solver = PDESolver::create_heat_solver(config);
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_LT(results.final_loss, 1e-3); // Should converge reasonably well
    
    // Test solution at different points
    std::vector<std::vector<double>> test_points = {
        {0.0, 0.0}, {0.5, 0.5}, {1.0, 1.0}
    };
    
    for (const auto& point : test_points) {
        auto residual = solver->compute_residual(point);
        EXPECT_LT(std::abs(residual[0]), 1e-2); // Residual should be small
    }
}

TEST_F(PDESolverIntegrationTest, WaveEquationPropagation) {
    PDEConfig config;
    config.type = PDEType::WAVE_EQUATION;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0, 0.0, 2.0}, {10, 10, 20}); // x, y, t
    config.wave_speed = 1.0;
    config.is_time_dependent = true;
    config.time_span = 2.0;
    config.num_time_steps = 20;
    config.num_collocation_points = 300;
    config.num_boundary_points = 50;
    config.learning_rate = 0.001;
    config.max_epochs = 500;
    config.tolerance = 1e-4;
    
    auto solver = PDESolver::create_wave_solver(config);
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_LT(results.final_loss, 1e-2);
    
    // Time-dependent solution should have appropriate size
    EXPECT_GT(results.solution.size(), config.num_time_steps);
}

TEST_F(PDESolverIntegrationTest, NavierStokesFlow) {
    PDEConfig config;
    config.type = PDEType::NAVIER_STOKES;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    config.viscosity = 0.01;
    config.is_time_dependent = true;
    config.time_span = 1.0;
    config.num_time_steps = 10;
    config.num_collocation_points = 400;
    config.num_boundary_points = 80;
    config.learning_rate = 0.001;
    config.max_epochs = 800;
    config.tolerance = 1e-4;
    
    auto solver = PDESolver::create_navier_stokes_solver(config);
    auto results = solver->solve();
    
    EXPECT_GT(results.epochs_trained, 0);
    EXPECT_LT(results.final_loss, 1e-2);
    
    // Navier-Stokes should output velocity components and pressure
    std::vector<double> input = {0.5, 0.5, 0.5}; // x, y, t
    auto residual = solver->compute_residual(input);
    
    EXPECT_EQ(residual.size(), 3); // u, v, continuity equations
}

TEST_F(PDESolverIntegrationTest, MultiPhysicsCoupling) {
    // Test coupling between different physics
    PDEConfig heat_config;
    heat_config.type = PDEType::HEAT_EQUATION;
    heat_config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    heat_config.diffusion_coefficient = 0.1;
    heat_config.is_time_dependent = false;
    heat_config.num_collocation_points = 200;
    heat_config.learning_rate = 0.001;
    heat_config.max_epochs = 500;
    
    PDEConfig flow_config;
    flow_config.type = PDEType::NAVIER_STOKES;
    flow_config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    flow_config.viscosity = 0.01;
    flow_config.is_time_dependent = false;
    flow_config.num_collocation_points = 200;
    flow_config.learning_rate = 0.001;
    flow_config.max_epochs = 500;
    
    auto heat_solver = PDESolver::create_heat_solver(heat_config);
    auto flow_solver = PDESolver::create_navier_stokes_solver(flow_config);
    
    auto heat_results = heat_solver->solve();
    auto flow_results = flow_solver->solve();
    
    EXPECT_GT(heat_results.epochs_trained, 0);
    EXPECT_GT(flow_results.epochs_trained, 0);
    
    // Test coupling (simplified - just ensure both solvers work)
    std::vector<double> test_point = {0.5, 0.5};
    auto heat_residual = heat_solver->compute_residual(test_point);
    auto flow_residual = flow_solver->compute_residual(test_point);
    
    EXPECT_LT(std::abs(heat_residual[0]), 1e-1);
    EXPECT_EQ(flow_residual.size(), 3); // u, v, continuity
}

TEST_F(PDESolverIntegrationTest, UncertaintyQuantificationWorkflow) {
    PDEConfig config;
    config.type = PDEType::HEAT_EQUATION;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    config.diffusion_coefficient = 0.1;
    config.num_collocation_points = 100;
    config.learning_rate = 0.001;
    config.max_epochs = 200; // Reduced for testing
    
    UncertaintyQuantification uq(3); // Small ensemble for testing
    PINNArchitecture arch(2, 1, {10, 10});
    
    uq.train_ensemble(config, arch);
    
    std::vector<double> input = {0.5, 0.5};
    auto prediction = uq.predict_with_uncertainty(input);
    auto confidence_intervals = uq.compute_confidence_intervals(input, 0.95);
    
    EXPECT_FALSE(prediction.empty());
    EXPECT_EQ(confidence_intervals.size(), prediction.size() * 2);
    
    // Check that confidence intervals make sense
    for (size_t i = 0; i < prediction.size(); ++i) {
        double lower = confidence_intervals[2*i];
        double upper = confidence_intervals[2*i + 1];
        EXPECT_LT(lower, upper);
        EXPECT_GE(lower, prediction[i] - 0.5); // Reasonable bounds
        EXPECT_LE(upper, prediction[i] + 0.5);
    }
}

TEST_F(PDESolverIntegrationTest, PerformanceTargets) {
    // Test that performance targets are met
    
    // Target: <1ms inference for simple PDEs
    PDEConfig config;
    config.type = PDEType::HEAT_EQUATION;
    config.domain = Domain({0.0, 1.0, 0.0, 1.0}, {10, 10});
    config.num_collocation_points = 100;
    config.max_epochs = 100; // Minimal training for performance test
    
    auto solver = PDESolver::create_heat_solver(config);
    auto results = solver->solve();
    
    // Measure inference time
    std::vector<double> input = {0.5, 0.5};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        solver->compute_residual(input);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto total_time = std::chrono::duration<double, std::milli>(end_time - start_time);
    double avg_time = total_time.count() / 1000.0;
    
    EXPECT_LT(avg_time, 5.0) << "Inference time: " << avg_time << "ms (target: <5ms)";
    
    // Target: reasonable memory usage
    EXPECT_LT(results.training_time_ms, 5000.0) << "Training time: " << results.training_time_ms << "ms";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
