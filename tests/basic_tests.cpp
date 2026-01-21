//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Basic Tests for TinyML Foundation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <gtest/gtest.h>
#include "Model.h"
#include "Perceptron.h"
#include "XSIMDOperations.h"

class BasicTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data
    }
};

// Test basic perceptron functionality
TEST_F(BasicTests, PerceptronBasicOperations) {
    std::cout << "\n=== Basic Perceptron Test ===\n";
    
    // Create a simple perceptron
    std::vector<unsigned> topology = {2, 2, 1};
    ML::Models::Perceptron perceptron(topology);
    
    // Test feedforward
    std::vector<double> input = {1.0, 0.5};
    perceptron.feedForward(input);
    
    // Test that it doesn't crash
    EXPECT_NO_THROW(perceptron.feedForward(input));
    
    std::cout << "Basic perceptron operations: PASS\n";
}

// Test XSIMD functionality
TEST_F(BasicTests, XSIMDBasicOperations) {
    std::cout << "\n=== XSIMD Basic Operations Test ===\n";
    std::cout << "SIMD batch size: " << ML::XSIMD::XSIMDVector::simd_batch_size() << std::endl;
    std::cout << "Has SIMD support: " << (ML::XSIMD::XSIMDVector::has_simd_support() ? "Yes" : "No") << std::endl;
    
    // Test vector operations
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {0.5f, 1.5f, 2.5f, 3.5f};
    std::vector<float> result(4);
    
    ML::XSIMD::VectorOps::vector_add_vector(data1.data(), data2.data(), result.data(), 4);
    
    // Verify results
    for (size_t i = 0; i < 4; ++i) {
        float expected = data1[i] + data2[i];
        EXPECT_NEAR(result[i], expected, 1e-5f);
    }
    
    std::cout << "XSIMD vector operations: PASS\n";
}

// Test matrix-vector multiplication
TEST_F(BasicTests, MatrixVectorMultiplication) {
    std::cout << "\n=== Matrix-Vector Multiplication Test ===\n";
    
    // Simple 2x3 matrix times 3x1 vector
    std::vector<float> matrix = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // 2x3
    std::vector<float> vector = {1.0f, 0.5f, -0.5f}; // 3x1
    std::vector<float> result(2); // 2x1
    
    ML::XSIMD::VectorOps::matrix_vector_multiply(
        matrix.data(), vector.data(), result.data(), 2, 3
    );
    
    // Expected: [1*1 + 2*0.5 + 3*(-0.5), 4*1 + 5*0.5 + 6*(-0.5)]
    // = [1 + 1 + (-1.5), 4 + 2.5 + (-3)] = [0.5, 3.5]
    EXPECT_NEAR(result[0], 0.5f, 1e-5f);
    EXPECT_NEAR(result[1], 3.5f, 1e-5f);
    
    std::cout << "Matrix-vector multiplication: PASS\n";
}

// Test activation functions
TEST_F(BasicTests, ActivationFunctions) {
    std::cout << "\n=== Activation Functions Test ===\n";
    
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output(5);
    
    // Test tanh
    ML::XSIMD::VectorOps::tanh_batch(input.data(), output.data(), 5);
    
    // Verify tanh range [-1, 1]
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_GE(output[i], -1.0f);
        EXPECT_LE(output[i], 1.0f);
    }
    
    // Test ReLU
    ML::XSIMD::VectorOps::relu_batch(input.data(), output.data(), 5);
    
    // Verify ReLU range [0, ∞)
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_GE(output[i], 0.0f);
    }
    
    std::cout << "Activation functions: PASS\n";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
