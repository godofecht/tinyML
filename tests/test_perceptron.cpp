#include <gtest/gtest.h>
#include "perceptron.h"
#include "logger.h"

// Loading and Sanity Checks

TEST(PerceptronTest, InitialWeightsSanityCheck)
{
    std::vector<unsigned> topology = {2, 2, 1};
    ML::Models::Perceptron perceptron(topology);

    // Check that weights are initialized to small non-zero values
    auto weights = perceptron.getWeights();
    for (auto weight : weights)
    {
        ASSERT_NE(weight, 0.0);           // Ensure weights are not zero
        ASSERT_LE(std::abs(weight), 1.0); // Ensure weights are small
    }
}

TEST(PerceptronTest, FeedForwardCheck)
{
    std::vector<unsigned> topology = {2, 2, 1};
    ML::Models::Perceptron perceptron(topology);

    std::vector<double> input = {0.0, 1.0};
    perceptron.feedForward (input);
    std::vector<double> output = perceptron.getResult();

    ASSERT_EQ(output.size(), 1);  // Ensure output size is correct
    ASSERT_LE(output[0], 1.0);    // Ensure output is within expected range
    ASSERT_GE(output[0], 0.0);
}

// TEST(PerceptronTest, TransferFunctionCheck)
// {
//     ML::Models::Perceptron perceptron({2, 2, 1});

//     // Test for a range of inputs
//     ASSERT_NEAR (perceptron.transferFunction(0.0), 0.5, 0.0001);
//     ASSERT_NEAR (perceptron.transferFunction(-10.0), 0.0, 0.0001);
//     ASSERT_NEAR (perceptron.transferFunction(10.0), 1.0, 0.0001);
// }

TEST(PerceptronTest, TrainingConvergence)
{
    std::vector<unsigned> topology = {3, 3, 1};
    ML::Models::Perceptron perceptron(topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> nandTrainingSet = {
        {{0, 0}, {1}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    // Train the network
    for (int i = 0; i < 10000; ++i)
    {
        for (const auto &[input, output] : nandTrainingSet)
        {
            perceptron.feedForward(input);
            perceptron.learnSupervised(output);
        }
    }

    // After training, evaluate the performance
    double tolerance = 0.01;

    auto output_00 = perceptron.process({0, 0})[0];
    auto output_01 = perceptron.process({0, 1})[0];
    auto output_10 = perceptron.process({1, 0})[0];
    auto output_11 = perceptron.process({1, 1})[0];

    ASSERT_NEAR(output_00, 1.0, tolerance);
    ASSERT_NEAR(output_01, 1.0, tolerance);
    ASSERT_NEAR(output_10, 1.0, tolerance);
    ASSERT_NEAR(output_11, 0.0, tolerance);
}

// TEST(PerceptronTest, GradientCheck)
// {
//     std::vector<unsigned> topology = {2, 2, 1};
//     ML::Models::Perceptron perceptron(topology);

//     // Assume perceptron has a method to compute gradients analytically
//     std::vector<double> input = {1.0, 0.0};
//     std::vector<double> output = {1.0};
//     perceptron.feedForward(input);
//     auto analyticalGradients = perceptron.backpropagate(output);

//     // Compute numerical gradients (finite difference approximation)
//     double epsilon = 1e-5;
//     auto numericalGradients = perceptron.computeNumericalGradients(input, output, epsilon);

//     // Compare analytical and numerical gradients
//     for (size_t i = 0; i < analyticalGradients.size(); ++i)
//     {
//         ASSERT_NEAR(analyticalGradients[i], numericalGradients[i], epsilon);
//     }
// }

// Macro to simplify result evaluation and logging
#define EVALUATE_AND_LOG(logger, case_num, expected, result)  \
    logger.logResult("Test Case " #case_num, {expected}, result); \
    ASSERT_NEAR(result[0], expected, 0.1);

// Test for AND function training
TEST (PerceptronTest, ANDTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {2, 2, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> andTrainingSet = {
        {{0, 0}, {0}},
        {{0, 1}, {0}},
        {{1, 0}, {0}},
        {{1, 1}, {1}}
    };

    for (int i = 0; i < 10000; ++i)
    {
        for (const auto &[input, output] : andTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }
    }

    EVALUATE_AND_LOG (logger, 1, 0.0, perceptron.process ({0, 0}));
    EVALUATE_AND_LOG (logger, 2, 0.0, perceptron.process ({0, 1}));
    EVALUATE_AND_LOG (logger, 3, 0.0, perceptron.process ({1, 0}));
    EVALUATE_AND_LOG (logger, 4, 1.0, perceptron.process ({1, 1}));
}

// Test for OR function training
TEST (PerceptronTest, ORTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {2, 2, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> orTrainingSet = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {1}}
    };

    for (int i = 0; i < 10000; ++i)
    {
        for (const auto &[input, output] : orTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }
    }

    EVALUATE_AND_LOG (logger, 1, 0.0, perceptron.process ({0, 0}));
    EVALUATE_AND_LOG (logger, 2, 1.0, perceptron.process ({0, 1}));
    EVALUATE_AND_LOG (logger, 3, 1.0, perceptron.process ({1, 0}));
    EVALUATE_AND_LOG (logger, 4, 1.0, perceptron.process ({1, 1}));
}

// Test for NAND function training
TEST (PerceptronTest, NANDTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {3, 4, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> nandTrainingSet = {
        {{0, 0}, {1}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    for (int i = 0; i < 10000; ++i)
    {
        for (const auto &[input, output] : nandTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }
    }

    EVALUATE_AND_LOG (logger, 1, 1.0, perceptron.process ({0, 0}));
    EVALUATE_AND_LOG (logger, 2, 1.0, perceptron.process ({0, 1}));
    EVALUATE_AND_LOG (logger, 3, 1.0, perceptron.process ({1, 0}));
    EVALUATE_AND_LOG (logger, 4, 0.0, perceptron.process ({1, 1}));
}

// Test for NOR function training
TEST (PerceptronTest, NORTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {2, 5, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> norTrainingSet = {
        {{0, 0}, {1}},
        {{0, 1}, {0}},
        {{1, 0}, {0}},
        {{1, 1}, {0}}
    };

    for (int i = 0; i < 10000; ++i)
    {
        for (const auto &[input, output] : norTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }
    }

    EVALUATE_AND_LOG (logger, 1, 1.0, perceptron.process ({0, 0}));
    EVALUATE_AND_LOG (logger, 2, 0.0, perceptron.process ({0, 1}));
    EVALUATE_AND_LOG (logger, 3, 0.0, perceptron.process ({1, 0}));
    EVALUATE_AND_LOG (logger, 4, 0.0, perceptron.process ({1, 1}));
}

// Test for XOR function training
TEST (PerceptronTest, XORTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {2, 2, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> xorTrainingSet = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    for (int i = 0; i < 10000; ++i)
    {
        for (const auto &[input, output] : xorTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }
    }

    EVALUATE_AND_LOG (logger, 1, 0.0, perceptron.process ({0, 0}));
    EVALUATE_AND_LOG (logger, 2, 1.0, perceptron.process ({0, 1}));
    EVALUATE_AND_LOG (logger, 3, 1.0, perceptron.process ({1, 0}));
    EVALUATE_AND_LOG (logger, 4, 0.0, perceptron.process ({1, 1}));
}

// Test for XNOR function training
TEST (PerceptronTest, XNORTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {3, 4, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> xnorTrainingSet = {
        {{0, 0}, {1}},
        {{0, 1}, {0}},
        {{1, 0}, {0}},
        {{1, 1}, {1}}
    };

    for (int i = 0; i < 10000; ++i)
    {
        for (const auto &[input, output] : xnorTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }
    }

    EVALUATE_AND_LOG (logger, 1, 1.0, perceptron.process ({0, 0}));
    EVALUATE_AND_LOG (logger, 2, 0.0, perceptron.process ({0, 1}));
    EVALUATE_AND_LOG (logger, 3, 0.0, perceptron.process ({1, 0}));
    EVALUATE_AND_LOG (logger, 4, 1.0, perceptron.process ({1, 1}));
}

// Test for Half-Adder Sum output
TEST (PerceptronTest, HalfAdderSumTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {2, 2, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> halfAdderSumTrainingSet = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}  // This is just XOR
    };

    for (int i = 0; i < 10000; ++i)
        for (const auto& [input, output] : halfAdderSumTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }

    EVALUATE_AND_LOG (logger, 1, 0.0, perceptron.process ({0, 0}));
    EVALUATE_AND_LOG (logger, 2, 1.0, perceptron.process ({0, 1}));
    EVALUATE_AND_LOG (logger, 3, 1.0, perceptron.process ({1, 0}));
    EVALUATE_AND_LOG (logger, 4, 0.0, perceptron.process ({1, 1}));
}

// Test for Half-Adder Carry output
TEST (PerceptronTest, HalfAdderCarryTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {3, 4, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> halfAdderCarryTrainingSet = {
        {{0, 0}, {0}},
        {{0, 1}, {0}},
        {{1, 0}, {0}},
        {{1, 1}, {1}}  // This is just AND
    };

    for (int i = 0; i < 10000; ++i)
        for (const auto& [input, output] : halfAdderCarryTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }

    EVALUATE_AND_LOG (logger, 1, 0.0, perceptron.process ({0, 0}));
    EVALUATE_AND_LOG (logger, 2, 0.0, perceptron.process ({0, 1}));
    EVALUATE_AND_LOG (logger, 3, 0.0, perceptron.process ({1, 0}));
    EVALUATE_AND_LOG (logger, 4, 1.0, perceptron.process ({1, 1}));
}

// Test for Full-Adder Sum output (A XOR B XOR C)
TEST (PerceptronTest, FullAdderSumTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {3, 3, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> fullAdderSumTrainingSet = {
        {{0, 0, 0}, {0}},
        {{0, 0, 1}, {1}},
        {{0, 1, 0}, {1}},
        {{0, 1, 1}, {0}},
        {{1, 0, 0}, {1}},
        {{1, 0, 1}, {0}},
        {{1, 1, 0}, {0}},
        {{1, 1, 1}, {1}}
    };

    for (int i = 0; i < 10000; ++i)
        for (const auto& [input, output] : fullAdderSumTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }

    EVALUATE_AND_LOG (logger, 1, 0.0, perceptron.process ({0, 0, 0}));
    EVALUATE_AND_LOG (logger, 2, 1.0, perceptron.process ({0, 0, 1}));
    EVALUATE_AND_LOG (logger, 3, 1.0, perceptron.process ({0, 1, 0}));
    EVALUATE_AND_LOG (logger, 4, 0.0, perceptron.process ({0, 1, 1}));
    EVALUATE_AND_LOG (logger, 5, 1.0, perceptron.process ({1, 0, 0}));
    EVALUATE_AND_LOG (logger, 6, 0.0, perceptron.process ({1, 0, 1}));
    EVALUATE_AND_LOG (logger, 7, 0.0, perceptron.process ({1, 1, 0}));
    EVALUATE_AND_LOG (logger, 8, 1.0, perceptron.process ({1, 1, 1}));
}

// Test for Full-Adder Carry output (A AND B) OR (C AND (A XOR B))
TEST (PerceptronTest, FullAdderCarryTrainingAndEvaluation)
{
    LoggerNS::Logger logger (LoggerNS::Logger::VerbosityLevel::INFO);

    std::vector<unsigned> topology = {3, 3, 1};
    ML::Models::Perceptron perceptron (topology);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> fullAdderCarryTrainingSet = {
        {{0, 0, 0}, {0}},
        {{0, 0, 1}, {0}},
        {{0, 1, 0}, {0}},
        {{0, 1, 1}, {1}},
        {{1, 0, 0}, {0}},
        {{1, 0, 1}, {1}},
        {{1, 1, 0}, {1}},
        {{1, 1, 1}, {1}}
    };

    for (int i = 0; i < 10000; ++i)
        for (const auto& [input, output] : fullAdderCarryTrainingSet)
        {
            perceptron.feedForward (input);
            perceptron.learnSupervised (output);
        }

    EVALUATE_AND_LOG (logger, 1, 0.0, perceptron.process ({0, 0, 0}));
    EVALUATE_AND_LOG (logger, 2, 0.0, perceptron.process ({0, 0, 1}));
    EVALUATE_AND_LOG (logger, 3, 0.0, perceptron.process ({0, 1, 0}));
    EVALUATE_AND_LOG (logger, 4, 1.0, perceptron.process ({0, 1, 1}));
    EVALUATE_AND_LOG (logger, 5, 0.0, perceptron.process ({1, 0, 0}));
    EVALUATE_AND_LOG (logger, 6, 1.0, perceptron.process ({1, 0, 1}));
    EVALUATE_AND_LOG (logger, 7, 1.0, perceptron.process ({1, 1, 0}));
    EVALUATE_AND_LOG (logger, 8, 1.0, perceptron.process ({1, 1, 1}));
}

// Main function for running all tests
int main (int argc, char **argv)
{
    ::testing::InitGoogleTest (&argc, argv);
    return RUN_ALL_TESTS();
}
