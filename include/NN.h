// SPDX-License-Identifier: MIT
// Copyright (c) Abhishek Shivakumar

#ifndef NN_H
#define NN_H

#include <memory>
#include <vector>

namespace ML
{
    class Neuron;
    using Layer = std::vector<std::unique_ptr<Neuron>>;

    class Neuron
    {
    public:
        Neuron (unsigned numOutputs, unsigned neuronIndex);

        void calcHiddenGradients (const Layer& nextLayer);
        void calcOutputGradients (double targetVal);
        void feedForward (Layer& prevLayer);
        void updateInputWeights (Layer& prevLayer);

        static double transferFunction (double x);
        static double transferFunctionDerivative (double x);

        double getOutputVal() const;
        void setOutputVal (double value);
        int getIndex() const;

    private:
        struct connection
        {
            double weight = 0.0;
            double deltaweight = 0.0;
        };

        double randomWeight();
        double sumDOW (const Layer& nextLayer) const;

        double outputVal = 0.0;
        double gradient = 0.0;
        double error = 0.0;
        double recentAverageError = 0.0;
        unsigned index = 0;
        std::vector<std::unique_ptr<connection>> outputWeights;

        static constexpr double eta = 0.15;
        static constexpr double alpha = 0.5;

    public:
        const std::vector<std::unique_ptr<connection>>& getOutputWeights() const noexcept
        {
            return outputWeights;
        }

        std::vector<std::unique_ptr<connection>>& getOutputWeights() noexcept
        {
            return outputWeights;
        }
    };
}

#endif // NN_H
