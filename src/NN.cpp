// SPDX-License-Identifier: MIT
// Copyright (c) Abhishek Shivakumar

#include "NN.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace ML
{
    Neuron::Neuron (unsigned numOutputs, unsigned neuronIndex)
        : index (neuronIndex)
    {
        outputWeights.reserve (numOutputs);

        for (unsigned i = 0; i < numOutputs; ++i)
        {
            auto connectionPtr = std::make_unique<connection>();
            connectionPtr->weight = randomWeight();
            outputWeights.push_back (std::move (connectionPtr));
        }
    }

    void Neuron::calcHiddenGradients (const Layer& nextLayer)
    {
        gradient = sumDOW (nextLayer) * transferFunctionDerivative (outputVal);
    }

    void Neuron::calcOutputGradients (double targetVal)
    {
        const double delta = targetVal - outputVal;
        gradient = delta * transferFunctionDerivative (outputVal);
    }

    void Neuron::feedForward (Layer& prevLayer)
    {
        double sum = 0.0;

        for (const auto& neuron : prevLayer)
        {
            const auto& weights = neuron->getOutputWeights();
            if (index >= weights.size())
            {
                throw std::out_of_range ("TinyML connection index is outside the previous layer");
            }

            sum += neuron->getOutputVal() * weights[index]->weight;
        }

        outputVal = transferFunction (sum);
    }

    double Neuron::getOutputVal() const
    {
        return outputVal;
    }

    void Neuron::updateInputWeights (Layer& prevLayer)
    {
        for (auto& neuron : prevLayer)
        {
            auto& weights = neuron->getOutputWeights();
            if (index >= weights.size())
            {
                throw std::out_of_range ("TinyML connection index is outside the previous layer");
            }

            auto& connection = *weights[index];
            const double newDeltaWeight = eta * neuron->getOutputVal() * gradient
                                        + alpha * connection.deltaweight;

            connection.deltaweight = newDeltaWeight;
            connection.weight += newDeltaWeight;
        }
    }

    double Neuron::randomWeight()
    {
        static thread_local std::mt19937 generator (std::random_device{}());
        static thread_local std::uniform_real_distribution<double> distribution (0.0, 1.0);
        return distribution (generator);
    }

    double Neuron::sumDOW (const Layer& nextLayer) const
    {
        if (nextLayer.size() <= 1)
        {
            return 0.0;
        }

        const std::size_t connectionCount = std::min (outputWeights.size(), nextLayer.size() - 1);
        double sum = 0.0;

        for (std::size_t i = 0; i < connectionCount; ++i)
        {
            sum += outputWeights[i]->weight * nextLayer[i]->gradient;
        }

        return sum;
    }

    double Neuron::transferFunctionDerivative (double x)
    {
        return 1.0 - x * x;
    }

    double Neuron::transferFunction (double x)
    {
        return std::tanh (x);
    }

    void Neuron::setOutputVal (double value)
    {
        outputVal = value;
    }

    int Neuron::getIndex() const
    {
        return static_cast<int> (index);
    }
}
