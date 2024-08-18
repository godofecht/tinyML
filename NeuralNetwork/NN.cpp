//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "NN.h"

namespace ML
{
    Neuron::Neuron(unsigned numOutputs, unsigned neuronIndex)
        : error(0.0f), gradient(0.0f), outputVal(0.0f), recentAverageError(0.0f), index(neuronIndex)
    {
        outputWeights.reserve(numOutputs);
        for (unsigned i = 0; i < numOutputs; ++i) {
            auto connectionPtr = std::make_unique<connection>();
            connectionPtr->weight = static_cast<double>(rand()) / RAND_MAX;
            outputWeights.push_back(std::move(connectionPtr));
        }
    }

    void Neuron::calcHiddenGradients(const Layer& nextLayer)
    {
        double dow = sumDOW(nextLayer);
        gradient = dow * Neuron::transferFunctionDerivative(outputVal);
    }

    void Neuron::calcOutputGradients(double targetVal)
    {
        double delta = targetVal - outputVal;
        gradient = delta * Neuron::transferFunctionDerivative(outputVal);
    }

    void Neuron::feedForward(Layer& prevLayer)
    {
        double sum = 0.0;
        for (const auto& neuron : prevLayer) {
            sum += neuron->getOutputVal() * neuron->outputWeights[index]->weight;
        }
        outputVal = Neuron::transferFunction(sum);
    }

    void Neuron::updateInputWeights(Layer& prevLayer)
    {
        for (auto& neuron : prevLayer) {
            double oldDeltaWeight = neuron->outputWeights[index]->deltaweight;
            double newDeltaWeight = eta * neuron->getOutputVal() * gradient + alpha * oldDeltaWeight;
            neuron->outputWeights[index]->deltaweight = newDeltaWeight;
            neuron->outputWeights[index]->weight += newDeltaWeight;
        }
    }

    double Neuron::randomWeight()
    {
        return static_cast<double>(rand()) / RAND_MAX;
    }

    double Neuron::sumDOW(const Layer& nextLayer) const
    {
        double sum = 0.0;
        for (unsigned i = 0; i < nextLayer.size() - 1; ++i) {
            sum += outputWeights[i]->weight * nextLayer[i]->gradient;
        }
        return sum;
    }

    double Neuron::transferFunctionDerivative(double x)
    {
        return 1.0 - x * x;
    }

    double Neuron::transferFunction(double x)
    {
        return tanh(x);
    }

    double Neuron::getOutputVal() const
    {
        return outputVal;
    }

    void Neuron::setOutputVal(double value)
    {
        outputVal = value;
    }

    int Neuron::getIndex() const
    {
        return index;
    }
}
