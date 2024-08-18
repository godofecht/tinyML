//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include <ctime> // For time()
#include <cassert>    // For assert()
#include "Network.h"

namespace ML
{
    Network::Network (const std::vector<unsigned>& topology)
    {
        srand (static_cast<unsigned int>(time (NULL)));
        unsigned numLayers = static_cast<unsigned>(topology.size());

        for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
        {
            layers.emplace_back();
            unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];

            // Create neurons for this layer, including a bias neuron
            for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
            {
                layers.back().push_back(std::make_unique<Neuron>(numOutputs, neuronNum));
            }

            // Set the bias neuron's output to 0.0
            layers.back().back()->setOutputVal(0.0);
        }
    }

    void Network::normalizeWeights (int connection_index)
    {
        double sum_weights_squared = 0.0;

        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                sum_weights_squared += neuron->getOutputWeights()[connection_index]->weight;
            }
        }

        double average = sum_weights_squared / 101.0;
        sum_weights_squared = 0.0;

        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                neuron->getOutputWeights()[connection_index]->weight -= average;
                sum_weights_squared += std::pow(neuron->getOutputWeights()[connection_index]->weight, 2);
            }
        }

        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                neuron->getOutputWeights()[connection_index]->weight /= std::sqrt(sum_weights_squared);
            }
        }
    }

    void Network::updateWeights()
    {
        for (std::size_t layerNum = 1; layerNum < layers.size(); ++layerNum)
        {
            Layer& prevLayer = layers[layerNum - 1];

            for (auto& neuron : prevLayer)
            {
                neuron->updateInputWeights(prevLayer);
            }
        }
    }

    void Network::backPropagate (const std::vector<double>& targetVals)
    {
        // Calculate overall net error (RMS of output neuron errors)
        Layer& outputLayer = layers.back();
        error = 0.0;

        for (std::size_t n = 0; n < outputLayer.size() - 1; ++n)
        {
            double delta = targetVals[n] - outputLayer[n]->getOutputVal();
            error += delta * delta;
        }

        error /= outputLayer.size() - 1;  // Average error squared
        error = std::sqrt(error);         // RMS

        // Implement a recent average measurement
        recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

        // Calculate output layer gradients
        for (std::size_t n = 0; n < outputLayer.size() - 1; ++n)
        {
            outputLayer[n]->calcOutputGradients(targetVals[n]);
        }

        // Calculate hidden layer gradients
        for (std::size_t layerNum = layers.size() - 2; layerNum > 0; --layerNum)
        {
            Layer& hiddenLayer = layers[layerNum];
            Layer& nextLayer = layers[layerNum + 1];

            for (auto& neuron : hiddenLayer)
            {
                neuron->calcHiddenGradients(nextLayer);
            }
        }

        // Update connection weights for all layers from output to first hidden layer
        for (std::size_t layerNum = layers.size() - 1; layerNum > 0; --layerNum)
        {
            Layer& layer = layers[layerNum];
            Layer& prevLayer = layers[layerNum - 1];

            for (std::size_t n = 0; n < layer.size() - 1; ++n)
            {
                layer[n]->updateInputWeights(prevLayer);
            }
        }
    }

    void Network::feedForward (std::vector<double> inputVals)
    {
        assert(inputVals.size() == layers[0].size() - 1);

        // Assign input values to input neurons
        for (std::size_t i = 0; i < inputVals.size(); ++i)
        {
            layers[0][i]->setOutputVal(inputVals[i]);
        }

        // Forward propagate
        for (std::size_t layerNum = 1; layerNum < layers.size(); ++layerNum)
        {
            Layer& prevLayer = layers[layerNum - 1];
            for (std::size_t n = 0; n < layers[layerNum].size() - 1; ++n)
            {
                layers[layerNum][n]->feedForward(prevLayer);
            }
        }
    }

    void Network::getResults (std::vector<double>& resultVals) const
    {
        resultVals.clear();
        for (const auto& neuron : layers.back())
        {
            if (&neuron != &layers.back().back()) // Ignore the bias neuron
            {
                resultVals.push_back(neuron->getOutputVal());
            }
        }
    }

    std::vector<double> Network::getWeights() const
    {
        std::vector<double> weights;

        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                for (const auto& weight : neuron->getOutputWeights())
                {
                    weights.push_back(weight->weight);
                }
            }
        }

        return weights;
    }

    void Network::putWeights (const std::vector<double>& weights)
    {
        std::size_t cWeight = 0;

        for (Layer& layer : layers)
        {
            for (auto& neuron : layer)
            {
                for (auto& weight : neuron->getOutputWeights())
                {
                    weight->weight = weights[cWeight++];
                }
            }
        }
    }
}
