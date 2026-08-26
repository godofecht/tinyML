// SPDX-License-Identifier: MIT
// Copyright (c) Abhishek Shivakumar

#include "Network.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ML
{
    Network::Network (const std::vector<unsigned>& topology)
    {
        if (topology.size() < 2)
        {
            throw std::invalid_argument ("TinyML network topology requires at least input and output layers");
        }

        for (const unsigned width : topology)
        {
            if (width == 0)
            {
                throw std::invalid_argument ("TinyML network layers must contain at least one neuron");
            }
        }

        const auto numLayers = static_cast<unsigned> (topology.size());

        for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
        {
            layers.emplace_back();
            const unsigned numOutputs = layerNum + 1 == numLayers ? 0 : topology[layerNum + 1];

            for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
            {
                layers.back().push_back (std::make_unique<Neuron> (numOutputs, neuronNum));
            }

            // Every non-output layer uses its final neuron as a conventional bias input.
            layers.back().back()->setOutputVal (1.0);
        }
    }

    void Network::normalizeWeights (int connectionIndex)
    {
        if (connectionIndex < 0)
        {
            throw std::invalid_argument ("TinyML connection index cannot be negative");
        }

        const auto index = static_cast<std::size_t> (connectionIndex);
        double sum = 0.0;
        std::size_t count = 0;

        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                const auto& weights = neuron->getOutputWeights();
                if (index < weights.size())
                {
                    sum += weights[index]->weight;
                    ++count;
                }
            }
        }

        if (count == 0)
        {
            throw std::out_of_range ("TinyML connection index does not exist in this network");
        }

        const double mean = sum / static_cast<double> (count);
        double squaredNorm = 0.0;

        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                auto& weights = neuron->getOutputWeights();
                if (index < weights.size())
                {
                    weights[index]->weight -= mean;
                    squaredNorm += weights[index]->weight * weights[index]->weight;
                }
            }
        }

        if (squaredNorm <= std::numeric_limits<double>::epsilon())
        {
            return;
        }

        const double norm = std::sqrt (squaredNorm);
        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                auto& weights = neuron->getOutputWeights();
                if (index < weights.size())
                {
                    weights[index]->weight /= norm;
                }
            }
        }
    }

    void Network::updateWeights()
    {
        for (std::size_t layerNum = 1; layerNum < layers.size(); ++layerNum)
        {
            Layer& layer = layers[layerNum];
            Layer& prevLayer = layers[layerNum - 1];

            for (std::size_t neuronNum = 0; neuronNum + 1 < layer.size(); ++neuronNum)
            {
                layer[neuronNum]->updateInputWeights (prevLayer);
            }
        }
    }

    void Network::backPropagate (const std::vector<double>& targetVals)
    {
        Layer& outputLayer = layers.back();
        const std::size_t outputCount = outputLayer.size() - 1;

        if (targetVals.size() != outputCount)
        {
            throw std::invalid_argument ("TinyML target vector size does not match the output layer");
        }

        error = 0.0;
        for (std::size_t n = 0; n < outputCount; ++n)
        {
            const double delta = targetVals[n] - outputLayer[n]->getOutputVal();
            error += delta * delta;
        }

        error = std::sqrt (error / static_cast<double> (outputCount));
        recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error)
                           / (recentAverageSmoothingFactor + 1.0);

        for (std::size_t n = 0; n < outputCount; ++n)
        {
            outputLayer[n]->calcOutputGradients (targetVals[n]);
        }

        for (std::size_t layerNum = layers.size() - 2; layerNum > 0; --layerNum)
        {
            Layer& hiddenLayer = layers[layerNum];
            Layer& nextLayer = layers[layerNum + 1];

            for (std::size_t n = 0; n + 1 < hiddenLayer.size(); ++n)
            {
                hiddenLayer[n]->calcHiddenGradients (nextLayer);
            }
        }

        for (std::size_t layerNum = layers.size() - 1; layerNum > 0; --layerNum)
        {
            Layer& layer = layers[layerNum];
            Layer& prevLayer = layers[layerNum - 1];

            for (std::size_t n = 0; n + 1 < layer.size(); ++n)
            {
                layer[n]->updateInputWeights (prevLayer);
            }
        }
    }

    void Network::feedForward (const std::vector<double>& inputVals)
    {
        const std::size_t expectedInputs = layers.front().size() - 1;
        if (inputVals.size() != expectedInputs)
        {
            throw std::invalid_argument ("TinyML input vector size does not match the input layer");
        }

        for (std::size_t i = 0; i < inputVals.size(); ++i)
        {
            layers.front()[i]->setOutputVal (inputVals[i]);
        }

        for (std::size_t layerNum = 1; layerNum < layers.size(); ++layerNum)
        {
            Layer& prevLayer = layers[layerNum - 1];
            Layer& layer = layers[layerNum];

            for (std::size_t n = 0; n + 1 < layer.size(); ++n)
            {
                layer[n]->feedForward (prevLayer);
            }
        }
    }

    void Network::getResults (std::vector<double>& resultVals) const
    {
        resultVals.clear();
        const Layer& outputLayer = layers.back();
        resultVals.reserve (outputLayer.size() - 1);

        for (std::size_t n = 0; n + 1 < outputLayer.size(); ++n)
        {
            resultVals.push_back (outputLayer[n]->getOutputVal());
        }
    }

    std::vector<double> Network::getWeights() const
    {
        std::vector<double> weights;

        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                for (const auto& connection : neuron->getOutputWeights())
                {
                    weights.push_back (connection->weight);
                }
            }
        }

        return weights;
    }

    void Network::putWeights (const std::vector<double>& weights)
    {
        std::size_t expectedCount = 0;
        for (const Layer& layer : layers)
        {
            for (const auto& neuron : layer)
            {
                expectedCount += neuron->getOutputWeights().size();
            }
        }

        if (weights.size() != expectedCount)
        {
            throw std::invalid_argument ("TinyML weight vector size does not match the network topology");
        }

        std::size_t currentWeight = 0;
        for (Layer& layer : layers)
        {
            for (auto& neuron : layer)
            {
                for (auto& connection : neuron->getOutputWeights())
                {
                    connection->weight = weights[currentWeight++];
                }
            }
        }
    }
}
