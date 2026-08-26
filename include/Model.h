// SPDX-License-Identifier: MIT
// Copyright (c) Abhishek Shivakumar

#ifndef MODEL_H
#define MODEL_H

#include "Network.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ML
{
    class Model
    {
    public:
        explicit Model (const std::vector<unsigned>& topology)
            : thisNetwork (topology), topology (topology)
        {
        }

        const std::vector<unsigned>& getTopology() const noexcept
        {
            return topology;
        }

        std::vector<double> getWeights() const
        {
            return thisNetwork.getWeights();
        }

        std::vector<std::vector<double>> getActivations() const
        {
            std::vector<std::vector<double>> activations;
            activations.reserve (thisNetwork.GetLayers().size());

            for (const auto& layer : thisNetwork.GetLayers())
            {
                std::vector<double> layerActivations;
                layerActivations.reserve (layer.size());

                for (const auto& neuron : layer)
                {
                    layerActivations.push_back (neuron->getOutputVal());
                }

                activations.push_back (std::move (layerActivations));
            }

            return activations;
        }

        double getRecentAverageError() const noexcept
        {
            return thisNetwork.getRecentAverageError();
        }

        void setTopology (const std::vector<unsigned>& newTopology)
        {
            Network replacement (newTopology);
            thisNetwork = std::move (replacement);
            topology = newTopology;
        }

        void backPropagate (const std::vector<double>& targetVals)
        {
            thisNetwork.backPropagate (targetVals);
        }

        Network* getNetwork() noexcept
        {
            return &thisNetwork;
        }

        const Network* getNetwork() const noexcept
        {
            return &thisNetwork;
        }

        void feedForward (const std::vector<double>& inputs)
        {
            thisNetwork.feedForward (inputs);
        }

        std::vector<double> getResult() const
        {
            std::vector<double> resultVals;
            thisNetwork.getResults (resultVals);
            return resultVals;
        }

        void setWeights (const std::vector<double>& newWeights)
        {
            thisNetwork.putWeights (newWeights);
        }

        void displayTopology() const
        {
            std::cout << "Network Topology:\n";
            for (const unsigned layerSize : topology)
            {
                std::cout << layerSize << " neurons\n";
            }
        }

        void updateWeights()
        {
            thisNetwork.updateWeights();
        }

        void displayWeights() const
        {
            std::cout << "Network Weights:\n";
            for (const double weight : getWeights())
            {
                std::cout << weight << ' ';
            }
            std::cout << '\n';
        }

        void saveWeightsToFile (const std::string& filename) const
        {
            std::ofstream outFile (filename, std::ios::trunc);
            if (!outFile)
            {
                throw std::runtime_error ("Unable to open TinyML weight file for writing: " + filename);
            }

            outFile << std::setprecision (std::numeric_limits<double>::max_digits10);
            for (const double weight : getWeights())
            {
                outFile << weight << '\n';
            }

            if (!outFile)
            {
                throw std::runtime_error ("Failed while writing TinyML weight file: " + filename);
            }
        }

        void loadWeightsFromFile (const std::string& filename)
        {
            std::ifstream inFile (filename);
            if (!inFile)
            {
                throw std::runtime_error ("Unable to open TinyML weight file for reading: " + filename);
            }

            std::vector<double> newWeights;
            double weight = 0.0;
            while (inFile >> weight)
            {
                newWeights.push_back (weight);
            }

            if (!inFile.eof())
            {
                throw std::runtime_error ("TinyML weight file contains invalid data: " + filename);
            }

            setWeights (newWeights);
        }

    private:
        Network thisNetwork;
        std::vector<unsigned> topology;
    };
}

#endif // MODEL_H
