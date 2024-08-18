//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef MODEL_H
#define MODEL_H

#include "Network.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

namespace ML
{
    /**
     * @brief The Model class encapsulates a neural network, providing methods to train,
     * update, and query the network. It also includes utilities for saving and loading weights.
     * 
     * Usage:
     * 
     * 1. Initialize the Model with a topology (vector<unsigned>) defining the number of neurons in each layer.
     * 2. Use the `FeedForward` method to pass inputs through the network.
     * 3. Use the `GetResult` method to obtain the network's output.
     * 4. Use the `BackPropagate` method to train the network with target values.
     * 5. Save and load weights using `SaveWeightsToFile` and `LoadWeightsFromFile`.
     * 
     * Example:
     * 
     * ```
     * std::vector<unsigned> topology = {3, 2, 1}; // 3 neurons in the input layer, 2 in the hidden, 1 in the output
     * ML::Model model(topology);
     * 
     * std::vector<double> inputVals = {1.0, 0.5, -1.5};
     * model.FeedForward(inputVals);
     * 
     * std::vector<double> results = model.GetResult();
     * model.BackPropagate({0.8}); // Assuming the target value is 0.8 for the output
     * 
     * model.SaveWeightsToFile("weights.txt");
     * model.LoadWeightsFromFile("weights.txt");
     * ```
     */
    class Model
    {
    private:
        Network thisNetwork;             // The neural network associated with this model
        std::vector<unsigned> topology;  // The topology of the network (number of neurons per layer)
        std::vector<double> weights;     // Cache for the weights of the network

    public:
        /**
         * @brief Constructor that initializes the model with the given topology.
         * 
         * @param tp The topology defining the number of neurons in each layer.
         */
        Model(const std::vector<unsigned>& tp) 
            : thisNetwork(tp), topology(tp)
        {
        }

        /**
         * @brief Set a new topology for the model.
         * 
         * This function allows you to change the network structure after initialization.
         * However, if you change the topology, you'll need to reinitialize the network with the new topology.
         * 
         * @param tp A vector representing the new topology.
         */
        void SetTopology(const std::vector<unsigned>& tp)
        {
            topology = tp;
            thisNetwork = Network(tp);  // Reinitialize the network with the new topology
        }

        /**
         * @brief Perform backpropagation on the network to update the weights based on target values.
         * 
         * @param targetVals The expected output values used for training.
         */
        void BackPropagate(const std::vector<double>& targetVals)
        {
            thisNetwork.backPropagate(targetVals);
        }

        /**
         * @brief Get a pointer to the internal network.
         * 
         * This allows you to directly access the Network class for advanced operations if needed.
         * 
         * @return Network* A pointer to the internal Network object.
         */
        Network* GetNetwork()
        {
            return &thisNetwork;
        }

        /**
         * @brief Get the current weights of the network.
         * 
         * This function retrieves the current weights of the network, which are cached for later use.
         * 
         * @return std::vector<double> A vector containing the current weights of the network.
         */
        std::vector<double> GetWeights()
        {
            weights = thisNetwork.GetWeights();  // Update the cached weights
            return weights;
        }

        /**
         * @brief Perform forward propagation through the network with the given inputs.
         * 
         * This method processes the input values through the network to calculate the output.
         * 
         * @param inputs A vector of input values corresponding to the input layer of the network.
         */
        void FeedForward(const std::vector<double>& inputs)
        {
            assert(inputs.size() == topology.front());  // Ensure the input size matches the network input layer
            thisNetwork.feedForward(inputs);
        }

        /**
         * @brief Get the results (output values) from the network.
         * 
         * After calling `FeedForward`, use this method to retrieve the calculated outputs.
         * 
         * @return std::vector<double> A vector containing the output values from the network.
         */
        std::vector<double> GetResult()
        {
            std::vector<double> resultVals;
            thisNetwork.getResults(resultVals);
            return resultVals;
        }

        /**
         * @brief Set new weights for the network.
         * 
         * This function allows you to manually set the weights of the network.
         * 
         * @param newWeights A vector containing the new weights to be applied to the network.
         */
        void SetWeights(const std::vector<double>& newWeights)
        {
            thisNetwork.PutWeights(newWeights);
        }

        /**
         * @brief Display the topology of the network.
         * 
         * This function prints the number of neurons in each layer of the network to the console.
         */
        void DisplayTopology() const
        {
            std::cout << "Network Topology:\n";
            for (unsigned layerSize : topology)
            {
                std::cout << layerSize << " neurons\n";
            }
        }

        /**
         * @brief Update the weights of the network.
         * 
         * This function applies updates to the weights based on the training (backpropagation) process.
         */
        void UpdateWeights()
        {
            thisNetwork.UpdateWeights();
        }

        /**
         * @brief Display the weights of the network.
         * 
         * This function prints the current weights of the network to the console.
         */
        void DisplayWeights() const
        {
            std::cout << "Network Weights:\n";
            const std::vector<double>& currentWeights = const_cast<Model*>(this)->GetWeights();
            for (double weight : currentWeights)
            {
                std::cout << weight << " ";
            }
            std::cout << "\n";
        }

        /**
         * @brief Save the network weights to a file.
         * 
         * This function writes the current weights of the network to a file for later retrieval.
         * 
         * @param filename The name of the file where weights will be saved.
         */
        void SaveWeightsToFile(const std::string& filename) const
        {
            std::ofstream outFile(filename);
            if (!outFile)
            {
                std::cerr << "Error: Unable to open file for saving weights\n";
                return;
            }

            const std::vector<double>& currentWeights = const_cast<Model*>(this)->GetWeights();
            for (double weight : currentWeights)
            {
                outFile << weight << "\n";
            }
            outFile.close();
        }

        /**
         * @brief Load the network weights from a file.
         * 
         * This function reads weights from a file and applies them to the network.
         * 
         * @param filename The name of the file from which to load weights.
         */
        void LoadWeightsFromFile(const std::string& filename)
        {
            std::ifstream inFile(filename);
            if (!inFile)
            {
                std::cerr << "Error: Unable to open file for loading weights\n";
                return;
            }

            std::vector<double> newWeights;
            double weight;
            while (inFile >> weight)
            {
                newWeights.push_back(weight);
            }

            if (!newWeights.empty())
            {
                SetWeights(newWeights);
            }

            inFile.close();
        }
    };
}

#endif // MODEL_H
