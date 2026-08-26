// SPDX-License-Identifier: MIT
// Copyright (c) Abhishek Shivakumar

#ifndef NETWORK_H
#define NETWORK_H

#include "NN.h"

#include <vector>

namespace ML
{
    class Network
    {
    public:
        explicit Network (const std::vector<unsigned>& topology);

        void backPropagate (const std::vector<double>& targetVals);
        void feedForward (const std::vector<double>& inputVals);
        void getResults (std::vector<double>& resultVals) const;
        void putWeights (const std::vector<double>& weights);
        void updateWeights();
        void normalizeWeights (int connectionIndex);

        std::vector<Layer>& GetLayers() noexcept
        {
            return layers;
        }

        const std::vector<Layer>& GetLayers() const noexcept
        {
            return layers;
        }

        double getRecentAverageError() const noexcept
        {
            return recentAverageError;
        }

        std::vector<double> getWeights() const;

        // Kept public for source compatibility with the original API.
        std::vector<Layer> layers;

    private:
        double gradient = 0.0;
        double error = 0.0;
        double recentAverageError = 0.0;
        double recentAverageSmoothingFactor = 0.0;
    };
}

#endif // NETWORK_H
