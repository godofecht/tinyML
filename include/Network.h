//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef NETWORK_H
#define NETWORK_H

#include "NN.h"
#include <vector>

namespace ML
{
	class Network
	{
	public:
		Network (const std::vector <unsigned>& topology);
		void backPropagate (const std::vector <double>& targetVals);
		void feedForward (std::vector <double> inputVals); //TODO: make const
		void getResults (std::vector <double>& resultVals) const;
		void putWeights (const std::vector<double>& weights);
		void updateWeights();
		void normalizeWeights (int connection_index);
		std::vector<Layer>& GetLayers() { return layers; }
		double getRecentAverageError (void) const { return recentAverageError; }

		std::vector<double> getWeights() const;

		std::vector <Layer> layers;
	private:
		double gradient = 0.0;
		double error = 0.0;
		double recentAverageError = 0.0;
		double recentAverageSmoothingFactor = 0.0;
	};
}

#endif
