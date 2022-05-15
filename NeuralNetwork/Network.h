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
		Network(const vector <unsigned>& topology);
		void backPropagate(const vector <double>& targetVals);
		void feedForward(vector <double>& inputVals);
		void getResults(vector <double>& resultVals);
		double getRecentAverageError(void) const { return recentAverageError; }

		vector<double> GetWeights() const;
		void PutWeights(vector<double>& weights);

		void UpdateWeights();

		void NormalizeWeights(int connection_index);


		//Change m_layers to layers
		vector<Layer>& GetLayers()
		{
			return m_layers;
		}

		//      bool NORMALIZE_WEIGHTS;

		vector <Layer> m_layers;
	private:
		double gradient = 0.0;
		double error = 0.0;
		double recentAverageError = 0.0;
		double recentAverageSmoothingFactor = 0.0;
	};
}

#endif
