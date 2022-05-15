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


//This class has been written in accordance with RAII
//principles.

//However, there are still some things that are left
//to do.

//1. Clean up unused variables.
//2. Clean up unnecessary memory usage.


//Steps to use:

//1. Initialize a topology object of type std::vector<int> topology
//2. Initialize an object of type Model with "Model m (topology)
//3. To train, follow steps 3.1 - 3.3:
//  3.1. call feedforward() on your input data
//  3.2. call getResult() to obtain output values
//  3.3. call backpropagate() on your expected values
//4. To process, simply call feedforward()

namespace ML
{
	class Model
	{
	public:
		Network thisNetwork;
		vector<unsigned> topology;
		vector<double> weights;
	public:

		Model(vector<unsigned> tp) : thisNetwork(tp)
		{
		}

		/*
		void SetTopology (int* tp)
		{
			int arrSize = (unsigned int) (* ( & tp + 1)) - (unsigned int) tp;
			vector<unsigned> topo;
			for (int i = 0; i < arrSize; ++i)
			{
				topo.push_back (tp[i]);
			}
			topology = topo;
		}
		*/

		void SetTopology(vector<unsigned> tp)
		{
			topology = tp;
		}

		void BackPropagate(vector<double>& targetVals)
		{
			thisNetwork.backPropagate(targetVals);
		}
		Network* getNetwork()
		{
			return &thisNetwork;
		}
		vector<double> GetWeights()
		{
			weights = thisNetwork.GetWeights();
			return weights;
		}
		void feedforward(vector<double> inputs)
		{
			thisNetwork.feedForward(inputs);
		}
		vector<double> GetResult()
		{
			vector<double> resultVals;
			thisNetwork.getResults(resultVals);
			return resultVals;
		}
		void SetWeights(vector<double> newWeights)
		{
			thisNetwork.PutWeights(newWeights);
		}

		void DisplayTopology()
		{
			for (int i = 0; i < topology.size(); i++)
			{
				std::cout << topology[i] << "\n";
			}
		}

		void UpdateWeights()
		{
			thisNetwork.UpdateWeights();
		}
	};
}

#endif
