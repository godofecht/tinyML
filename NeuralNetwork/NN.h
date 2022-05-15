//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/


#ifndef NN_H
#define NN_H

#include<cmath>
#include<cassert>
#include<iostream>
#include<vector>
#include<time.h>
#include<stdlib.h>

using namespace std;

namespace ML
{
	class Neuron;
	typedef vector<std::unique_ptr<Neuron>> Layer;
	class connection
	{
	public:
		double weight;
		double deltaweight;

		void setDW(double dw)
		{
			deltaweight = dw;
		}
	};
	class Neuron
	{
	public:
		Neuron(unsigned numOutputs, unsigned myIndex);
		void feedForward(Layer& prevLayer);
		double getOutputVal();
		void setOutputVal(double n);
		void calcHiddenGradients(const Layer& nextLayer);
		void calcOutputGradients(double targetVal);
		void updateInputWeights(Layer& prevLayer);
		double eta = 0.15;
		double alpha = 0.5;
		vector <std::unique_ptr<connection>> outputWeights;

		double error;
		double recentAverageError;
		double recentAverageSmoothingFactor = 100.0;

		int getIndex();
	private:
		double gradient;
		double outputVal;
		static double randomWeight();
		unsigned index;
		double sumDOW(const Layer& nextLayer) const;
		static double transferFunctionDerivative(double x);
		static double transferFunction(double x);
	};
}

#endif
