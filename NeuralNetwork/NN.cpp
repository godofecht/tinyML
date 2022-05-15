//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "NN.h"

namespace ML
{
	Neuron::Neuron(unsigned numOutputs, unsigned newIndex)
	{
		error = 0.0f;
		gradient = 0.0f;
		outputVal = 0.0f;
		recentAverageError = 0.0f;

		for (unsigned c = 0; c < numOutputs; ++c) {
			outputWeights.push_back (std::make_unique<connection>());
			outputWeights[c]->weight = ((rand() / double(RAND_MAX)));
		}

		index = newIndex;
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
		for (unsigned n = 0; n < prevLayer.size(); ++n) {
			double outputVal = prevLayer[n]->getOutputVal();
			double weight = prevLayer[n]->outputWeights[index]->weight;
			sum += (outputVal * weight);
		}
		outputVal = Neuron::transferFunction(sum);
	}

	void Neuron::updateInputWeights(Layer& prevLayer)
	{
		for (unsigned n = 0; n < prevLayer.size(); ++n) {
			Neuron* neuron = prevLayer[n].get();
			double oldDeltaWeight = neuron->outputWeights[index]->deltaweight;

			double newDeltaWeight =
				// Individual input is magnified by the gradient and train rate:
				eta
				* neuron->getOutputVal()
				* gradient
				// Also adding momentum = a fraction of the previous delta weight;
				+ alpha
				* oldDeltaWeight;

			neuron->outputWeights[index]->deltaweight = newDeltaWeight;
			neuron->outputWeights[index]->weight += newDeltaWeight;
		}
	}

	double Neuron::randomWeight()
	{
		return (rand() / double(RAND_MAX));
	}
	double Neuron::sumDOW(const Layer& nextLayer) const
	{
		double sum = 0.0;

		// Sum our contributions of the errors at the nodes we feed.

		for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
			sum += outputWeights[n]->weight * nextLayer[n]->gradient;
		}

		return sum;
	}

	double Neuron::transferFunctionDerivative(double x)
	{
		return 1 - x * x;
	}
	double Neuron::transferFunction(double x)
	{
		return tanh(x);
	}

	double Neuron::getOutputVal()
	{
		return outputVal;
	}
	void Neuron::setOutputVal(double n)
	{
		outputVal = n;
	}

	int Neuron::getIndex()
	{
		return index;
	}
}
