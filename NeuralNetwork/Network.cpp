//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "Network.h"

namespace ML
{
	Network::Network(const vector <unsigned>& topology)
	{
		srand((unsigned int)time(NULL));

		unsigned numLayers = (unsigned)topology.size();
		for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
			m_layers.push_back(Layer());
			unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[size_t(layerNum) + 1];

			// We have a new layer, now fill it with neurons, and a bias neuron in each layer.
			for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
				m_layers.back().push_back(std::make_unique<Neuron>(numOutputs, neuronNum));

			}

			// Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
			m_layers.back().back()->setOutputVal(0.0);
		}


		//	if (NORMALIZE_WEIGHTS) // needed for stone and bray, eg.
		//        for (int i=0; i < (int) topology.back(); i++)
		//        {
			  //      NormalizeWeights(i);
		//        }
	}

	void Network::NormalizeWeights(int connection_index)
	{
		double sum_weights_squared = 0.0f;
		double checksum = 0.0f;
		for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {

			//      Layer& layer = m_layers[layerNum];
			Layer& prevLayer = m_layers[size_t(layerNum) - 1];

			for (unsigned n = 0; n < prevLayer.size(); n++) {
				Neuron* neuron = &(*prevLayer[n]);

				sum_weights_squared = sum_weights_squared + neuron->outputWeights[connection_index]->weight;// pow(neuron->m_outputWeights[0].weight,2);
			}
		}
		double average = sum_weights_squared / 101.0f;
		sum_weights_squared = 0.0f;
		for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {

			//      Layer& layer = m_layers[layerNum];
			Layer& prevLayer = m_layers[size_t(layerNum) - 1];




			for (unsigned n = 0; n < prevLayer.size(); n++) {
				Neuron* neuron = &(*prevLayer[n]);
				neuron->outputWeights[connection_index]->weight -= average;
				sum_weights_squared += pow(neuron->outputWeights[connection_index]->weight, 2);
				// pow(neuron->m_outputWeights[0].weight,2);
			}
		}

		for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
			//	Layer& layer = m_layers[layerNum];
			Layer& prevLayer = m_layers[size_t(layerNum) - 1];

			for (unsigned n = 0; n < prevLayer.size(); n++) {
				//	Neuron* neuron = &(prevLayer)[n].get();
				//    Layer& reflayer = (*prevLayer);
				//    Neuron* neuron = &(*reflayer[n].get());

				Neuron* neuron = &(*prevLayer[n]);

				//	double weight_squared = pow(neuron->m_outputWeights[connection_index]->weight,2);
				double newWeight = neuron->outputWeights[connection_index]->weight / sqrt(sum_weights_squared);
				neuron->outputWeights[connection_index]->weight = newWeight;
				//	cout<<newWeight<<endl;
			//			cout<<neuron->m_outputWeights[0].weight<<endl;
				checksum += pow(newWeight, 2);
			}

		}
		//	cout<<checksum<<endl;
		//	double checksumfactor = 1.0f/checksum;
		checksum = 0.0f;

		/*
		for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
			Layer& layer = m_layers[layerNum];
			Layer& prevLayer = m_layers[layerNum - 1];

			for (unsigned n = 0; n < prevLayer.size(); n++) {
				Neuron* neuron = &(prevLayer[n]);
				double weight_squared = pow(neuron->m_outputWeights[0].weight,2);
				double newWeight =(neuron->m_outputWeights[0].weight*sqrt(checksumfactor));// sqrt(1.0f-sum_weights_squared + weight_squared);
				neuron->m_outputWeights[0].weight = newWeight;
			//	cout<<newWeight<<endl;
		//			cout<<neuron->m_outputWeights[0].weight<<endl;
				checksum+= pow(newWeight,2);
			}
			double checksumfactor = 1.0f/ checksum;
			cout<<checksum<<endl;
		}
		*/
	}

	void Network::UpdateWeights()
	{
		for (unsigned int layerNum = 1; layerNum < (int)m_layers.size(); layerNum++) {

			//    Layer& layer = m_layers[layerNum];
			Layer& prevLayer = m_layers[size_t(layerNum) - 1];

			for (unsigned n = 0; n < prevLayer.size(); n++) {
				Neuron* neuron = &(*prevLayer[n]);

				neuron->updateInputWeights(prevLayer);
			}
		}
	}

	void Network::backPropagate(const vector <double>& targetVals)
	{
		// Calculate overall net error (RMS of output neuron errors)

		Layer& outputLayer = m_layers.back();
		error = 0.0;

		for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
			double delta = targetVals[n] - outputLayer[n]->getOutputVal();
			error += delta * delta;
		}
		error /= outputLayer.size() - 1; // get average error squared
		error = sqrt(error); // RMS

		// Implement a recent average measurement

		recentAverageError =
			(recentAverageError * recentAverageSmoothingFactor + error)
			/ (recentAverageSmoothingFactor + 1.0);

		// Calculate output layer gradients

		for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
			outputLayer[n]->calcOutputGradients(targetVals[n]);
		}

		// Calculate hidden layer gradients

		for (unsigned layerNum = (unsigned int)(m_layers.size() - 2); layerNum > 0; --layerNum) {
			Layer& hiddenLayer = m_layers[layerNum];
			Layer& nextLayer = m_layers[size_t(layerNum) + 1];

			for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
				hiddenLayer[n]->calcHiddenGradients(nextLayer);
			}
		}

		// For all layers from outputs to first hidden layer,
		// update connection weights

		for (unsigned layerNum = (int)m_layers.size() - 1; layerNum > 0; --layerNum) {
			Layer& layer = m_layers[layerNum];
			Layer& prevLayer = m_layers[size_t(layerNum) - 1];

			for (unsigned n = 0; n < layer.size() - 1; ++n) {
				layer[n]->updateInputWeights(prevLayer);
			}
		}
	}



	void Network::feedForward(vector <double>& inputVals)
	{
		assert(inputVals.size() == m_layers[0].size() - 1);
		// Assign (latch) the input values into the input neurons
		for (unsigned i = 0; i < inputVals.size(); ++i) {
			m_layers[0][i]->setOutputVal(inputVals[i]);
		}
		// forward propagate
		for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
			Layer& prevLayer = m_layers[size_t(layerNum) - 1];
			for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
				m_layers[layerNum][n]->feedForward(prevLayer);
			}
		}
	}

	void Network::getResults(vector <double>& resultVals)
	{
		resultVals.clear();
		for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
			resultVals.push_back(m_layers.back()[n]->getOutputVal());
		}
	}

	vector<double> Network::GetWeights() const
	{
		//this will hold the weights
		vector<double> weights;
		//for each layer
		for (int i = 0; i < m_layers.size() - 1; ++i)
		{
			//for each neuron
			for (int j = 0; j < m_layers[i].size(); ++j)
			{
				//for each weight
				for (int k = 0; k < m_layers[i][j]->outputWeights.size(); ++k)
				{
					weights.push_back(m_layers[i][j]->outputWeights[k]->weight);
				}
			}
		}
		return weights;
	}

	void Network::PutWeights(vector<double>& weights)
	{
		int cWeight = 0;
		//for each layer
		for (int i = 0; i < m_layers.size() - 1; ++i)
		{
			//for each neuron
			for (int j = 0; j < m_layers[i].size(); ++j)
			{
				//for each weight
				for (int k = 0; k < m_layers[i][j]->outputWeights.size(); ++k)
				{
					m_layers[i][j]->outputWeights[k]->weight = weights[cWeight++];
				}
			}
		}
	}
}