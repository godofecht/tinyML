//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef NN_H
#define NN_H

#include <vector>
#include <memory>
#include <cmath>

namespace ML
{
    class Neuron;
    using Layer = std::vector<std::unique_ptr<Neuron>>;

    class Neuron
    {
    public:
        Neuron(unsigned numOutputs, unsigned neuronIndex);

        void calcHiddenGradients(const Layer& nextLayer);
        void calcOutputGradients(double targetVal);
        void feedForward(Layer& prevLayer);
        void updateInputWeights(Layer& prevLayer);

        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);

        double getOutputVal() const;
        void setOutputVal(double value);
        int getIndex() const;

    private:
        double randomWeight();
        double sumDOW(const Layer& nextLayer) const;

        double outputVal;
        double gradient;
        double error;
        double recentAverageError;
        unsigned index;

        struct connection
        {
            double weight;
            double deltaweight;
        };

        std::vector<std::unique_ptr<connection>> outputWeights;

        static constexpr double eta = 0.15;   // learning rate
        static constexpr double alpha = 0.5;  // momentum

	public:
		std::vector<std::unique_ptr<connection>> getOutputWeights() { return std::move(outputWeights); }
	};
}

#endif // NN_H
