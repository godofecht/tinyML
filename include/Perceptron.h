//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#pragma once
#include "Model.h"

namespace ML
{
namespace Models
{
    class Perceptron : public Model
    {
    public:
        Perceptron (std::vector<unsigned> topology) : Model (topology)
        {
            setTopology (topology);
        }

        void learnSupervised (std::vector<double> targetStream)
        {
            backPropagate (targetStream);
        }

        std::vector<double> process (std::vector<double> inputStream)
        {
            feedForward (inputStream);
            return (getResult());
        }

        void executeBehavior()
        {

        }
    };

    //Move this to its own spot in another folder.
    //At the moment this is set up to have only 1 hidden layer.
    //Maybe we can allow for more later on.
    //Please change this to NNMapper or DimensionReducer or smt.
    class LinReg2D : public Perceptron
    {
    public:
        //This is basically a 2D Box with 2 controllable axes.
        //It effectively is a combination of 2 sliders
        //albeit with a different graphical output

        std::vector<double> valuesToLearn;

        float xAxisValue;
        float yAxisValue;

        std::vector<double> results;

        LinReg2D (int numParameters, int numHiddenLayerUnits) : Perceptron ({ 2, (unsigned) numHiddenLayerUnits, (unsigned)numParameters })
        {
            xAxisValue = 0.0f;
            yAxisValue = 0.0f;

            for (auto value : results)
            {
                value = 0.0f;
            }
        }

        void setValuesToLearn(std::vector<double> values)
        {
            valuesToLearn = values;
        }

        //We call this function from the UI when we move the dot
        //PRT maybe sort out this weirdness with the doubles
        void setXAndYValues (double x, double y)
        {
            xAxisValue = (float)x;
            yAxisValue = (float)y;
        }

        void Learn()
        {
            feedForward ({ xAxisValue, yAxisValue });
            learnSupervised ({ valuesToLearn });
        }

        void LearnParameters(std::vector<double> Parameters)
        {
            feedForward ({ xAxisValue, yAxisValue });
            learnSupervised (Parameters);
        }

        void calculateResult()
        {
            results = getResult();
        }

        //To Do:: Make labelX position itself on X axis
        // Make labelY position itself on Y axis
    };
}
}
