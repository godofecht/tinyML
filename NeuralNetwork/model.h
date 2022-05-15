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

class Model
{
public:
	Network thisNetwork;
	vector<unsigned> topology;
	vector<double> weights;
	public:

	Model (vector<unsigned> tp) : thisNetwork (tp)
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

    void SetTopology (vector<unsigned> tp)
    {
        topology = tp;
    }
    
	void BackPropagate (vector<double>& targetVals)
	{
		thisNetwork.backPropagate (targetVals);
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
	void feedforward (vector<double> inputs)
	{
		thisNetwork.feedForward (inputs);
	}
	vector<double> GetResult()
	{
		vector<double> resultVals;
		thisNetwork.getResults (resultVals);
		return resultVals;
	}
	void SetWeights (vector<double> newWeights)
	{
		thisNetwork.PutWeights (newWeights);
	}

    void DisplayTopology()
    {
        for(int i=0; i<topology.size(); i++)
        {
            std::cout<<topology[i]<<"\n";
        }
    }
	
	void UpdateWeights()
	{
		thisNetwork.UpdateWeights();
	}
};

class Agent : public Model
{
public:
    Agent (vector<unsigned> topology) : Model (topology)
    {
        SetTopology (topology);
    }
    
    void LearnSupervised (vector<double> targetStream)
    {
        BackPropagate (targetStream);
    }
    
    //Need to write a method (or multiple) for this
    void LearnUnsupervised()
    {
    }
    
    vector<double> Process (vector<double> inputStream)
    {
        feedforward (inputStream);
        return (GetResult());
    }
    
    void ExecuteBehavior()
    {
        
    }
};



//Move this to its own spot in another folder.
//At the moment this is set up to have only 1 hidden layer.
//Maybe we can allow for more later on.
class FeatureMapper2D : public Agent
{
public:
    //This is basically a 2D Box with 2 controllable axes.
    //It effectively is a combination of 2 sliders
    //albeit with a different graphical output

    std::vector<double> valuesToLearn;

    float xAxisValue;
    float yAxisValue;

    std::vector<double> results;
    
    FeatureMapper2D (int numParameters, int numHiddenLayerUnits) : 
        Agent ({2, (unsigned) numHiddenLayerUnits, (unsigned) numParameters})
    {
        xAxisValue = 0.0f;
        yAxisValue = 0.0f;

        for (auto value : results)
        {
            value = 0.0f;
        }
    }

    void setValuesToLearn (std::vector<double> values)
    {
        valuesToLearn = values;
    }

    //We call this function from the UI when we move the dot
    //PRT maybe sort out this weirdness with the doubles
    void setXAndYValues (double x, double y)
    {
        xAxisValue = (float) x;
        yAxisValue = (float) y;
    }

    void Learn()
    {
        feedforward ({ xAxisValue, yAxisValue });
        LearnSupervised ({ valuesToLearn });
    }

    void LearnParameters (std::vector<double> Parameters)
    {
        feedforward ({ xAxisValue, yAxisValue });
        LearnSupervised (Parameters);
    }

    void calculateResult()
    {
        results = GetResult();
    }


    
    //To Do:: Make labelX position itself on X axis
    // Make labelY position itself on Y axis
   
    
};


#endif
