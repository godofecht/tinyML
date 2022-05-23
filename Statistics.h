/*
  ==============================================================================

    Statistics.h
    Created: 23 May 2022 12:13:38pm
    Author:  Abhishek Shivakumar

  ==============================================================================
*/

#pragma once

inline float pearsoncoeff (std::vector<float> X, std::vector<float> Y)
{
    return sum ((X - mean (X))* (Y - mean (Y))) / (X.size() * stdev (X) * stdev (Y));
}