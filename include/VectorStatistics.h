//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 23/04/2022
*****************************************************************************/

#pragma once
#include <vector>
#include <cmath>
#include "VectorOperations.h"

inline double sum (std::vector<float> a)
{
    double s = 0;
    for (int i = 0; i < a.size(); i++)
    {
        s += a[i];
    }
    return s;
}

inline double mean  (std::vector<float> a)
{
    return sum (a) / (double) a.size();
}

inline double sqsum (std::vector<float> a)
{
    double s = 0;
    for (int i = 0; i < a.size(); i++)
    {
        s += pow(a[i], 2);
    }
    return s;
}

inline double stdev (std::vector<float> nums)
{
    double N = nums.size();
    return pow(sqsum(nums) / N - pow(sum(nums) / N, 2), 0.5);
}

inline float pearsoncoeff (std::vector<float> X, std::vector<float> Y)
{
    return sum ((X - mean (X))* (Y - mean (Y))) / (X.size() * stdev (X) * stdev (Y));
}
