#include "blasone.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

double dot(const double x[], const double y[], int len)
{
    double sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < len; ++i)
    {
        sum += x[i] * y[i];
    }
    return sum;
}

double norm(const double x[], int len)
{
    return sqrt(dot(x, x, len));
}

// w = a*x + b*y
void waxpby(int n, double w[], double alpha, const double x[], double beta, const double y[])
{
    if (1 == alpha)
    {
        if (1 == beta)
        {
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
            {
                w[i] = x[i] + y[i];
            }
        }
        else if (-1 == beta)
        {
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
            {
                w[i] = x[i] - y[i];
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
            {
                w[i] = x[i] + beta * y[i];
            }
        }
    }
    else if (1 == beta)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            w[i] = alpha * x[i] + y[i];
        }
    }
    else if (-1 == alpha)
    {
        if (0 == beta)
        {
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
            {
                w[i] = -x[i];
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
            {
                w[i] = beta * y[i] - x[i];
            }
        }
    }
    else if (0 == beta)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            w[i] = alpha * x[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            w[i] = alpha * x[i] + beta * y[i];
        }
    }
}

