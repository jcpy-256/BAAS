#pragma once 

double dot(const double x[], const double y[], int len);

double norm(const double x[], int len);

// w = a*x + b*y
void waxpby(int n, double w[], double alpha, const double x[], double beta, const double y[]);