#pragma once 


#include "SPM.hpp"
using SPM::CSR;

/**
 * Ax = y
 */
void spmv_omp(const CSR &A, const double *x, double *&y);

void dcsrmv(const int m, const double *a, const int *ia, const int *ja, const double *x, double *y, const double alpha, const double beta);
