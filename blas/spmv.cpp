#include "omp.h"
#include "SPM.hpp"
using SPM::CSR;

/**
 * Ax = y
 */
void spmv_omp(const CSR &A, const double *x, double *&y)
{
    const int m = A.m;
    const int n = A.n;
    const int nnz = A.getNnz();
    const int *rowptr = A.rowptr;
    const int *colidx = A.colidx;
    const double *val = A.values;
    
#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        double sum = 0.0;
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
        {
            sum += val[j] * x[colidx[j]];
        }
        y[i] = sum;
    }
}

void dcsrmv(const int m, const double *a, const int *ia, const int *ja, const double *x, double *y, const double alpha, const double beta)
{
    const int *rowptr = ia;
    const int *colidx = ja;
    const double *val = a;
    
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
    {
        double sum = 0.0;
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
        {
            sum += val[j] * x[colidx[j]];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}
