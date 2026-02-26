
#include <iostream>
#include <cassert>
#include<cstdlib>
#include<cstring>

#include "CSC.hpp"
#include "COO.hpp"
#include "MarcoUtils.hpp"
#include "spmUtils.hpp"

namespace SPM
{
    CSC::CSC() : colptr(NULL), rowidx(NULL), values(NULL),
                 idiag(NULL), diag(NULL), diagptr(NULL), ownData_(false)
    {
    }

    void CSC::alloc(int n, int nnz, bool createSeparateDiagData /*= true*/)
    {
        this->n = n;
        colptr = MALLOC(int, n + 1);
        rowidx = MALLOC(int, nnz);
        values = MALLOC(double, nnz);
        diagptr = MALLOC(int, n);

        assert(colptr != NULL);
        assert(rowidx != NULL);
        assert(values != NULL);
        assert(diagptr != NULL);
        // initalize 0
        memset(colptr, 0, (n+1) * sizeof(int));
        memset(rowidx, 0, (nnz) * sizeof(int));
        memset(diagptr, 0, (n) * sizeof(int));

        if (createSeparateDiagData)
        {
            idiag = MALLOC(double, n);
            diag = MALLOC(double, n);

            assert(idiag != NULL);
            assert(diag != NULL);
        }

        ownData_ = true;
    }

    CSC::CSC(int m, int n, int nnz) : m(m), n(n)
    {
        alloc(n, nnz);
    }

    CSC::CSC(const char *fileName, int base /**0 */, bool forceSymmetric /**false*/, int pad /** 1 */)
    {
        COO Acoo;
        // CSC cscA();
        // bool issymmetric;
        loadMatrixMarket_mmio_highlevel_coo(fileName, Acoo);
        alloc(Acoo.n, Acoo.nnz);
        dcoo2csc(this, &Acoo);
        // Acoo.dealloc();

        // CSR csr;
        // dcoo2csr(&csr, &Acoo);
        // dcoo2csc()
        
        // dcoo2csc()
    }
    CSC::CSC(const char *fileName, bool isLower /** true */, int base /** 0 */)
    {
        COO Acoo;
        loadLowerMatrixMarket(fileName, Acoo);
        alloc(Acoo.n, Acoo.nnz);    // allocate memory for this CSC 
        dcoo2csc(this, &Acoo);
        // Acoo.dealloc();
    }

    CSC::CSC(const CSC &A) : m(A.m), n(A.n), values(NULL), idiag(NULL), diag(NULL), diagptr(NULL), ownData_(true)
    {
        int nnz = A.getNnz();

        colptr = MALLOC(int, n + 1);
        rowidx = MALLOC(int, nnz);
        if (A.values)
            values = MALLOC(double, nnz);
        if (A.diagptr)
            diagptr = MALLOC(int, n);
        if (A.idiag)
            idiag = MALLOC(double, n);
        if (A.diag)
            diag = MALLOC(double, n);

        copyVector(colptr, A.colptr, A.n + 1);
        copyVector(rowidx, A.rowidx, nnz);
        if (A.values)
            copyVector(values, A.values, nnz);
        if (A.diagptr)
            copyVector(diagptr, A.diagptr, n);
        if (A.idiag)
            copyVector(idiag, A.idiag, n);
        if (A.diag)
            copyVector(diag, A.diag, n);
    }

    CSC::~CSC()
    {
        dealloc();
    }

    void CSC::dealloc()
    {
        if (ownData_)
        {
            FREE(colptr);
            FREE(rowidx);
            FREE(values);
        }

        FREE(idiag);
        FREE(diag);
        FREE(diagptr);
    }

    bool CSC::isalloc()
    {
        if(!colptr || !rowidx || !values || m == 0 || n == 0)
        {
            return false;
        }
        return true;
    }
} // namespace SPM
