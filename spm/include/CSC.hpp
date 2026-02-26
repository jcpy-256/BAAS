

#pragma once

// #include "spmUtils.hpp"
// #include "COO.hpp"
// #include "mmio_highlevel.h"

namespace SPM
{
    class CSC
    {
    private:
        bool ownData_;

    public:
        int m;
        int n;
        int *colptr;
        int *rowidx;
        double *values;

        double *idiag; /**< inverse of diagonal elements */
        double *diag;
        int *diagptr;
        CSC();
        CSC(int m, int n, int nnz);
        // Following two constructors will make CSR own the data
        CSC(const char *fileName, int base = 0, bool forceSymmetric = false, int pad = 1);
        CSC(const char *fileName, bool isLower = true, int base = 0);

        CSC(const CSC &A);
        ~CSC();
        void alloc(int n, int nnz, bool createSeparateDiagData = true);
        void dealloc();
        bool isalloc();

        int getNnz() const { return colptr[n] - getBase(); }
        int getBase() const { return colptr[0]; }
    };

} // namespace SPM
