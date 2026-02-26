

#pragma once

#include "CSR.hpp"
#include "CSC.hpp"
// #include "mmio_highlevel.h"

namespace SPM
{

  class COO
  { // 0-based index
  public:
    int m;
    int n;
    int nnz;
    int *rowidx;
    int *colidx;
    double *values;
    bool isSymmetric;

    COO();
    ~COO();
    void dealloc();

    void storeMatrixMarket(const char *fileName) const;
  };

  /**
   * @ret true if succeed
   */
  void loadMatrixMarket_mmio_highlevel_coo(const char *fileName, COO &Acoo);
  bool loadMatrixMarket(const char *fileName, COO &Acoo, bool force_symmetric = false, int pad = 1);
  bool loadMatrixMarketTransposed(const char *fileName, COO &Acoo, int pad = 1);

  void dcoo2csc(CSC *Acsc, const COO *Acoo, int outBase = 0, bool createSeparateDiagData = true);

  void dcoo2csc(
      int n, int nnz,
      int *colptr, int *rowidx, double *values,
      const int *cooRowidx, const int *cooColidx, const double *cooValues,
      bool sort = true,
      int outBase = 0);

  /**
   * @param createSeparateDiagData true then populate diag and idiag
   */
  void dcoo2csr(CSR *Acrs, const COO *Acoo, int outBase = 0, bool createSeparateDiagData = true);
  void dcoo2csr(
      int m, int nnz,
      int *rowptr, int *colidx, double *values,
      const int *cooRowidx, const int *cooColidx, const double *cooValues,
      bool sort = true,
      int outBase = 0);
  /**
   * 将COO格式的矩阵进行拆分，仅仅保留下三角的元素，并且
   */
  void splitLowerAndAddMajorDiagonal(COO &coo, double diagElem = 1.0);

  bool loadLowerMatrixMarket(const char *file, COO &coo);
} // namespace SPM
