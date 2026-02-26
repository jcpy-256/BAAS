
#include <cstring>
#include <algorithm>

#include "COO.hpp"
#include "mm_io.h"
#include "spmUtils.hpp"
#include "mmio_highlevel.h"
#include "MarcoUtils.hpp"

using namespace std;

namespace SPM
{

  COO::COO() : rowidx(NULL), colidx(NULL), values(NULL), isSymmetric(false)
  {
  }

  COO::~COO()
  {
    dealloc();
  }

  void COO::dealloc()
  {
    FREE(rowidx);
    FREE(colidx);
    FREE(values);
  }

  void COO::storeMatrixMarket(const char *fileName) const
  {
    FILE *fp = fopen(fileName, "w");
    if (NULL == fp)
    {
      fprintf(stderr, "Fail to open file %s\n", fileName);
      return;
    }

    MM_typecode matcode;
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_sparse(&matcode);
    mm_set_real(&matcode);

    int err = mm_write_mtx_crd(
        (char *)fileName, m, n, nnz, rowidx, colidx, values, matcode);
    if (err)
    {
      fprintf(
          stderr,
          "Fail to write matrix to %s (error code = %d)\n", fileName, err);
    }
  }

  /** quick sort in [left, right] from small to large */
  template <class T>
  static void qsort(int *idx, T *w, int left, int right)
  {
    if (left >= right)
      return;

    swap(idx[left], idx[left + (right - left) / 2]);
    swap(w[left], w[left + (right - left) / 2]);

    int last = left;
    for (int i = left + 1; i <= right; i++)
    {
      if (idx[i] < idx[left])
      {
        ++last;
        swap(idx[last], idx[i]);
        swap(w[last], w[i]);
      }
    }

    swap(idx[left], idx[last]);
    swap(w[left], w[last]);

    qsort(idx, w, left, last - 1);
    qsort(idx, w, last + 1, right);
  }

  /**
   * converts COO format to CSC format, not in-place if SORT_IN_COL is defined,
   * each row is sorted in row index. assume COO is zero-based index
   */

  template <class T>
  void coo2csc(
      int n, int nnz,
      int *colptr, int *rowidx, T *values,
      const int *cooRowidx, const int *cooColidx, const T *cooValues,
      bool sort,
      int outBase = 0)
  {
    int i, l;
    /* determine column lengths */
    for (i = 0; i < nnz; i++)
      colptr[cooColidx[i] + 1]++;
    // exclusive scan
    for (i = 0; i < n; i++)
      colptr[i + 1] += colptr[i];
    printf("col ptr max: %d\n", colptr[n]);
    printf("nnz: %d", nnz);

    /* go through the structure  once more. Fill in output matrix. */
    for (l = 0; l < nnz; l++)
    {
      i = colptr[cooColidx[l]];
      values[i] = cooValues[l];
      rowidx[i] = cooRowidx[l] + outBase;
      colptr[cooColidx[l]]++;
    }

    /* shift back rowptr */
    for (i = n; i > 0; i--)
      colptr[i] = colptr[i - 1] + outBase;

    colptr[0] = outBase;

    if (sort)
    {
#pragma omp parallel for
      for (i = 0; i < n; i++)
      {
        qsort(rowidx, values, colptr[i] - outBase, colptr[i + 1] - 1 - outBase);
        assert(is_sorted(rowidx + colptr[i] - outBase, rowidx + colptr[i + 1] - outBase));
      }
    }
  }

  void dcoo2csc(
      int n, int nnz,
      int *colptr, int *rowidx, double *values,
      const int *cooRowidx, const int *cooColidx, const double *cooValues,
      bool sort /*=true*/,
      int outBase /*=0*/)
  {
    coo2csc(n, nnz, colptr, rowidx, values, cooRowidx, cooColidx, cooValues, sort, outBase);
  }

  void dcoo2csc(CSC *Acsc, const COO *Acoo, int outBase /*=0*/, bool createSeparateDiagData /*= true*/)
  {
    Acsc->n = Acoo->n;
    Acsc->m = Acoo->m;

    dcoo2csc(
        Acsc->n, Acoo->nnz,
        Acsc->colptr, Acsc->rowidx, Acsc->values,
        Acoo->rowidx, Acoo->colidx, Acoo->values,
        true /*sort*/, outBase);

    int base = Acsc->getBase();
    if (Acsc->diagptr)
    {
      if (!Acsc->idiag || !Acsc->diag)
      {
        createSeparateDiagData = false;
      }
#pragma omp parallel for
      for (int i = 0; i < Acsc->n; ++i)
      {
        for (int j = Acsc->colptr[i] - base; j < Acsc->colptr[i + 1] - base; ++j)
        {
          if (Acsc->rowidx[j] - base == i)
          {
            Acsc->diagptr[i] = j + base;

            if (createSeparateDiagData)
            {
              Acsc->idiag[i] = 1 / Acsc->values[j];
              Acsc->diag[i] = Acsc->values[j];
            }
          }
        }
      }
    }
  }

  //   void dcoo2csc(CSC *Acsc, const COO *Acoo, int outBase /*=0*/, bool createSeparateDiagData /*= true*/)
  //   {
  //     Acrs->n = Acoo->n;
  //     Acrs->m = Acoo->m;

  //     dcoo2csc(
  //         Acsc->n, Acoo->nnz,
  //         Acrs->rowptr, Acrs->colidx, Acrs->values,
  //         Acoo->rowidx, Acoo->colidx, Acoo->values,
  //         true /*sort*/, outBase);

  //     int base = Acrs->getBase();
  //     if (Acrs->diagptr)
  //     {
  //       if (!Acrs->idiag || !Acrs->diag)
  //       {
  //         createSeparateDiagData = false;
  //       }
  // #pragma omp parallel for
  //       for (int i = 0; i < Acrs->m; ++i)
  //       {
  //         for (int j = Acrs->rowptr[i] - base; j < Acrs->rowptr[i + 1] - base; ++j)
  //         {
  //           if (Acrs->colidx[j] - base == i)
  //           {
  //             Acrs->diagptr[i] = j + base;

  //             if (createSeparateDiagData)
  //             {
  //               Acrs->idiag[i] = 1 / Acrs->values[j];
  //               Acrs->diag[i] = Acrs->values[j];
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }

  /* converts COO format to CSR format, not in-place,
     if SORT_IN_ROW is defined, each row is sorted in column index.
  assume COO is one-based index */

  template <class T>
  void coo2csr(
      int m, int nnz,
      int *rowptr, int *colidx, T *values,
      const int *cooRowidx, const int *cooColidx, const T *cooValues,
      bool sort,
      int outBase = 0)
  {
    int i, l;
  
    // printf("end element row: %d, col: %d, val: %lf\n", cooRowidx[nnz-1], cooColidx[nnz -1], cooValues[nnz - 1]);

#pragma omp parallel for
    for (i = 0; i <= m; i++)
      rowptr[i] = 0;
    CHECK_POINTER(cooRowidx);

    /* determine row lengths */
    for (i = 0; i < nnz; i++)
      rowptr[cooRowidx[i] + 1]++;

    for (i = 0; i < m; i++)
      rowptr[i + 1] += rowptr[i];

    /* go through the structure  once more. Fill in output matrix. */
    for (l = 0; l < nnz; l++)
    {
      i = rowptr[cooRowidx[l]];
      values[i] = cooValues[l];
      colidx[i] = cooColidx[l] + outBase;
      rowptr[cooRowidx[l]]++;
    }

    /* shift back rowptr */
    for (i = m; i > 0; i--)
      rowptr[i] = rowptr[i - 1] + outBase;

    rowptr[0] = outBase;

    if (sort)
    {
#pragma omp parallel for
      for (i = 0; i < m; i++)
      {
        qsort(colidx, values, rowptr[i] - outBase, rowptr[i + 1] - 1 - outBase);
        assert(is_sorted(colidx + rowptr[i] - outBase, colidx + rowptr[i + 1] - outBase));
      }
    }
  }

  void dcoo2csr(
      int m, int nnz,
      int *rowptr, int *colidx, double *values,
      const int *cooRowidx, const int *cooColidx, const double *cooValues,
      bool sort /*=true*/,
      int outBase /*=0*/)
  {
    coo2csr(m, nnz, rowptr, colidx, values, cooRowidx, cooColidx, cooValues, sort, outBase);
  }

  void dcoo2csr(CSR *Acrs, const COO *Acoo, int outBase /*=0*/, bool createSeparateDiagData /*= true*/)
  {
    Acrs->n = Acoo->n;
    Acrs->m = Acoo->m;

    dcoo2csr(
        Acrs->m, Acoo->nnz,
        Acrs->rowptr, Acrs->colidx, Acrs->values,
        Acoo->rowidx, Acoo->colidx, Acoo->values,
        true /*sort*/, outBase);

    int base = Acrs->getBase();
    if (Acrs->diagptr)
    {
      if (!Acrs->idiag || !Acrs->diag)
      {
        createSeparateDiagData = false;
      }
#pragma omp parallel for
      for (int i = 0; i < Acrs->m; ++i)
      {
        for (int j = Acrs->rowptr[i] - base; j < Acrs->rowptr[i + 1] - base; ++j)
        {
          if (Acrs->colidx[j] - base == i)
          {
            Acrs->diagptr[i] = j + base;

            if (createSeparateDiagData)
            {
              Acrs->idiag[i] = 1 / Acrs->values[j];
              Acrs->diag[i] = Acrs->values[j];
            }
          }
        }
      }
    }
  }

  void loadMatrixMarket_mmio_highlevel_coo(const char *file, COO &coo)
  {
    int m, n, nnz;
    int *rowidx, *colidx;
    double *values;
    int issymmetric;
    // int base = 0;
    /** base 0 */
    mmio_allinone_coo(&m, &n, &nnz, &issymmetric, &rowidx, &colidx, &values, file);
    coo.m = m;
    coo.n = n;
    coo.nnz = nnz;
    coo.dealloc();
    coo.values = values;
    coo.colidx = colidx;
    coo.rowidx = rowidx;
  }

  static bool loadMatrixMarket_(const char *file, COO &coo, bool force_symmetric, bool transpose, int pad)
  {
    FILE *fp = fopen(file, "r");
    if (NULL == fp)
    {
      fprintf(stderr, "Failed to open file %s\n", file);
      exit(-1);
    }

    // read banner
    MM_typecode matcode;
    if (mm_read_banner(fp, &matcode) != 0)
    {
      fprintf(stderr, "Error: could not process Matrix Market banner.\n");
      fclose(fp);
      return false;
    }

    if (!mm_is_valid(matcode) || mm_is_array(matcode) || mm_is_dense(matcode))
    {
      fprintf(stderr, "Error: only support sparse and real matrices.\n");
      fclose(fp);
      return false;
    }
    bool pattern = mm_is_pattern(matcode);

    // read sizes
    int m, n;
    int nnz; // # of non-zeros specified in the file
    if (mm_read_mtx_crd_size(fp, &m, &n, &nnz) != 0)
    {
      fprintf(stderr, "Error: could not read matrix size.\n");
      fclose(fp);
      return false;
    }
    if (transpose)
    {
      assert(!force_symmetric);
      swap(m, n);
    }

    int origM = m, origN = n;
    m = (m + pad - 1) / pad * pad;
    n = (n + pad - 1) / pad * pad;

    size_t count;
    if (force_symmetric || mm_is_symmetric(matcode) == 1)
    {
      coo.isSymmetric = true;
      count = 2L * nnz;
    }
    else
    {
      count = nnz;
    }

    // allocate memory
    size_t extraCount = min(m, n) - min(origM, origN);
    double *values = MALLOC(double, count + extraCount);
    int *colidx = MALLOC(int, count + extraCount);
    int *rowidx = MALLOC(int, count + extraCount);
    if (!values || !colidx || !rowidx)
    {
      fprintf(stderr, "Failed to allocate memory\n");
      fclose(fp);
      return false;
    }

    int *colidx_temp, *rowcnt = NULL;
    if (coo.isSymmetric)
    {
      colidx_temp = MALLOC(int, count);
      rowcnt = MALLOC(int, m + 1);
      if (!colidx_temp || !rowcnt)
      {
        fprintf(stderr, "Failed to allocate memory\n");
        fclose(fp);
        return false;
      }
      memset(rowcnt, 0, sizeof(int) * (m + 1));
    }

    // read values
    count = 0;
    int lines = 0;
    int x, y;
    double real, imag;
    int base = 1;
    while (mm_read_mtx_crd_entry(fp, &x, &y, &real, &imag, matcode) == 0)
    {
      if (transpose)
        swap(x, y);

      if (x > origM || y > origN)
      {
        fprintf(stderr, "Error: (%d %d) coordinate is out of range.\n", x, y);
        fclose(fp);
        return false;
      }

      rowidx[count] = x;
      colidx[count] = y;
      values[count] = pattern ? 1 : real;
      if (0 == x || 0 == y)
        base = 0;

      ++count;
      ++lines;
      if (coo.isSymmetric)
        rowcnt[x]++;
      // this is not a bug. we're intentionally indexing rowcnt[x] instead of rowcnt[x-1]
    }
    // padding for vectorization
    for (int i = min(origM, origN); i < min(m, n); ++i)
    {
      rowidx[count] = i + 1;
      colidx[count] = i + 1;
      values[count] = 1;
      ++count;
    }
    fclose(fp);

    if (0 == base)
    {
      for (size_t i = 0; i < count; ++i)
      {
        rowidx[i]++;
        colidx[i]++;
      }
      if (coo.isSymmetric)
      {
        for (int i = m; i > 0; --i)
        {
          rowcnt[i] = rowcnt[i - 1];
        }
      }
    }

    if (lines != nnz)
    {
      fprintf(stderr, "Error: nnz (%d) specified in the header doesn't match with # of lines (%d) in file %s\n",
              nnz, lines, file);
      return false;
    }

    if (coo.isSymmetric)
    {
      // add transposed elements only if it doesn't exist
      size_t real_count = count;
      // preix-sum
      for (int i = 0; i < m; ++i)
      {
        rowcnt[i + 1] += rowcnt[i];
      }
      for (size_t i = 0; i < count; ++i)
      {
        int j = rowcnt[rowidx[i] - 1];
        colidx_temp[j] = colidx[i];
        rowcnt[rowidx[i] - 1]++;
      }
      for (int i = m; i > 0; --i)
      {
        rowcnt[i] = rowcnt[i - 1];
      }
      rowcnt[0] = 0;

#pragma omp parallel for
      for (int i = 0; i < m; ++i)
      {
        sort(colidx_temp + rowcnt[i], colidx_temp + rowcnt[i + 1]);
      }

      for (size_t i = 0; i < count; ++i)
      {
        int x = rowidx[i], y = colidx[i];
        if (x != y)
        {
          if (!binary_search(
                  colidx_temp + rowcnt[y - 1], colidx_temp + rowcnt[y], x))
          {
            rowidx[real_count] = y;
            colidx[real_count] = x;
            values[real_count] = values[i];
            ++real_count;
          }
        }
      }
      count = real_count;

      FREE(rowcnt);
      FREE(colidx_temp);
    }

    coo.m = m;
    coo.n = n;
    coo.nnz = count;
    coo.dealloc();
    coo.values = values;
    coo.colidx = colidx;
    coo.rowidx = rowidx;

    return true;
  }

  bool loadMatrixMarketTransposed(const char *file, COO &coo, int pad /*= 1*/)
  {
    return loadMatrixMarket_(file, coo, false, true /*transpose*/, pad);
  }

  bool loadMatrixMarket(const char *file, COO &coo, bool force_symmetric /*=false*/, int pad /*=1*/)
  {
    return loadMatrixMarket_(file, coo, force_symmetric, false /*no-transpose*/, pad);
  }

  bool loadLowerMatrixMarket(const char *file, COO &coo)
  {
    loadMatrixMarket_mmio_highlevel_coo(file, coo);
    splitLowerAndAddMajorDiagonal(coo);
    return true;
  }

  /**
   * 将COO格式的矩阵进行拆分，仅仅保留下三角的元素，并且
   */
  void splitLowerAndAddMajorDiagonal(COO &coo, double diagElem /*=1.0 */)
  {
    if (coo.m != coo.n)
    {
      printf("non-square matrix can not split lower matrix!\n");
      exit(2);
    }
    int lower_nnz = 0;
    int m = coo.m;
    int *colidx = coo.colidx;
    int *rowidx = coo.rowidx;
    double *values = coo.values;

    // 对coo进行双指针遍历
    int offset = 0;
    for (int i = 0; i < coo.nnz; i++)
    {
      if (colidx[i] < rowidx[i])
      {
        colidx[offset] = colidx[i];
        rowidx[offset] = rowidx[i];
        values[offset] = values[i];
        offset++;
      }
    }
    lower_nnz = offset + m;
    // 加上对角的数据进行空间重分配
    int *rowidx_temp = MALLOC(int, lower_nnz);
    int *colidx_temp = MALLOC(int, lower_nnz);
    double *values_temp = MALLOC(double, lower_nnz);
    CHECK_POINTER(rowidx_temp);
    CHECK_POINTER(colidx_temp);
    CHECK_POINTER(values_temp);

    memcpy(rowidx_temp, rowidx, offset * sizeof(int));
    memcpy(colidx_temp, colidx, offset * sizeof(int));
    memcpy(values_temp, values, offset * sizeof(double));
    // rowidx = (int *)realloc(rowidx, lower_nnz * sizeof(int));
    // colidx = (int *)realloc(colidx, lower_nnz * sizeof(int));
    // values = (double *)realloc(values, lower_nnz * sizeof(double));
    // 复制对角数据到新空间
    for (int i = 0; i < m; i++)
    {
      colidx_temp[offset] = i;
      rowidx_temp[offset] = i;
      values_temp[offset] = diagElem;
      offset++;
    }
    FREE(colidx);
    FREE(rowidx);
    FREE(values);

    lower_nnz = offset;
    coo.colidx = colidx_temp;
    coo.rowidx = rowidx_temp;
    coo.values = values_temp;

    coo.nnz = lower_nnz;
  }

} // namespace SPM
