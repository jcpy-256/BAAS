

// #include "common.h"
// #include "btools.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <MarcoUtils.hpp>
#include "mmio_highlevel.h"
#include "mm_io.h"
// #define int int
// #define double double

void exclusive_scan(int *input, int length)
{
    if (length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

/**
 * quick sort [left, right] from small to large
 */
void quicksort(int *idx, double *w, int left, int right)
{
    if (left >= right)
        return;

    std::swap(idx[left], idx[left + (right - left) / 2]);
    std::swap(w[left], w[left + (right - left) / 2]);

    int last = left;
    for (int i = left + 1; i <= right; i++)
    {
        if (idx[i] < idx[left])
        {
            ++last;
            std::swap(idx[last], idx[i]);
            std::swap(w[last], w[i]);
        }
    }

    std::swap(idx[left], idx[last]);
    std::swap(w[left], w[last]);

    quicksort(idx, w, left, last - 1);
    quicksort(idx, w, last + 1, right);
}

int mmio_feature(int *m, int *n, int *nnz, int *isSymmetric, char *matrix_type, const char *filename)
{
    int m_tmp, n_tmp, nnz_tmp;
    int ret_code;

    MM_typecode matcode;
    FILE *f;
    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;
    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp + 1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));
    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *csrVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */

    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */

    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based

        idxi--;
        idxj--;
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }
    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    int old_val, new_val;
    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;

    for (int i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];
        old_val = new_val;
    }
    nnz_tmp = csrRowPtr_counter[m_tmp];
    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    sprintf(matrix_type, "%c", matcode[2]);

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);
    return 0;
}

// read matrix infomation from mtx file
int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, const char *filename)
{

    int m_tmp, n_tmp, nnz_tmp;
    int ret_code;

    MM_typecode matcode;
    FILE *f;
    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;
    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp + 1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));
    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *csrVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */

    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */

    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based

        idxi--;
        idxj--;
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }
    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    int old_val, new_val;
    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;

    for (int i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];
        old_val = new_val;
    }
    nnz_tmp = csrRowPtr_counter[m_tmp];
    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);
    return 0;
}

// read matrix infomation from mtx file

int mmio_data(int *csrRowPtr, int *csrColIdx, double *csrVal, const char *filename)
{

    int m_tmp, n_tmp, nnz_tmp;
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix

    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }

    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }

    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }

    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp + 1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));
    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *csrVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */

    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */

    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;
        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }

        else if (isPattern)

        {

            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);

            fval = 1.0;
        }

        // adjust from 1-based to 0-based

        idxi--;

        idxj--;

        csrRowPtr_counter[idxi]++;

        csrRowIdx_tmp[i] = idxi;

        csrColIdx_tmp[i] = idxj;

        csrVal_tmp[i] = fval;
    }

    if (f != stdin)

        fclose(f);

    if (isSymmetric_tmp)

    {

        for (int i = 0; i < nnz_mtx_report; i++)

        {

            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])

                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter

    int old_val, new_val;

    old_val = csrRowPtr_counter[0];

    csrRowPtr_counter[0] = 0;

    for (int i = 1; i <= m_tmp; i++)

    {

        new_val = csrRowPtr_counter[i];

        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];

        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];

    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp + 1) * sizeof(int));

    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

    if (isSymmetric_tmp)

    {

        for (int i = 0; i < nnz_mtx_report; i++)

        {

            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])

            {

                int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

                csrColIdx[offset] = csrColIdx_tmp[i];

                csrVal[offset] = csrVal_tmp[i];

                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];

                csrColIdx[offset] = csrRowIdx_tmp[i];

                csrVal[offset] = csrVal_tmp[i];

                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }

            else

            {

                int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

                csrColIdx[offset] = csrColIdx_tmp[i];

                csrVal[offset] = csrVal_tmp[i];

                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }

    else

    {

        for (int i = 0; i < nnz_mtx_report; i++)

        {

            int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

            csrColIdx[offset] = csrColIdx_tmp[i];

            csrVal[offset] = csrVal_tmp[i];

            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    // free tmp space

    free(csrColIdx_tmp);

    free(csrVal_tmp);

    free(csrRowIdx_tmp);

    free(csrRowPtr_counter);

    return 0;
}

// read matrix infomation from mtx file buf don't processing symmetric matrix
int mmio_allinone(int *m, int *n, int *nnz, int *isSymmetric,
                  int **cooRowIdx, int **cooColIdx, double **cooVal,
                  const char *filename)
{
    int m_tmp, n_tmp;
    int nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }
    // printf("row: %d, col: %d \n",m_tmp,n_tmp);

    // int *csrRowPtr_counter = (int *)malloc((m_tmp + 1) * sizeof(int));
    // memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

    int *cooRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *cooColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *cooVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        // csrRowPtr_counter[idxi]++;
        cooRowIdx_tmp[i] = idxi;
        cooColIdx_tmp[i] = idxj;
        cooVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    // if (isSymmetric_tmp)
    // {
    //     for (int i = 0; i < nnz_mtx_report; i++)
    //     {
    //         if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
    //             csrRowPtr_counter[csrColIdx_tmp[i]]++;
    //     }
    // }

    // exclusive scan for csrRowPtr_counter
    // prefix_sum(csrRowPtr_counter, m_tmp + 1);

    // int *csrRowPtr_alias = (int *)malloc((m_tmp + 1) * sizeof(int));
    // nnz_tmp = csrRowPtr_counter[m_tmp];
    // int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    // double *csrVal_alias = (double *)malloc(nnz_tmp * sizeof(double));

    // memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp + 1) * sizeof(int));
    // memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

    // if (isSymmetric_tmp)
    // {
    //     for (int i = 0; i < nnz_mtx_report; i++)
    //     {
    //         if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
    //         {
    //             int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
    //             csrColIdx_alias[offset] = csrColIdx_tmp[i];
    //             csrVal_alias[offset] = csrVal_tmp[i];
    //             csrRowPtr_counter[csrRowIdx_tmp[i]]++;

    //             offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
    //             csrColIdx_alias[offset] = csrRowIdx_tmp[i];
    //             csrVal_alias[offset] = csrVal_tmp[i];
    //             csrRowPtr_counter[csrColIdx_tmp[i]]++;
    //         }
    //         else
    //         {
    //             int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
    //             csrColIdx_alias[offset] = csrColIdx_tmp[i];
    //             csrVal_alias[offset] = csrVal_tmp[i];
    //             csrRowPtr_counter[csrRowIdx_tmp[i]]++;
    //         }
    //     }
    // }
    // else
    // {
    //     for (int i = 0; i < nnz_mtx_report; i++)
    //     {
    //         int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
    //         csrColIdx_alias[offset] = csrColIdx_tmp[i];
    //         csrVal_alias[offset] = csrVal_tmp[i];
    //         csrRowPtr_counter[csrRowIdx_tmp[i]]++;
    //     }
    // }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_mtx_report;
    *isSymmetric = isSymmetric_tmp;

    *cooRowIdx = cooRowIdx_tmp;
    *cooColIdx = cooColIdx_tmp;
    *cooVal = cooVal_tmp;

    // free tmp space
    // free(csrColIdx_tmp);
    // free(csrVal_tmp);
    // free(csrRowIdx_tmp);
    // free(csrRowPtr_counter);
    return 0;
}

// read matrix infomation from mtx file
int mmio_allinone_coo(int *m, int *n, int *nnz, int *isSymmetric,
                      int **cooRowIdx, int **cooColIdx, double **cooVal,
                      const char *filename)
{
    int m_tmp, n_tmp;
    int nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    // int *csrRowPtr_counter = (int *)malloc((m_tmp + 1) * sizeof(int));
    // memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

    int *cooRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *cooColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *cooVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));
    // printf("nnz_mtx_report: %d\n", nnz_mtx_report);

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        // csrRowPtr_counter[idxi]++;
        cooRowIdx_tmp[i] = idxi;
        cooColIdx_tmp[i] = idxj;
        cooVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    int index = 0;
    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int row = cooRowIdx_tmp[i];
        int col = cooColIdx_tmp[i];
        if (row != col)
        {
            index++;
        }
    }
    // printf("index: %d\n", index);

    // processing symmetric square matrix
    if (isSymmetric_tmp)
    {
        int count = nnz_mtx_report * 2;
        // printf("count: %d, nnz_mtx_report: %d, m_tmp:%d\n", count, nnz_mtx_report, m_tmp);
        int *rowidx_new = (int *)malloc(count * sizeof(int));
        int *colidx_new = (int *)malloc(count * sizeof(int));
        double *values_new = (double *)malloc(count * sizeof(double));
        // CHECK_POINTER(rowidx_new)
        // CHECK_POINTER(colidx_new)
        // CHECK_POINTER(values_new)
        memcpy(rowidx_new, cooRowIdx_tmp, nnz_mtx_report * sizeof(int));
        memcpy(colidx_new, cooColIdx_tmp, nnz_mtx_report * sizeof(int));
        memcpy(values_new, cooVal_tmp, nnz_mtx_report * sizeof(double));
        int org_nnz = nnz_mtx_report;
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            int row = cooRowIdx_tmp[i];
            int col = cooColIdx_tmp[i];
            if (row != col)
            {
                rowidx_new[org_nnz] = col;
                colidx_new[org_nnz] = row;
                values_new[org_nnz] = cooVal_tmp[i];
                org_nnz++;
            }
        }
        // if (org_nnz != count)
        // {
        //     printf("symmetric opertaion failure in %s of file %s\n", __LINE__, __FILE__);
        //     exit(2);
        // }
        colidx_new = (int *)realloc(colidx_new, org_nnz * sizeof(int ));
        rowidx_new = (int *)realloc(rowidx_new, org_nnz * sizeof(int ));
        values_new = (double *)realloc(values_new, org_nnz * sizeof(double ));

        free(cooRowIdx_tmp);
        free(cooColIdx_tmp);
        free(cooVal_tmp);
        cooColIdx_tmp = colidx_new;
        cooRowIdx_tmp = rowidx_new;
        cooVal_tmp = values_new;
        nnz_mtx_report = org_nnz;
    }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_mtx_report;
    *isSymmetric = isSymmetric_tmp;

    *cooRowIdx = cooRowIdx_tmp;
    *cooColIdx = cooColIdx_tmp;
    *cooVal = cooVal_tmp;
    return 0;
}

int mmio_allinone_csr(int *m, int *n, int *nnz, int *isSymmetric,
                      int **csrRowPtr, int **csrColIdx, double **csrVal,
                      const char *filename)
{
    int m_tmp, n_tmp;
    int nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp + 1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *csrVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp + 1);

    int *csrRowPtr_alias = (int *)malloc((m_tmp + 1) * sizeof(int));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    double *csrVal_alias = (double *)malloc(nnz_tmp * sizeof(double));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp + 1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < m_tmp; i++)
    {
        quicksort(csrColIdx_alias, csrVal_alias, csrRowPtr_alias[i], csrRowPtr_alias[i + 1] - 1);
    }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal = csrVal_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

int mmio_allinone_csc(int *m, int *n, int *nnz, int *isSymmetric,
                      int **cscColPtr, int **cscRowIdx, double **cscVal,
                      const char *filename)

{
    int m_tmp, n_tmp;
    int nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    int *cscColPtr_counter = (int *)malloc((n_tmp + 1) * sizeof(int));
    memset(cscColPtr_counter, 0, (n_tmp + 1) * sizeof(int));

    int *cscRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *cscColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *cscVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        cscColPtr_counter[idxi]++;
        cscRowIdx_tmp[i] = idxi;
        cscColIdx_tmp[i] = idxj;
        cscVal_tmp[i] = fval;
    }

    if (isSymmetric_tmp) // 修改col counter 的数据
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (cscColIdx_tmp[i] != cscRowIdx_tmp[i])
            {
                cscColPtr_counter[cscRowIdx_tmp[i]]++;
            }
        }
    }

    exclusive_scan(cscColPtr_counter, n_tmp + 1);
    nnz_mtx_report = cscColPtr_counter[n_tmp];

    int *cscColPtr_alias = (int *)malloc((n_tmp + 1) * sizeof(int));
    nnz_tmp = cscColPtr_counter[n_tmp];
    int *cscRowIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    double *cscVal_alias = (double *)malloc(nnz_tmp * sizeof(double));

    memcpy(cscColPtr_alias, cscColPtr_counter, (n_tmp + 1) * sizeof(int));
    memset(cscColPtr_counter, 0, (n_tmp + 1) * sizeof(int));

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (cscColIdx_tmp[i] != cscRowIdx_tmp[i])
            {
                int offset = cscColPtr_alias[cscColIdx_tmp[i]] + cscColPtr_counter[cscColIdx_tmp[i]];
                cscRowIdx_alias[offset] = cscRowIdx_tmp[i];
                cscVal_alias[offset] = cscVal_tmp[i];
                cscColPtr_counter[cscColIdx_tmp[i]]++;

                offset = cscColPtr_alias[cscRowIdx_tmp[i]] + cscColPtr_counter[cscRowIdx_tmp[i]];
                cscRowIdx_alias[offset] = cscColIdx_tmp[i];
                cscVal_alias[offset] = cscVal_tmp[i];
                cscColPtr_counter[cscRowIdx_tmp[i]]++;
            }
            else
            {
                int offset = cscColPtr_alias[cscColIdx_tmp[i]] + cscColPtr_counter[cscColIdx_tmp[i]];
                cscRowIdx_alias[offset] = cscRowIdx_tmp[i];
                cscVal_alias[offset] = cscVal_tmp[i];
                cscColPtr_counter[cscColIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            int offset = cscColPtr_alias[cscColIdx_tmp[i]] + cscColPtr_counter[cscColIdx_tmp[i]];
            cscRowIdx_alias[offset] = cscRowIdx_tmp[i];
            cscVal_alias[offset] = cscVal_tmp[i];
            cscColPtr_counter[cscColIdx_tmp[i]]++;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n_tmp; i++)
    {
        quicksort(cscRowIdx_alias, cscVal_alias, cscColPtr_alias[i], cscColPtr_alias[i + 1] - 1);
    }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *cscColPtr = cscColPtr_alias;
    *cscRowIdx = cscRowIdx_alias;
    *cscVal = cscVal_alias;

    // free tmp space
    free(cscColIdx_tmp);
    free(cscRowIdx_tmp);
    free(cscVal_tmp);
    free(cscColPtr_counter);

    return 0;
}