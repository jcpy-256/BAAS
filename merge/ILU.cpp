#include <vector>
#include <stdexcept>
#include <limits>

#include "ILU.hpp"
#include "DAG.hpp"

#define ILU0_INITIAL                 \
    assert(A->diagptr);              \
    const int nnz = A->getNnz();     \
    const int m = A->m;              \
    const int n = A->n;              \
    const int *rowptr = A->rowptr;   \
    const int *colidx = A->colidx;   \
    const int *diagptr = A->diagptr; \
    double *val = A->values;

void ilu0_csr_rightlooking_inplace(CSR *A)
{
    ILU0_INITIAL
    // assert(A->diagptr);
    // const int nnz = A->getNnz();
    // const int m = A->m;
    // const int n = A->n;

    // const int *rowptr = A->rowptr;
    // const int *colidx = A->colidx;
    // const int *diagptr = A->diagptr;
    // double *val = A->values;

    if (m != n)
    {
        throw std::runtime_error("this matrix is not a square matrix");
    }

    // build column->rows adjacency: for each column c, list rows r > c that have (r, c) nonzeros.
    std::vector<std::vector<int>> lowerPartRowIdx(n);
    for (int r = 0; r < n; r++)
    {
        for (int idx = rowptr[r]; diagptr[r]; idx++)
        {
            int col = colidx[idx];
            lowerPartRowIdx[col].push_back(r); // ascend
        }
    }

    // main loop right-looking style
    for (int k = 0; k < n; k++)
    {
        int uk_pos = diagptr[k];
        double ukk = val[uk_pos];
        if (ukk == 0.0)
        {
            throw std::runtime_error("Zero pivot at row " + std::to_string(k));
        }
        int pU_start = uk_pos;
        int pU_end = rowptr[k + 1];

        // iterate for each row that depends on row(k)
        for (int r : lowerPartRowIdx[k])
        {
            // int updateBegin = , updateEnd = ;
            int l = rowptr[r], h = rowptr[r + 1];
            // int updateBegin = -1;
            // find the k position of row(r) colidx
            auto item = lower_bound(colidx + l, colidx + h, k);
            int updateBegin = (item != colidx + h && *item == k) ? item - colidx : -1;

            if (updateBegin == -1) // the element row(r, k) is zero
            {
                continue;
            }

            double Lrk = val[updateBegin] / ukk;
            val[updateBegin] = Lrk;

            // update [updatgeBegin, rowEnd] using row(k)
            for (int p = pU_start + 1; p < pU_end; p++)
            {
                int j = colidx[p];
                auto ret = lower_bound(colidx + l, colidx + h, j);
                if (ret != colidx + h && *ret == j)
                {
                    int pos_rj = ret - colidx;
                    val[pos_rj] -= Lrk * val[p];
                }
            } // loop for update row(r) using row (p)
        } // loop: iterate row k+1 ro n
    }
}

void ilu0_csr_leftlooking_inplace(CSR *A)
{
    // printf("matrix row: %d, matrix col: %d\n", A.n, A.m);
    assert(A->diagptr);
    const int nnz = A->getNnz();
    const int m = A->m;
    const int n = A->n;

    const int *rowptr = A->rowptr;
    const int *colidx = A->colidx;
    const int *diagptr = A->diagptr;
    double *val = A->values;
    // ILU0_INITIAL

    if (m != n)
    {
        throw std::runtime_error("this matrix is not a square matrix");
    }
    std::vector<int> colIdxMapping(n, -1);
    // for each row
    for (int i = 0; i < n; i++)
    {
        // create col idx mapping for this row
        for (int p = rowptr[i]; p < rowptr[i + 1]; p++)
        {

            colIdxMapping[colidx[p]] = p; // get this row nnz element offset by colidx
        }
        for (int k = rowptr[i]; k < rowptr[i + 1]; k++)
        {
            int j = colidx[k]; // the row j depended by row i
            if (j >= i)
                break;                    // processing only lower triangular part, this part depend the depedency between them
            double ujj = val[diagptr[j]]; // the diagonal element of dependency caused by row(i, j)
            assert(fabs(ujj) > 1.0e-16);
            if (fabs(ujj) <= 1.0e-16)
            {
                throw std::runtime_error("Zero pivot encountered at row " + std::to_string(j));
            }
            val[k] /= ujj;
            const double lij = val[k];
            // update row i using the upper triangular part of row j
            for (int p = diagptr[j] + 1; p < rowptr[j + 1]; ++p)
            {
                int col = colidx[p];
                int q = colIdxMapping[col];
                if (q != -1)
                {
                    val[q] -= lij * val[p];
                }
            }
        }

        // reset colIdxMapping for next row update
        for (int p = rowptr[i]; p < rowptr[i + 1]; p++)
        {
            colIdxMapping[colidx[p]] = -1;
        }
    }
}

void ilu0_csr_leftlooking_inplace_levelset(CSR *A)
{
    using SPM::DAG, SPM::LevelSet;
    using SPM::DAG_MAT;
    ILU0_INITIAL

    if (m != n)
    {
        throw std::runtime_error("this matrix is not a square matrix");
    }
    std::vector<int> colIdxMapping(n, -1);

    // construct DAG by lower triangular part of matrix A

    // traverse CSR A to get the non-zero number in lower triangular
    int lower_nnz = 0;
    for (int i = 0; i < m; i++)
    {
        // for(int j = rowptr[i]; j < rowptr[i+1]; j++)
        // {
        lower_nnz += diagptr[i] - rowptr[i] + 1;
        // }
    }
    printf("lower triangular part has %d nnz\n", lower_nnz);

    fflush(stdout);

    DAG *ilu0DAG = new DAG(m, lower_nnz, DAG_MAT::DAG_CSR);
    ilu0DAG->DAG_ptr[0] = 0;
    for (int i = 0; i < m; i++)
    {
        ilu0DAG->DAG_ptr[i + 1] = ilu0DAG->DAG_ptr[i] + diagptr[i] - rowptr[i] + 1;
        std::copy(colidx + rowptr[i], colidx + diagptr[i] + 1, ilu0DAG->DAG_set.begin() + ilu0DAG->DAG_ptr[i]);
    }
    // printf("DAG ptr back is %d\n", ilu0DAG->DAG_ptr.back());
    fflush(stdout);
    assert(ilu0DAG->DAG_ptr.back() == lower_nnz);
    assert(ilu0DAG->DAG_set.back() == m - 1);

    LevelSet *levelset = new LevelSet();
    ilu0DAG->findLevelsPostOrder(levelset);
#ifndef NDEBUG
    // DAG *ilu0CSCDAG = DAG::inverseDAG(*ilu0DAG, true);
    // ilu0CSCDAG->forma

    // ilu0CSCDAG->findLevelsPostOrder(levelset);
    LevelSet *levelsetVf = new LevelSet();
    ilu0DAG->findLevels(levelsetVf);
    bool check = levelset->equal(*levelsetVf);
    fflush(stdout);
    assert(check);
    delete levelsetVf;
#endif

    int nlevels = levelset->getLevels();
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;

    int nthread = omp_get_max_threads();
    int *tmp = MALLOC(int, m *nthread);
    CHECK_POINTER(tmp);
    // printf("nlevels: %d\n", nlevels);

// parallel computing  A \approx L \times U
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *tmpLocal = tmp + tid * m;

        for (int l = 0; l < nlevels; l++) // each level
        {
            int start = level_ptr[l];
            int end = level_ptr[l + 1];
// int chunk = (end - start + nthread - 1) / nthread;
// int my_start = start + tid * chunk;
// int my_end = std::min(my_start + chunk, end);
#pragma omp for schedule(static)
            for (int i = start; i < end; i++) // each row
            {
                std::fill_n(tmpLocal, m, -1);
                int row = permToOrig[i];
                // create colidx mapping
                for (int p = rowptr[row]; p < rowptr[row + 1]; p++)
                {
                    tmpLocal[colidx[p]] = p;
                }

                // update row from left to right
                for (int k = rowptr[row]; k < rowptr[row + 1]; k++)
                {
                    int j = colidx[k];
                    if (j >= row)
                        break;
                    double ujj = val[diagptr[j]]; // dependency row
                    assert(fabs(ujj) > 1.0e-16);
                    if (fabs(ujj) <= 1.0e-16)
                    {
                        throw std::runtime_error("Zero pivot encountered at row " + std::to_string(j));
                    }
                    val[k] /= ujj;
                    const double lij = val[k];
                    for (int p = diagptr[j] + 1; p < rowptr[j + 1]; ++p)
                    {
                        int col = colidx[p];
                        int q = tmpLocal[col];
                        if (q != -1)
                        {
                            val[q] -= lij * val[p];
                        }
                    }
                }
            }
            // #pragma omp barrier
        }
    } // omp parallel

    FREE(tmp);
    delete levelset;
    delete ilu0DAG;
}

void ilu0csr_uplooking_ref(CSR *A)
{
    ILU0_INITIAL
    // if (m != n)
    // {
    //     throw std::runtime_error("this matrix is not a square matrix");
    // }
    for (int i = 0; i < m; i++)
    {
        for (int j = rowptr[i]; j < diagptr[i]; ++j) // processing only lower triangular part
        {
            int col = colidx[j];
            double pivot = val[diagptr[col]];
            assert(fabs(pivot) > 1.0e-16);
            // if (fabs(pivot) < 1.0e-16)
            // {
            //     throw std::runtime_error("Zero pivot encountered at row " + std::to_string(col));
            // }
            double tmp = val[j] /= pivot;

            int k1 = j + 1, k2 = diagptr[col] + 1;
            while (k1 < rowptr[i + 1] && k2 < rowptr[col + 1])
            {
                if (colidx[k1] < colidx[k2])
                    k1++;
                else if (colidx[k1] > colidx[k2])
                    k2++;
                else
                {
                    val[k1] -= tmp * val[k2];
                    ++k1;
                    ++k2;
                }
            } // two pointer scan and update vector
        } // for: from rowptr[row] to diagptr[row], processing only lower triangular part
    } // for: traverse all row
}

// void ilu0csr_uplooking_ref(CSR *A)
// {
//     ILU0_INITIAL
//     if (m != n)
//     {
//         throw std::runtime_error("this matrix is not a square matrix");
//     }
//     for (int i = 0; i < m; i++)
//     {
//         for (int j = rowptr[i]; j < diagptr[i]; ++j) // processing only lower triangular part
//         {
//             int col = colidx[j];
//             double pivot = val[diagptr[col]];
//             assert(fabs(pivot) > 1.0e-16);
//             if (fabs(pivot) < 1.0e-16)
//             {
//                 throw std::runtime_error("Zero pivot encountered at row " + std::to_string(col));
//             }
//             double tmp = val[j] /= pivot;

//             int k1 = j + 1, k2 = diagptr[col] + 1;
//             while (k1 < rowptr[i + 1] && k2 < rowptr[col + 1])
//             {
//                 if (colidx[k1] < colidx[k2])
//                     k1++;
//                 else if (colidx[k1] > colidx[k2])
//                     k2++;
//                 else
//                 {
//                     val[k1] -= tmp * val[k2];
//                     ++k1;
//                     ++k2;
//                 }
//             }   // two pointer scan and update vector
//         }   // for: from rowptr[row] to diagptr[row], processing only lower triangular part
//     }   // for: traverse all row
// }

void ilu0csr_uplooking_ref(CSR *A, double *lu)
{
    ILU0_INITIAL
    // if (m != n)
    // {
    //     throw std::runtime_error("this matrix is not a square matrix");
    // }
    for (int i = 0; i < m; i++)
    {
        for (int j = rowptr[i]; j < diagptr[i]; ++j) // processing only lower triangular part
        {
            int col = colidx[j];
            double pivot = lu[diagptr[col]];
            assert(fabs(pivot) > 1.0e-16);
            // if (fabs(pivot) < 1.0e-16)
            // {
            //     throw std::runtime_error("Zero pivot encountered at row " + std::to_string(col));
            // }
            double tmp = lu[j] /= pivot;

            int k1 = j + 1, k2 = diagptr[col] + 1;
            while (k1 < rowptr[i + 1] && k2 < rowptr[col + 1])
            {
                if (colidx[k1] < colidx[k2])
                    k1++;
                else if (colidx[k1] > colidx[k2])
                    k2++;
                else
                {
                    lu[k1] -= tmp * lu[k2];
                    ++k1;
                    ++k2;
                }
            } // two pointer scan and update vector
        } // for: from rowptr[row] to diagptr[row], processing only lower triangular part
    } // for: traverse all row
}

void ilu0csr_uplooking_levelset(CSR *A)
{
    ILU0_INITIAL
    using SPM::DAG, SPM::LevelSet;
    using SPM::DAG_MAT;

    // if (m != n)
    // {
    //     throw std::runtime_error("this matrix is not a square matrix");
    // }
    std::vector<int> colIdxMapping(n, -1);

    // construct DAG by lower triangular part of matrix A
    // traverse CSR A to get the non-zero number in lower triangular
    int lower_nnz = 0;
    for (int i = 0; i < m; i++)
    {
        lower_nnz += diagptr[i] - rowptr[i] + 1;
    }
    printf("lower triangular part has %d nnz\n", lower_nnz);

    DAG *ilu0DAG = new DAG(m, lower_nnz, DAG_MAT::DAG_CSR);
    ilu0DAG->DAG_ptr[0] = 0;
    for (int i = 0; i < m; i++)
    {
        ilu0DAG->DAG_ptr[i + 1] = ilu0DAG->DAG_ptr[i] + diagptr[i] - rowptr[i] + 1;
        std::copy(colidx + rowptr[i], colidx + diagptr[i] + 1, ilu0DAG->DAG_set.begin() + ilu0DAG->DAG_ptr[i]);
    }
    // printf("DAG ptr back is %d\n", ilu0DAG->DAG_ptr.back());

    assert(ilu0DAG->DAG_ptr.back() == lower_nnz);
    assert(ilu0DAG->DAG_set.back() == m - 1);

    LevelSet *levelset = new LevelSet();
    ilu0DAG->findLevelsPostOrder(levelset);
#ifndef NDEBUG
    LevelSet *levelsetVf = new LevelSet();
    ilu0DAG->findLevels(levelsetVf);
    bool check = levelset->equal(*levelsetVf);
    fflush(stdout);
    assert(check);
    delete levelsetVf;
#endif

    int nlevels = levelset->getLevels();
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;

    int nthread = omp_get_max_threads();
    // int *tmp = MALLOC(int, m *nthread);
    // CHECK_POINTER(tmp);
    // printf("nlevels: %d\n", nlevels);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int l = 0; l < nlevels; l++)
        {
            int start = level_ptr[l];
            int end = level_ptr[l + 1];

#pragma omp for schedule(static)
            for (int i = start; i < end; i++)
            {
                int row = permToOrig[i];
                for (int j = rowptr[row]; j < diagptr[row]; j++) // processing only lower triangular part
                {
                    int col = colidx[j];
                    double pivot = val[diagptr[col]];
                    assert(fabs(pivot) > 1.0e-16);
                    if (fabs(pivot) < 1.0e-16)
                    {
                        throw std::runtime_error("Zero pivot encountered at row " + std::to_string(col));
                    }
                    double tmp = val[j] /= pivot;
                    int k1 = j + 1, k2 = diagptr[col] + 1;
                    while (k1 < rowptr[row + 1] && k2 < rowptr[col + 1])
                    {
                        if (colidx[k1] < colidx[k2])
                            k1++;
                        else if (colidx[k1] > colidx[k2])
                            k2++;
                        else
                        {
                            val[k1] -= tmp * val[k2];
                            k1++;
                            k2++;
                        }
                    } // while: two pointer scan and update vector
                } // for: rowptr[row] to diagptr[row]
            } // for: traverse node in one level
        } // for: each level
    } // for: omp parallel
}

void ilu0csr_uplooking_levelset(const CSR *A, double *lu)
{
    ILU0_INITIAL
    using SPM::DAG, SPM::LevelSet;
    using SPM::DAG_MAT;

    // if (m != n)
    // {
    //     throw std::runtime_error("this matrix is not a square matrix");
    // }
    std::vector<int> colIdxMapping(n, -1);

    // construct DAG by lower triangular part of matrix A
    // traverse CSR A to get the non-zero number in lower triangular
    int lower_nnz = 0;
    for (int i = 0; i < m; i++)
    {
        lower_nnz += diagptr[i] - rowptr[i] + 1;
    }
    printf("lower triangular part has %d nnz\n", lower_nnz);

    DAG *ilu0DAG = new DAG(m, lower_nnz, DAG_MAT::DAG_CSR);
    ilu0DAG->DAG_ptr[0] = 0;
    for (int i = 0; i < m; i++)
    {
        ilu0DAG->DAG_ptr[i + 1] = ilu0DAG->DAG_ptr[i] + diagptr[i] - rowptr[i] + 1;
        std::copy(colidx + rowptr[i], colidx + diagptr[i] + 1, ilu0DAG->DAG_set.begin() + ilu0DAG->DAG_ptr[i]);
    }
    // printf("DAG ptr back is %d\n", ilu0DAG->DAG_ptr.back());

    assert(ilu0DAG->DAG_ptr.back() == lower_nnz);
    assert(ilu0DAG->DAG_set.back() == m - 1);

    LevelSet *levelset = new LevelSet();
    ilu0DAG->findLevelsPostOrder(levelset);
#ifndef NDEBUG
    LevelSet *levelsetVf = new LevelSet();
    ilu0DAG->findLevels(levelsetVf);
    bool check = levelset->equal(*levelsetVf);
    fflush(stdout);
    assert(check);
    delete levelsetVf;
#endif

    int nlevels = levelset->getLevels();
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;

    int nthread = omp_get_max_threads();
    // int *tmp = MALLOC(int, m *nthread);
    // CHECK_POINTER(tmp);
    // printf("nlevels: %d\n", nlevels);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int l = 0; l < nlevels; l++)
        {
            int start = level_ptr[l];
            int end = level_ptr[l + 1];

#pragma omp for schedule(static)
            for (int i = start; i < end; i++)
            {
                int row = permToOrig[i];
                for (int j = rowptr[row]; j < diagptr[row]; j++) // processing only lower triangular part
                {
                    int col = colidx[j];
                    double pivot = lu[diagptr[col]];
                    assert(fabs(pivot) > 1.0e-16);
                    // if (fabs(pivot) < 1.0e-16)
                    // {
                    //     throw std::runtime_error("Zero pivot encountered at row " + std::to_string(col));
                    // }
                    double tmp = lu[j] /= pivot;
                    int k1 = j + 1, k2 = diagptr[col] + 1;
                    while (k1 < rowptr[row + 1] && k2 < rowptr[col + 1])
                    {
                        if (colidx[k1] < colidx[k2])
                            k1++;
                        else if (colidx[k1] > colidx[k2])
                            k2++;
                        else
                        {
                            lu[k1] -= tmp * lu[k2];
                            k1++;
                            k2++;
                        }
                    } // while: two pointer scan and update vector
                } // for: rowptr[row] to diagptr[row]
            } // for: traverse node in one level
        } // for: each level
    } // for: omp parallel
}

void ilu0csr_uplooking_levelset_kernel(const CSR *A, const LevelSet *levelset, double *lu)
{
    ILU0_INITIAL
    int nlevels = levelset->getLevels();
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;
    int nthread = omp_get_max_threads();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int l = 0; l < nlevels; l++)
        {
            int start = level_ptr[l];
            int end = level_ptr[l + 1];

#pragma omp for schedule(static)
            for (int i = start; i < end; i++)
            {
                int row = permToOrig[i];
                for (int j = rowptr[row]; j < diagptr[row]; j++) // processing only lower triangular part
                {
                    int col = colidx[j];
                    double pivot = lu[diagptr[col]];
                    assert(fabs(pivot) > 1.0e-16);
                    // if (fabs(pivot) < 1.0e-16)
                    // {
                    //     throw std::runtime_error("Zero pivot encountered at row " + std::to_string(col));
                    // }
                    double tmp = lu[j] /= pivot;
                    int k1 = j + 1, k2 = diagptr[col] + 1;
                    while (k1 < rowptr[row + 1] && k2 < rowptr[col + 1])
                    {
                        if (colidx[k1] < colidx[k2])
                            k1++;
                        else if (colidx[k1] > colidx[k2])
                            k2++;
                        else
                        {
                            lu[k1] -= tmp * lu[k2];
                            k1++;
                            k2++;
                        }
                    } // while: two pointer scan and update vector
                } // for: rowptr[row] to diagptr[row]
            } // for: traverse node in one level
        } // for: each level
    } // for: omp parallel
}

/**
 * @note: perform ILU0 factorization (A \approx LU) using vertex coarsening method in level set with synchronization point-to-point, and the task has been organized in level-by-level
 * @param csrA: CSR format for sparse matrix A, and the column idx of each row is strictly ascending
 * @param lu: the factorization matrix L and U value. Before this routine running, the data of lu is equal to A.values
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 * @param schdule: the schedule information, such as parents task list, parent number, and finished flag, etc.
 */
void ilu0_p2p_csr_group_merge_no_perm_alloc(const CSR *A, double *lu, const LevelSet *levelset, const int *nthread_per_level,
                                            const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule)
{
    const int *permToOrig = levelset->permToOrig;
    const int *rowptr = A->rowptr;
    const int *colidx = A->colidx;
    const int *diagptr = A->diagptr;
    #ifndef NDEBUG
    std::vector<int> mask(A->m, 0);
    #endif
    // std::vector<std::string> outStr(omp_get_max_threads(), "");

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthread = omp_get_num_threads();

        const int ntasks = task->getTaskNum();
        const int *nparents = schedule->nparents;
        const int *threadBoundaries = task->threadBoundaries.data();
        const int *taskBoundaries = task->taskBoundaries.data();
        const int *threadContToOrigPerm = task->threadContToOrigPerm;

        int nPerthread = (ntasks + nthread - 1) / nthread;
        int nBegin = std::min(nPerthread * tid, ntasks);
        int nEnd = std::min(nPerthread * (tid + 1), ntasks);

        volatile int *taskFinished = schedule->taskFinsished;
        int **parents = schedule->parents;
        memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin) * sizeof(int));

#pragma omp barrier

        for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task)
        {
            ILU0_SCHEDULE_WAIT
            for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; i++)
            {
                int group_idx = threadContToOrigPerm[i];
                for (int k = group_ptr[group_idx]; k < group_ptr[group_idx + 1]; k++)
                {
                    int node = group_set[k];
                    #ifndef NDEBUG
                    mask[node] = 1;
                    #endif
                    for (int j = rowptr[node]; j < diagptr[node]; j++) // processing only lower triangular part
                    {
                        int col = colidx[j];
                        double pivot = lu[diagptr[col]];
                        assert(fabs(pivot) > 1.0e-16);
                        // if (fabs(pivot) < 1.0e-16)
                        // {
                        //     throw std::runtime_error("Zero pivot encountered at row " + std::to_string(col));
                        // }
                        double tmp = lu[j] /= pivot;
                        int k1 = j + 1, k2 = diagptr[col] + 1;
                        while (k1 < rowptr[node + 1] && k2 < rowptr[col + 1])
                        {
                            if (colidx[k1] < colidx[k2])
                                k1++;
                            else if (colidx[k1] > colidx[k2])
                                k2++;
                            else
                            {
                                lu[k1] -= tmp * lu[k2];
                                k1++;
                                k2++;
                            }
                        } // while: two pointer scan and update vector
                    } // for: rowptr[row] to diagptr[row]
                }
            }
            ILU0_SCHEDULE_NOTIFY
        }
    }

#ifndef NDEBUG
    for (auto &item : mask)
    {
        assert(item == 1);
    }
#endif
}