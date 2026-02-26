
#include "omp.h"
#include "IC.hpp"
#include "Merge.hpp"

inline double sparse_dot_product(int l1, int u1, int l2, int u2, const int *indices, const double *data)
{
    double result = 0.0;
    while (l1 < u1 && l2 < u2)
    {
        if (indices[l1] == indices[l2]) // matching column?
        {
            assert(std::isnan(data[l1]) == false);
            assert(std::isnan(data[l2]) == false);
            assert(std::isinf(data[l1]) == false);
            assert(std::isinf(data[l2]) == false);
            result += data[l1++] * data[l2++];
        }
        else if (indices[l1] < indices[l2]) // else proceed until we find matching columns
        {
            l1++;
        }
        else
        {
            l2++;
        }
    }
    return result;
}

//=============================== Up looking Looking ==============================
void spic0_csr_uL_serial(const CSR *A, double *lu)
{
    const int n = A->n;
    const int *rowptr = A->rowptr;
    const int *colidx = A->colidx;
    const int *diagptr = A->diagptr;
    const double *values = A->values;

    // bool finish_flag = true;
    for (int i = 0; i < n; ++i)
    {
        for (int k = rowptr[i]; k < diagptr[i]; ++k)
        {
            const int j = colidx[k]; // column
            double dp = sparse_dot_product(
                rowptr[i], diagptr[i], // i-th row minus diagonal
                rowptr[j], diagptr[j], // j-th row minus diagonal
                colidx, lu);

            const double A_ij = values[k];

            // below diagonal?
            const double L_jj = lu[diagptr[j]]; // diagonal is last entry of j-th row
            // assert(fabs(L_jj) > 1.0 - 16);
            //    /
            assert(L_jj > 1.0e-10);
            assert(std::isnan(dp) == false);
            assert(std::isnan(A_ij) == false);
            assert(std::isnan(A_ij - dp) == false);
            assert(std::isnan(L_jj) == false);

            assert(std::isinf(dp) == false);
            assert(std::isinf(A_ij) == false);
            assert(std::isinf(A_ij - dp) == false);
            assert(std::isinf(L_jj) == false);
            lu[k] = (A_ij - dp) / L_jj;
            assert(std::isnan(lu[k]) == false);
            assert(std::isinf(lu[k]) == false);
            // }/
        }

        // update diagonal element
        double dv = sparse_dot_product(
            rowptr[i], diagptr[i], // i-th row minus diagonal
            rowptr[i], diagptr[i], // j-th row minus diagonal
            colidx, lu);
        const double A_ii = values[diagptr[i]];
        // assert(A_ii - dv >= 0);
        if (A_ii - dv > 0)
        {
            assert(std::isnan(A_ii - dv) == false);
            assert(std::isinf(A_ii - dv) == false);
            lu[diagptr[i]] = std::sqrt(A_ii - dv);
            assert(std::isnan(lu[diagptr[i]]) == false);
            assert(std::isinf(lu[diagptr[i]]) == false);
        }
    }
}

//=============================== Up looking Looking ==============================
void spic0_csr_uL_levelset(const CSR *A, double *lu)
{

    using SPM::DAG, SPM::LevelSet;
    using SPM::DAG_MAT;

    assert(A->diagptr);
    const int nnz = A->getNnz();
    const int m = A->m;
    const int n = A->n;
    const int *rowptr = A->rowptr;
    const int *colidx = A->colidx;
    const int *diagptr = A->diagptr;
    double *values = A->values;

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
        lower_nnz += diagptr[i] - rowptr[i] + 1;
    }
    // printf("lower triangular part has %d nnz\n", lower_nnz);

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

    // *************** IC0 LevelSet implementation *********************
    int nlevels = levelset->getLevels();
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;
    int nthread = omp_get_max_threads();

    // printf("nlevels: %d\n", nlevels);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int l = 0; l < nlevels; l++)
        {
            int start = level_ptr[l];
            int end = level_ptr[l + 1];

#pragma omp for schedule(static)
            for (int s = start; s < end; s++)
            {
                int i = permToOrig[s];
                for (int k = rowptr[i]; k < diagptr[i]; k++) // processing only lower triangular part
                {
                    const int j = colidx[k]; // column
                    double dp = sparse_dot_product(
                        rowptr[i], diagptr[i], // i-th row minus diagonal
                        rowptr[j], diagptr[j], // j-th row minus diagonal
                        colidx, lu);
                    const double A_ij = values[k];
                    // below diagonal?
                    const double L_jj = lu[diagptr[j]]; // diagonal is last entry of j-th row
                    assert(L_jj > 1.0e-16);
                    assert(std::isnan(dp) == false);
                    assert(std::isnan(A_ij) == false);
                    assert(std::isnan(A_ij - dp) == false);
                    assert(std::isnan(L_jj) == false);

                    assert(std::isinf(dp) == false);
                    assert(std::isinf(A_ij) == false);
                    assert(std::isinf(A_ij - dp) == false);
                    assert(std::isinf(L_jj) == false);
                    lu[k] = (A_ij - dp) / L_jj;
                    assert(std::isnan(lu[k]) == false);
                    assert(std::isinf(lu[k]) == false);
                } // for: rowptr[row] to diagptr[row]
                // update diagonal element
                double dv = sparse_dot_product(
                    rowptr[i], diagptr[i], // i-th row minus diagonal
                    rowptr[i], diagptr[i], // j-th row minus diagonal
                    colidx, lu);
                const double A_ii = values[diagptr[i]];
                if (A_ii - dv > 0)
                {
                    assert(std::isnan(A_ii - dv) == false);
                    assert(std::isinf(A_ii - dv) == false);
                    lu[diagptr[i]] = std::sqrt(A_ii - dv);
                    assert(std::isnan(lu[diagptr[i]]) == false);
                    assert(std::isinf(lu[diagptr[i]]) == false);
                }

            } // for: traverse node in one level
        } // for: each level
    } // for: omp parallel
}

//=============================== Up looking Looking ==============================
void spic0_csr_uL_levelset_kernel(const CSR *A, double *lu, const LevelSet *levelset)
{

    assert(A->diagptr);
    const int nnz = A->getNnz();
    const int m = A->m;
    const int n = A->n;
    const int *rowptr = A->rowptr;
    const int *colidx = A->colidx;
    const int *diagptr = A->diagptr;
    double *values = A->values;

    // *************** IC0 LevelSet implementation *********************
    int nlevels = levelset->getLevels();
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;
    int nthread = omp_get_max_threads();

    // printf("nlevels: %d\n", nlevels);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int l = 0; l < nlevels; l++)
        {
            int start = level_ptr[l];
            int end = level_ptr[l + 1];

#pragma omp for schedule(static)
            for (int s = start; s < end; s++)
            {
                int i = permToOrig[s];
                for (int k = rowptr[i]; k < diagptr[i]; k++) // processing only lower triangular part
                {
                    const int j = colidx[k]; // column
                    double dp = sparse_dot_product(
                        rowptr[i], diagptr[i], // i-th row minus diagonal
                        rowptr[j], diagptr[j], // j-th row minus diagonal
                        colidx, lu);
                    const double A_ij = values[k];
                    // below diagonal?
                    const double L_jj = lu[diagptr[j]]; // diagonal is last entry of j-th row
                    assert(L_jj > 1.0e-10);
                    assert(std::isnan(dp) == false);
                    assert(std::isnan(A_ij) == false);
                    assert(std::isnan(A_ij - dp) == false);
                    assert(std::isnan(L_jj) == false);

                    assert(std::isinf(dp) == false);
                    assert(std::isinf(A_ij) == false);
                    assert(std::isinf(A_ij - dp) == false);
                    assert(std::isinf(L_jj) == false);
                    lu[k] = (A_ij - dp) / L_jj;
                    assert(std::isnan(lu[k]) == false);
                    assert(std::isinf(lu[k]) == false);
                } // for: rowptr[row] to diagptr[row]
                // update diagonal element
                double dv = sparse_dot_product(
                    rowptr[i], diagptr[i], // i-th row minus diagonal
                    rowptr[i], diagptr[i], // j-th row minus diagonal
                    colidx, lu);
                const double A_ii = values[diagptr[i]];
                if (A_ii - dv > 0)
                {
                    assert(std::isnan(A_ii - dv) == false);
                    assert(std::isinf(A_ii - dv) == false);
                    lu[diagptr[i]] = std::sqrt(A_ii - dv);
                    assert(std::isnan(lu[diagptr[i]]) == false);
                    assert(std::isinf(lu[diagptr[i]]) == false);
                }
            } // for: traverse node in one level
        } // for: each level
    } // for: omp parallel
}

/**
 * @note: perform IC0 factorization (A \approx LL^T) using vertex coarsening method in level set with synchronization point-to-point,
 * and the task has been organized in level-by-level. After the program finishes, the resulting matrix L will be stored in the lower
 * triangular part of lu, and the matrix L^T can be obtained by scanning once.
 * @param csrA: CSR format for sparse matrix A, and the column idx of each row is strictly ascending
 * @param lu: the factorization matrix L and L^T value. Before this routine running, the data of lu is equal to A.values.
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 * @param schdule: the schedule information, such as parents task list, parent number, and finished flag, etc.
 */
void spic0_csr_uL_p2p_group_merge_no_perm(const CSR *A, double *lu, const LevelSet *levelset, const int *nthread_per_level,
                                          const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule)
{
    const int *permToOrig = levelset->permToOrig;
    const int *rowptr = A->rowptr;
    const int *colidx = A->colidx;
    const int *diagptr = A->diagptr;
#ifndef NDEBUG
    std::vector<int> mask(A->m, 0);
#endif

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

        for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; task++)
        {
            IC0_SCHEDULE_WAIT;
            for (int t = taskBoundaries[task]; t < taskBoundaries[task + 1]; t++)
            {
                int g = threadContToOrigPerm[t];
                for (int gb = group_ptr[g]; gb < group_ptr[g + 1]; gb++)
                {
                    const int i = group_set[gb]; // node i
#ifndef NDEBUG
                    mask[i] = 1;
#endif
                    for (int k = rowptr[i]; k < diagptr[i]; k++)
                    {
                        const int j = colidx[k]; // column
                        double dp = sparse_dot_product(
                            rowptr[i], diagptr[i], // i-th row minus diagonal
                            rowptr[j], diagptr[j], // j-th row minus diagonal
                            colidx, lu);
                        const double A_ij = lu[k];
                        // below diagonal
                        const double L_jj = lu[diagptr[j]]; // diagonal element
                        assert(L_jj > 1.0e-10);
                        assert(std::isnan(dp) == false);
                        assert(std::isnan(A_ij) == false);
                        assert(std::isnan(A_ij - dp) == false);
                        assert(std::isnan(L_jj) == false);

                        assert(std::isinf(dp) == false);
                        assert(std::isinf(A_ij) == false);
                        assert(std::isinf(A_ij - dp) == false);
                        assert(std::isinf(L_jj) == false);
                        lu[k] = (A_ij - dp) / L_jj;
                        assert(std::isnan(lu[k]) == false);
                        assert(std::isinf(lu[k]) == false);
                    } // for: rowptr[i] to diagptr[i]

                    // update diagonal element
                    double dv = sparse_dot_product(
                        rowptr[i], diagptr[i],
                        rowptr[i], diagptr[i],
                        colidx, lu);
                    const double A_ii = lu[diagptr[i]];
                    if (A_ii - dv > 0)
                    {
                        assert(std::isnan(A_ii - dv) == false);
                        assert(std::isinf(A_ii - dv) == false);
                        lu[diagptr[i]] = std::sqrt(A_ii - dv);
                        assert(std::isnan(lu[diagptr[i]]) == false);
                        assert(std::isinf(lu[diagptr[i]]) == false);
                    }
                } // for: traverse each node in a group
            } // for: iterate in one task

            IC0_SCHEDULE_NOTIFY;
            
        } // for: traverse each task of one thread
    } // omp parallel

#ifndef NDEBUG
    for (auto &item : mask)
    {
        assert(item == 1);
    }
#endif
}


/**
 * @note: perform IC0 factorization (A \approx LL^T) using vertex coarsening method in level set with synchronization point-to-point,
 * and the task has been organized in level-by-level. After the program finishes, the resulting matrix L will be stored in the lower
 * triangular part of lu, and the matrix L^T can be obtained by scanning once.
 * @param csrA: CSR format for sparse matrix A, and the column idx of each row is strictly ascending
 * @param lu: the factorization matrix L and L^T value. Before this routine running, the data of lu is equal to A.values.
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 * @param schdule: the schedule information, such as parents task list, parent number, and finished flag, etc.
 */
void spic0_csr_uL_p2p_group_merge_perm(const CSR *A, double *lu, const LevelSet *levelset, const int *nthread_per_level,
                                          const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule)
{
    const int *permToOrig = levelset->permToOrig;
    const int *rowptr = A->rowptr;
    const int *colidx = A->colidx;
    const int *diagptr = A->diagptr;
#ifndef NDEBUG
    std::vector<int> mask(A->m, 0);
#endif

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

        for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; task++)
        {
            IC0_SCHEDULE_WAIT;
            for (int t = taskBoundaries[task]; t < taskBoundaries[task + 1]; t++)
            {
                // int g = threadContToOrigPerm[t];
                int g = t;
                for (int gb = group_ptr[g]; gb < group_ptr[g + 1]; gb++)
                {
                    // const int i = group_set[gb]; // node i
                    int i = gb; // node i
#ifndef NDEBUG
                    mask[i] = 1;
#endif
                    for (int k = rowptr[i]; k < diagptr[i]; k++)
                    {
                        const int j = colidx[k]; // column
                        double dp = sparse_dot_product(
                            rowptr[i], diagptr[i], // i-th row minus diagonal
                            rowptr[j], diagptr[j], // j-th row minus diagonal
                            colidx, lu);
                        const double A_ij = lu[k];
                        // below diagonal
                        const double L_jj = lu[diagptr[j]]; // diagonal element
                        assert(L_jj > 1.0e-10);
                        assert(std::isnan(dp) == false);
                        assert(std::isnan(A_ij) == false);
                        assert(std::isnan(A_ij - dp) == false);
                        assert(std::isnan(L_jj) == false);

                        assert(std::isinf(dp) == false);
                        assert(std::isinf(A_ij) == false);
                        assert(std::isinf(A_ij - dp) == false);
                        assert(std::isinf(L_jj) == false);
                        lu[k] = (A_ij - dp) / L_jj;
                        assert(std::isnan(lu[k]) == false);
                        assert(std::isinf(lu[k]) == false);
                    } // for: rowptr[i] to diagptr[i]

                    // update diagonal element
                    double dv = sparse_dot_product(
                        rowptr[i], diagptr[i],
                        rowptr[i], diagptr[i],
                        colidx, lu);
                    const double A_ii = lu[diagptr[i]];
                    if (A_ii - dv > 0)
                    {
                        assert(std::isnan(A_ii - dv) == false);
                        assert(std::isinf(A_ii - dv) == false);
                        lu[diagptr[i]] = std::sqrt(A_ii - dv);
                        assert(std::isnan(lu[diagptr[i]]) == false);
                        assert(std::isinf(lu[diagptr[i]]) == false);
                    }
                } // for: traverse each node in a group
            } // for: iterate in one task

            IC0_SCHEDULE_NOTIFY;
            
        } // for: traverse each task of one thread
    } // omp parallel

#ifndef NDEBUG
    for (auto &item : mask)
    {
        assert(item == 1);
    }
#endif
}
