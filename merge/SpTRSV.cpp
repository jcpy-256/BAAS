#include <iostream>
#include <cstdio>
#include <cstring>
#include <omp.h>

#include "SpTRSV.hpp"
// #include "CSR.hpp"
#include "SPM.hpp"
#include "papiProfiling.hpp"
// #include "test.hpp"

#define ADJUST_FOR_BASE                     \
    int base = A.getBase();                 \
    const int *rowptr = A.rowptr - base;    \
    const int *colidx = A.colidx - base;    \
    const double *values = A.values - base; \
    x -= base;                              \
    y -= base;

using namespace SPM;
/**
 * Reference sequential sparse triangular solver: Ay = b
 */
void sptrsv_serial_csr(const CSR &A, double x[], const double y[])
{
    // printf("nrow:%d,ncol:%d, nnz: %d\n", A.m, A.n, A.getNnz());
    ADJUST_FOR_BASE;
    for (int i = base; i < A.m + base; ++i)
    {
        double sum = 0.0;
        // double sum = ;
        // int j = rowptr[i];
        for (int j = rowptr[i]; j < rowptr[i + 1] - 1; j++)
        {
            sum += values[j] * x[colidx[j]];
        }
        x[i] = (y[i] - sum) / values[rowptr[i + 1] - 1];
    } // for each row
}

void sptrsv_level_csr_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset)
{
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;
    // double *x_perm = MALLOC(double, csrA->n);

    int *rowptr = csrA->rowptr;
    int *colidx = csrA->colidx;
    double *values = csrA->values;
    int nlevel = levelset->getLevels();
    // #pragma omp parallel
    //     {
    // int tid = omp_get_thread_num();
    // int nthread = omp_get_max_threads();

    for (int l = 0; l < nlevel; l++)
    {
#pragma omp parallel for
        for (int i = level_ptr[l]; i < level_ptr[l + 1]; i++)
        {
            int row = permToOrig[i];
            // double sum = ;
            double sum = 0.0;
            for (int j = rowptr[row]; j < rowptr[row + 1] - 1; j++)
            {
                sum += x[colidx[j]] * values[j];
            }
            x[row] = (b[row] - sum) / values[rowptr[row + 1] - 1];
        }
#pragma omp barrier
    }
    // #pragma omp barrier
    // // permuate to origin after solve
    // #pragma omp for
    // for(int i=0; i < csrA->n; i++)
    // {
    //     x[permToOrig[i]] = x_perm[i];
    // }
    // }
}

// level set SpTRSV and the matrix has been permuate by level
void sptrsv_level_csr(const CSR *csrA, double *&x, const double *b, const LevelSet *levelset)
{
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;
    // double *x_perm = MALLOC(double, csrA->n);

    int *rowptr = csrA->rowptr;
    int *colidx = csrA->colidx;
    double *values = csrA->values;
    int nlevel = levelset->getLevels();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthread = omp_get_max_threads();

        for (int l = 0; l < nlevel; l++)
        {
            // int row_beg = level_ptr[l];
            // int row_end = level_ptr[l + 1];
// a permuated matrix, level_ptr denotes the range of row index.
#pragma omp for schedule(auto)
            for (int i = level_ptr[l]; i < level_ptr[l + 1]; i++)
            {
                double sum = 0;
                for (int j = rowptr[i]; j < rowptr[i + 1] - 1; j++)
                {
                    sum += x[colidx[j]] * values[j];
                }
                x[i] = (b[i] - sum) / values[rowptr[i + 1] - 1];
            }
        }
        // #pragma omp barrier
        // permuate to origin after solve
        // #pragma omp for
        //         for (int i = 0; i < csrA->n; i++)
        //         {
        //             x[permToOrig[i]] = x_perm[i];
        //         }
    }

    // FREE(x_perm);
}

void sptrsv_level_csr_merge_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level)
{
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;

    int *rowptr = csrA->rowptr;
    int *colidx = csrA->colidx;
    double *values = csrA->values;
    int nlevel = levelset->getLevels();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthread = omp_get_max_threads();
        // printf("tid: %d, nthread: %d\n", tid, nthread);

        for (int l = 0; l < nlevel; l++)
        {
            int exec_thread = nthread_per_level[l];
            if (tid < exec_thread)
            {
                int start = level_ptr[l];
                int end = level_ptr[l + 1];
                int chunk = (end - start + exec_thread - 1) / exec_thread;
                int my_start = start + tid * chunk;
                int my_end = std::min(my_start + chunk, end);
                for (int i = my_start; i < my_end; i++)
                {
                    int row = permToOrig[i];
                    double sum = 0;
                    for (int j = rowptr[row]; j < rowptr[row + 1] - 1; j++)
                    {
                        sum += x[colidx[j]] * values[j];
                    }
                    x[row] = (b[row] - sum) / values[rowptr[row + 1] - 1];
                }
            }
#pragma omp barrier
        }
    }
}

void sptrsv_level_csr_group_merge_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level, const int *group_ptr, const int *group_set)
{
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;

    int *rowptr = csrA->rowptr;
    int *colidx = csrA->colidx;
    double *values = csrA->values;
    int nlevel = levelset->getLevels();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthread = omp_get_max_threads();

        for (int l = 0; l < nlevel; l++)
        {
            int exec_thread = nthread_per_level[l];
            if (tid < exec_thread)
            {
                int start = level_ptr[l];
                int end = level_ptr[l + 1];
                int chunk = (end - start + exec_thread - 1) / exec_thread;
                int my_start = start + tid * chunk;
                int my_end = std::min(my_start + chunk, end);
                for (int i = my_start; i < my_end; i++)
                {
                    int group_idx = permToOrig[i];
                    for (int k = group_ptr[group_idx]; k < group_ptr[group_idx + 1]; k++)
                    {
                        int node = group_set[k];
                        double sum = 0;
                        for (int j = rowptr[node]; j < rowptr[node + 1] - 1; j++)
                        {
                            sum += x[colidx[j]] * values[j];
                        }
                        x[node] = (b[node] - sum) / values[rowptr[node + 1] - 1];
                    }
                }
            }
#pragma omp barrier
        }
    }
}

// void sptrsv_serial_csr_group_merge_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level, const int *group_ptr, const int *group_set)
// {
//     int *level_ptr = levelset->level_ptr;
//     int *permToOrig = levelset->permToOrig;

//     int ngroup = levelset->getNodeNum();
//     std::vector<int> group_visited(ngroup, 0);

//     int *rowptr = csrA->rowptr;
//     int *colidx = csrA->colidx;
//     double *values = csrA->values;
//     int nlevel = levelset->getLevels();
//     std::vector<int> mask(csrA->n, 0);
//     int idx = 0;
//     int nnz = 0;
//     int group_reduction = 0;
//     int levels = 0;
//     // reduction(+ : idx, nnz, group_reduction, levels)
// #pragma omp parallel reduction(+ : idx, group_reduction, levels)
//     {
//         int tid = omp_get_thread_num();
//         int nthread = omp_get_max_threads();
//         // printf("tid: %d, nthread: %d\n", tid, nthread);

//         for (int l = 0; l < nlevel; l++)
//         {

//             int exec_thread = nthread_per_level[l];
//             if (tid < exec_thread)
//             // for(int tid =exec_thread-1; tid >= 0 ; tid--)
//             {
//                 levels++;
//                 int start = level_ptr[l];
//                 int end = level_ptr[l + 1];
//                 int chunk = (end - start + exec_thread - 1) / exec_thread;
//                 int my_start = start + tid * chunk;
//                 int my_end = std::min(my_start + chunk, end);
//                 for (int i = my_start; i < my_end; i++)
//                 // for (int i = start; i < end; i++)
//                 {
//                     int group_idx = permToOrig[i];
//                     group_visited[group_idx]++;
//                     group_reduction++;
//                     for (int k = group_ptr[group_idx]; k < group_ptr[group_idx + 1]; k++)
//                     {
//                         int node = group_set[k];
// #pragma omp critical
//                         {
//                             mask[node]++;
//                         }
//                         idx++;

//                         // // #pragma omp atomic
//                         // nnz++;
//                         // // capture

//                         double sum = 0;
//                         for (int j = rowptr[node]; j < rowptr[node + 1] - 1; j++)
//                         {
//                             sum += x[colidx[j]] * values[j];
//                         }
//                         x[node] = (b[node] - sum) / values[rowptr[node + 1] - 1];
//                     }
//                 }
//             }
// #pragma omp barrier
//         }
//     }
//     printf("idx num: %d\n", idx);
//     //  printf("nnz num: %d", nnz);
//     printf("level num: %d\n", levels);
//     printf("group num: %d\n", group_reduction);
//     int c = 0;
//     for (int i = 0; i < ngroup; i++)
//     {
//         int item = group_visited[i];
//         if (item == 1)
//         {
//             c++;
//         }
//         else
//         {
//             printf("group %d is visited %d num \n", i, item);
//         }
//     }
//     printf("group num through group visited: %d\n", c);

//     int node_num = 0;
//     for (int i = 0; i < csrA->n; i++)
//     {
//         if (mask[i] == 0)
//         {
//             printf("node: %d is not calculate\n", i);
//         }
//     }
//     exit(2);
//     // printf("visited node num")
//     // bool flag = false;
//     // for(int i=0; i < csrA->n; i ++)
//     // {
//     //     if(mask[i] == false)
//     //     {
//     //         printf("node: %d is not calculate\n", i);
//     //         // assert(true);
//     //         fflush(stdout);
//     //         exit(2);
//     //     }

//     // }
//     // if(flag)
//     // {
//     //     // exit(2);

//     // }
// }

/*
 * It is left looking Sparse Triangular Solve
 * @param n Number of iterations or node
 * @param Lp the pointer array in CSC version
 * @param Li the index array in CSC version
 * @param Lx the value array in CSC version
 * @return x the output
 * @param levels number of levels in the DAg
 * @param levelPtr the pointer array in CSC format
 * that point to starting and ending point of nodes in a level
 * @param LevelSet the array that store nodes sorted based on their level
 * @param groupPtr the array pointer for groups
 * @param groupSet the array set for groups. Nodes are sorted based on their group
 */
void sptrsv_csr_group_levelset(int *Lp, int *Li, double *Lx, double *x, double *y,
                               int level_no, int *level_ptr, int *level_set,
                               int *groupPtr, int *groupSet)
{
#pragma omp parallel
    {
        for (int i1 = 0; i1 < level_no; ++i1)
        {
#pragma omp for schedule(auto)
            for (int j1 = level_ptr[i1]; j1 < level_ptr[i1 + 1]; ++j1)
            {
                int group_idx = level_set[j1];
                for (int k = groupPtr[group_idx]; k < groupPtr[group_idx + 1]; ++k)
                {
                    int i = groupSet[k];
                    double sum = 0;
                    for (int j = Lp[i]; j < Lp[i + 1] - 1; j++)
                    {
                        sum += Lx[j] * x[Li[j]];
                    }
                    x[i] = (y[i] - sum) / Lx[Lp[i + 1] - 1];
                }
            }
        }
    }
}

/**
 * @note: solve the SpTRSV using vertex coarsening method in level set, and the task has been organized in level-by-level
 * @param csrA: CSR format for sparse lower triangular matrix
 * @param x: the solution vector x
 * @param b: single right-hand vector
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 */
void sptrsv_level_csr_group_merge_no_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level, const int *group_ptr, const int *group_set, const Merge::Task *task)
{
    int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;
    const int *threadBoundaries = task->threadBoundaries.data();
    const int *taskBoundaries = task->taskBoundaries.data();
    const int *threadContToOrigPerm = task->threadContToOrigPerm;

    int *rowptr = csrA->rowptr;
    int *colidx = csrA->colidx;
    double *values = csrA->values;
    int nlevel = levelset->getLevels();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthread = omp_get_max_threads();
        int taskBeg = threadBoundaries[tid];
        int taskEnd = threadBoundaries[tid + 1];

        for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; task++)
        {
            for (int item = taskBoundaries[task]; item < taskBoundaries[task + 1]; item++)
            {
                int group_idx = threadContToOrigPerm[item];
                for (int k = group_ptr[group_idx]; k < group_ptr[group_idx + 1]; k++)
                {
                    int node = group_set[k];
                    double sum = 0;
                    for (int j = rowptr[node]; j < rowptr[node + 1] - 1; j++)
                    {
                        sum += x[colidx[j]] * values[j];
                    }
                    x[node] = (b[node] - sum) / values[rowptr[node + 1] - 1];
                } // for each node
            } // for each group
#pragma omp barrier
        } // for each task (level)
    } // parallel omp
}

void sptrsv_p2p_csr_group_merge_no_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                              const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule)
{
    // int *level_ptr = levelset->level_ptr;
    // int *permToOrig = levelset->permToOrig;

    int *rowptr = csrA->rowptr;
    int *colidx = csrA->colidx;
    double *values = csrA->values;
    // std::vector<std::string> outStr(omp_get_max_threads(), "");
    // int nlevel = levelset->getLevels();

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
        // Profiler::PAPIOMPCacheProfiler profInstance;
        // profInstance.start();

        for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task)
        {
            SpMP_SCHEDULE_WAIT;
            for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; i++)
            {
                int group_idx = threadContToOrigPerm[i];
                for (int k = group_ptr[group_idx]; k < group_ptr[group_idx + 1]; k++)
                {
                    int node = group_set[k];
                    double sum = 0;
                    for (int j = rowptr[node]; j < rowptr[node + 1] - 1; j++)
                    {
                        sum += x[colidx[j]] * values[j];
                    }
                    x[node] = (b[node] - sum) / values[rowptr[node + 1] - 1];
                } // for each node
            }
            SPMP_SCHEDULE_NOTIFY;
        }

        // profInstance.stop();
    }
}

void sptrsv_p2p_csr_group_merge_perm_alloc(const CSR *csrAPerm, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                           const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schdule)
{
    // int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;

    int *rowptr = csrAPerm->rowptr;
    int *colidx = csrAPerm->colidx;
    double *values = csrAPerm->values;
    // std::vector<int> arrived(csrAPerm->n, 0);
    // int nlevel = levelset->getLevels();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthread = omp_get_max_threads();
        const int ntasks = task->getTaskNum();
        const int *nparents = schdule->nparents;
        const int *threadBoundaries = task->threadBoundaries.data();
        const int *taskBoundaries = task->taskBoundaries.data();
        const int *threadContToOrigPerm = task->threadContToOrigPerm;
        const int *origToThreadContPerm = task->origToThreadContPerm;

        int nPerthread = (ntasks + nthread - 1) / nthread;
        int nBegin = std::min(nPerthread * tid, ntasks);
        int nEnd = std::min(nPerthread * (tid + 1), ntasks);

        volatile int *taskFinished = schdule->taskFinsished;
        int **parents = schdule->parents;

        memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin) * sizeof(int));
#pragma omp barrier
        // Profiler::PAPIOMPCacheProfiler profInstance;
        // profInstance.start();

        for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task)
        {
            SpMP_SCHEDULE_WAIT;
            for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; i++)
            {
                // int group_idx = threadContToOrigPerm[i];
                int group_idx = i;
                for (int k = group_ptr[group_idx]; k < group_ptr[group_idx + 1]; k++)
                {
                    // int node = group_set[k];
                    // int node = k;
                    double sum = 0;
                    for (int j = rowptr[k]; j < rowptr[k + 1] - 1; j++)
                    {
                        sum += x[colidx[j]] * values[j];
                    }
                    // arrived[group_set[k]] = 1;
                    x[k] = (b[k] - sum) / values[rowptr[k + 1] - 1];
                } // for each node
            }
            SPMP_SCHEDULE_NOTIFY;
        }

// #pragma omp barrier

        // profInstance.stop();
    }
}

// ******************************************** the following serivces for Solver Test *******************************************

/**
 * @note: solve Ay = b, and the matrix A is a lower triangular matrix, which elements in each row is ordered by column idx. Notably, the
 * diagonal element is not separately stored in diag array, but in values array
 * @param A: the lower triangular matrix
 * @param y: the solution vector
 * @param b: the right-hands vector
 */
void lower_csr_trsv_serial(const CSR &A, double *y, const double *b)
{
    int base = A.getBase();
    const int *rowptr = A.rowptr - base;
    const int *colidx = A.colidx - base;
    const double *values = A.values - base;
    const double *idiag = A.idiag - base;
    y -= base;
    b -= base;
    for (int i = base; i < A.m + base; ++i)
    {
        double sum = 0.0;
        for (int j = rowptr[i]; j < rowptr[i + 1] - 1; ++j)
        {
            sum += values[j] * y[colidx[j]];
        }
        y[i] = (b[i] - sum) / values[rowptr[i + 1] - 1];
    } // for each row
}

/**
 * @note: solve Ay = b, and the matrix A is a upper triangular matrix, which elements in each row is ordered by column idx. Notably, the
 * diagonal element is not separately stored in diag array, but in values array
 * @param A: the upper triangular matrix
 * @param y: the solution vector
 * @param b: the right-hands vector
 */
void upper_csr_trsv_serial(const CSR &A, double *y, const double *b)
{
    int base = A.getBase();
    const int *rowptr = A.rowptr - base;
    const int *colidx = A.colidx - base;
    const double *values = A.values - base;
    const double *idiag = A.idiag - base;
    y -= base;
    b -= base;

    for (int i = A.m - 1 + base; i >= base; --i)
    {
        double sum = 0.0;
        for (int j = rowptr[i] + 1; j < rowptr[i + 1]; ++j)
        {
            sum += values[j] * y[colidx[j]];
        }
        y[i] = (b[i] - sum) / values[rowptr[i]];
    } // for each row
}

void sptrsv_backward_p2p_csr_group_merge_no_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                                       const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule)
{
    // int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;

    int *rowptr = csrA->rowptr;
    int *colidx = csrA->colidx;
    double *values = csrA->values;
    // std::vector<std::string> outStr(omp_get_max_threads(), "");
    // int nlevel = levelset->getLevels();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthread = omp_get_num_threads();
        const int ntasks = task->getTaskNum();
        const int *nparents = schedule->nparentsBackward;
        const int *threadBoundaries = task->threadBoundaries.data();
        const int *taskBoundaries = task->taskBoundaries.data();
        const int *threadContToOrigPerm = task->threadContToOrigPerm;

        int nPerthread = (ntasks + nthread - 1) / nthread;
        int nBegin = std::min(nPerthread * tid, ntasks);
        int nEnd = std::min(nPerthread * (tid + 1), ntasks);

        volatile int *taskFinished = schedule->taskFinsished;
        int **parents = schedule->parentsBackward;
        memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin) * sizeof(int));
        //      printf("tid: %d\n", tid);
        // fflush(stdout);
#pragma omp barrier
        // Profiler::PAPIOMPCacheProfiler profInstance;
        // profInstance.start();

        for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task)
        {
            SpMP_SCHEDULE_WAIT;
            for (int i = taskBoundaries[task+1] - 1; i >= taskBoundaries[task]; i--)
            {
                int group_idx = threadContToOrigPerm[i];
                for (int k = group_ptr[group_idx + 1] - 1; k >= group_ptr[group_idx]; k--)
                {
                    int node = group_set[k];
                    double sum = 0;
                    // the first this diagonal elment
                    for (int j = rowptr[node + 1]- 1; j > rowptr[node]; j--)
                    {
                        sum += x[colidx[j]] * values[j];
                    }
                    x[node] = (b[node] - sum) / values[rowptr[node]];
                } // for each node
                // printf("thread idx: %d, group idx: %d\n", tid, group_idx);
            }
            SPMP_SCHEDULE_NOTIFY
        }

        // profInstance.stop();
    }
}


void sptrsv_backward_p2p_csr_group_merge_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                                       const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule)
{
    // int *level_ptr = levelset->level_ptr;
    int *permToOrig = levelset->permToOrig;

    int *rowptr = csrA->rowptr;
    int *colidx = csrA->colidx;
    double *values = csrA->values;
    // std::vector<std::string> outStr(omp_get_max_threads(), "");
    // int nlevel = levelset->getLevels();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthread = omp_get_num_threads();
        const int ntasks = task->getTaskNum();
        const int *nparents = schedule->nparentsBackward;
        const int *threadBoundaries = task->threadBoundaries.data();
        const int *taskBoundaries = task->taskBoundaries.data();
        const int *threadContToOrigPerm = task->threadContToOrigPerm;

        int nPerthread = (ntasks + nthread - 1) / nthread;
        int nBegin = std::min(nPerthread * tid, ntasks);
        int nEnd = std::min(nPerthread * (tid + 1), ntasks);

        volatile int *taskFinished = schedule->taskFinsished;
        int **parents = schedule->parentsBackward;
        memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin) * sizeof(int));
        //      printf("tid: %d\n", tid);
        // fflush(stdout);
#pragma omp barrier
        // Profiler::PAPIOMPCacheProfiler profInstance;
        // profInstance.start();

        for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task)
        {
            SpMP_SCHEDULE_WAIT;
            for (int i = taskBoundaries[task+1] - 1; i >= taskBoundaries[task]; i--)
            {
                // int group_idx = threadContToOrigPerm[i];
                int group_idx = i;
                for (int k = group_ptr[group_idx + 1] - 1; k >= group_ptr[group_idx]; k--)
                {
                    // int node = group_set[k];
                    int node = i;
                    double sum = 0;
                    // the first this diagonal elment
                    for (int j = rowptr[node + 1]- 1; j > rowptr[node]; j--)
                    {
                        sum += x[colidx[j]] * values[j];
                    }
                    x[node] = (b[node] - sum) / values[rowptr[node]];
                } // for each node
                // printf("thread idx: %d, group idx: %d\n", tid, group_idx);
            }
            SPMP_SCHEDULE_NOTIFY
        }

        // profInstance.stop();
    }
}
