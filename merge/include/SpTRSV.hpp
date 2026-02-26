#pragma once

#include <iostream>
#include <immintrin.h>

#include "omp.h"
#include "SPM.hpp"
#include "LevelTask.hpp"
#include "LevelMerge.hpp"
#include "MarcoUtils.hpp"

// using namespace Co

using namespace SPM;

#define SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT \
    {                                    \
        int n = nparents[task];          \
        int *c = parents[task];          \
        for (int i = 0; i < n; ++i)      \
            while (!taskFinished[c[i]])  \
                _mm_pause();             \
    }

#define SPMP_LEVEL_SCHEDULE_NOTIFY_DEFAULT \
    {                                      \
        taskFinished[task] = 1;            \
    }

#ifndef SpMP_SCHEDULE_WAIT
#define SpMP_SCHEDULE_WAIT SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT
#endif

#ifndef SPMP_SCHEDULE_NOTIFY
#define SPMP_SCHEDULE_NOTIFY SPMP_LEVEL_SCHEDULE_NOTIFY_DEFAULT
#endif

// Sptrsv serial reference
void sptrsv_serial_csr(const CSR &A, double x[], const double y[]);

// Sptrsv level set without matrix permutation
void sptrsv_level_csr_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset);

// Sptrsv level set with matrix permutation
void sptrsv_level_csr(const CSR *csrA, double *&x, const double *b, const LevelSet *levelset);

void sptrsv_level_csr_merge_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level);
// void sptrsv_serial_csr_group_merge_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level);

// SpTRSV: group the vertex and use level merge strategy
void sptrsv_level_csr_group_merge_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level, const int *group_ptr, const int *group_set);

// void sptrsv_serial_csr_group_merge_no_perm(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level, const int* group_ptr, const int *group_set);

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
                               int *groupPtr, int *groupSet);

/**
 * @note: solve the SpTRSV using vertex coarsening method in level set with synchronization barrier, and the task has been organized in level-by-level
 * @param csrA: CSR format for sparse lower triangular matrix
 * @param x: the solution vector x
 * @param b: single right-hand vector
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 */
void sptrsv_level_csr_group_merge_no_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                                const int *group_ptr, const int *group_set, const Merge::Task *task);

/**
 * @note: solve the SpTRSV using vertex coarsening method in level set with synchonizrion point-to-point, and the task has been organized in level-by-level
 * @param csrA: CSR format for sparse lower triangular matrix
 * @param x: the solution vector x
 * @param b: single right-hand vector
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 * @param schdule: the schedule information, such as parents task list, parent number, and finished flag, etc.
 */
void sptrsv_p2p_csr_group_merge_no_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                              const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schdule);

/**
 * @note: solve the SpTRSV using vertex coarsening method in level set with synchonizrion point-to-point, and the task has been organized in level-by-level. Before this process, the matrix A is permuted by thread execution sequence.
 * @param csrA: CSR format for sparse lower triangular matrix
 * @param x: the solution vector x
 * @param b: single right-hand vector
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 * @param schdule: the schedule information, such as parents task list, parent number, and finished flag, etc.
 */
void sptrsv_p2p_csr_group_merge_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                           const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schdule);

// ******************************************** the following serivces for Solver Test *******************************************

/**
 * @note: solve Ay = b, and the matrix A is a lower triangular matrix, which elements in each row is ordered by column idx. Notably, the
 * diagonal element is not separately stored in diag array, but in values array
 * @param A: the lower triangular matrix
 * @param y: the solution vector
 * @param b: the right-hands vector
 */
void lower_csr_trsv_serial(const CSR &A, double *y, const double *b);

/**
 * @note: solve Ay = b, and the matrix A is a upper triangular matrix, which elements in each row is ordered by column idx. Notably, the
 * diagonal element is not separately stored in diag array, but in values array
 * @param A: the upper triangular matrix
 * @param y: the solution vector
 * @param b: the right-hands vector
 */
void upper_csr_trsv_serial(const CSR &A, double *y, const double *b);

/**
 * @note: solve the SpTRSV using vertex coarsening method in level set with synchonizrion point-to-point, and the task has been organized in level-by-level. Before this process, the matrix A is permuted by thread execution sequence.
 * @param csrA: CSR format for sparse upper triangular matrix
 * @param x: the solution vector x
 * @param b: single right-hand vector
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 * @param schdule: the schedule information, such as parents task list, parent number, and finished flag, etc.
 */
void sptrsv_backward_p2p_csr_group_merge_no_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                                       const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule);

void sptrsv_backward_p2p_csr_group_merge_perm_alloc(const CSR *csrA, double *x, const double *b, const LevelSet *levelset, const int *nthread_per_level,
                                                    const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule);
