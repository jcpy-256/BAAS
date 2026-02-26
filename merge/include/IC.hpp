#pragma once

#include "SPM.hpp"
#include "Merge.hpp"
using SPM::CSR;
using SPM::LevelSet;

#ifndef IC0_LEVEL_SCHEDULE_WAIT_DEFAULT
#define IC0_LEVEL_SCHEDULE_WAIT_DEFAULT \
    {                                   \
        int n = nparents[task];         \
        int *c = parents[task];         \
        for (int i = 0; i < n; ++i)     \
            while (!taskFinished[c[i]]) \
                _mm_pause();            \
    }
#endif

#ifndef IC0_LEVEL_SCHEDULE_NOTIFY_DEFAULT
#define IC0_LEVEL_SCHEDULE_NOTIFY_DEFAULT \
    {                                     \
        taskFinished[task] = 1;           \
    }

#endif

#ifndef IC0_SCHEDULE_WAIT
#define IC0_SCHEDULE_WAIT IC0_LEVEL_SCHEDULE_WAIT_DEFAULT
#endif

#ifndef IC0_SCHEDULE_NOTIFY
#define IC0_SCHEDULE_NOTIFY IC0_LEVEL_SCHEDULE_NOTIFY_DEFAULT
#endif

inline double sparse_dot_product(int l1, int u1, int l2, int u2, const int *indices, const double *data);

//=============================== Up looking Looking ==============================
void spic0_csr_uL_serial(const CSR *A, double *lu);

//=============================== Up looking Looking ==============================
void spic0_csr_uL_levelset(const CSR *A, double *lu);

//=============================== Up looking Looking  Kernel ==============================
void spic0_csr_uL_levelset_kernel(const CSR *A, double *lu, const LevelSet *levelset);

void spic0_csr_uL_p2p_group_merge_no_perm(const CSR *A, double *lu, const LevelSet *levelset, const int *nthread_per_level,
                                          const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule);

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
                                       const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule);
