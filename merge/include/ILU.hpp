#pragma once
#include "immintrin.h"
#include "SPM.hpp"
#include "MarcoUtils.hpp"
#include "LevelTask.hpp"
#include "LevelMerge.hpp"

using SPM::CSR;
using SPM::LevelSet;

#ifndef SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT
#define SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT \
    {                                    \
        int n = nparents[task];          \
        int *c = parents[task];          \
        for (int i = 0; i < n; ++i)      \
            while (!taskFinished[c[i]])  \
                _mm_pause();             \
    }
#endif

#ifndef SPMP_LEVEL_SCHEDULE_NOTIFY_DEFAULT
#define SPMP_LEVEL_SCHEDULE_NOTIFY_DEFAULT \
    {                                      \
        taskFinished[task] = 1;            \
    }
#endif

#ifndef ILU0_LEVEL_SCHEDULE_WAIT_DEFAULT
#define ILU0_LEVEL_SCHEDULE_WAIT_DEFAULT \
    {                                    \
        int n = nparents[task];          \
        int *c = parents[task];          \
        for (int i = 0; i < n; ++i)      \
            while (!taskFinished[c[i]])  \
                _mm_pause();             \
    }
#endif

#ifndef ILU0_LEVEL_SCHEDULE_NOTIFY_DEFAULT
#define ILU0_LEVEL_SCHEDULE_NOTIFY_DEFAULT \
    {                                      \
        taskFinished[task] = 1;            \
    }

#endif 


#ifndef ILU0_SCHEDULE_WAIT
#define ILU0_SCHEDULE_WAIT ILU0_LEVEL_SCHEDULE_WAIT_DEFAULT
#endif

#ifndef ILU0_SCHEDULE_NOTIFY
#define ILU0_SCHEDULE_NOTIFY ILU0_LEVEL_SCHEDULE_NOTIFY_DEFAULT
#endif


void ilu0_csr_rightlooking_inplace(CSR *A);


/**
 * @note: run ilu0 in serial with up-looking order. the result L and U is stored in CSR A inplace.
 */
void ilu0_csr_leftlooking_inplace(CSR *A);

/**
 * @note: run ilu0 routine in parallel. the result L and U is stored in CSR A inplace.
 * The CSR A has construct diagptr to ensure that the routine can located diagonal element through it.
 * Also, column indices must be ascending within each row.
 */
void ilu0_csr_leftlooking_inplace_levelset(CSR *A);

/**
 * @note: run ilu0 routine in serial. the result L and U is stored in CSR A inplace.
 * The CSR A has construct diagptr to ensure that the routine can located diagonal element through it.
 * Also, column indices must be ascending within each row. In this function, we don't use colidxMapping to update row vector.
 * Conversely, the two pointer scan for ordered sequence is used.
 */
void ilu0csr_uplooking_ref(CSR *A);


/**
 * @note: run ilu0 routine in serial. the result L and U is stored in lu
 * The CSR A has construct diagptr to ensure that the routine can located diagonal element through it.
 * Also, column indices must be ascending within each row. In this function, we don't use colidxMapping to update row vector.
 * Conversely, the two pointer scan for ordered sequence is used.
 */
void ilu0csr_uplooking_ref(CSR *A, double *lu);

/**
 * @note: run ilu0 routine in parallel with level-set method. the result L and U is stored in CSR A inplace.
 * The CSR A has construct diagptr to ensure that the routine can located diagonal element through it.
 * Also, column indices must be ascending within each row. In this function, we don't use colidxMapping to update row vector.
 * Conversely, the two pointer scan for ordered sequence is used.
 */
void ilu0csr_uplooking_levelset(CSR *A);

/**
 * @note: run ilu0 routine in parallel with level-set method. the result L and U is stored in lu array.
 * The CSR A has construct diagptr to ensure that the routine can located diagonal element through it.
 * Also, column indices must be ascending within each row. In this function, we don't use colidxMapping to update row vector.
 * Conversely, the two pointer scan for ordered sequence is used.
 */
void ilu0csr_uplooking_levelset(const CSR *A, double *lu);


void ilu0csr_uplooking_levelset_kernel(const CSR *A, const LevelSet *levelset, double *lu);


/**
 * @note: perform ILU0 factorization (A \approx LU) using vertex coarsening method in level set with synchronization point-to-point, and the task has been organized in level-by-level
 * @param csrA: CSR format for sparse matrix A, and the column idx of each row is strictly ascending
 * @param lu: the factorization matrix L and U value. 
 * @param levelset: level set information
 * @param nthread_per_level: the executing thread number of each level
 * @param group_ptr: group ptr
 * @param group_set: node idx
 * @param task: the Task class, which stores the task organization in level-by-level
 * @param schdule: the schedule information, such as parents task list, parent number, and finished flag, etc.
 */
void ilu0_p2p_csr_group_merge_no_perm_alloc(const CSR *A, double *lu, const LevelSet *levelset, const int *nthread_per_level, 
                                            const int *group_ptr, const int *group_set, const Merge::Task *task, const Merge::TaskSchedule *schedule);