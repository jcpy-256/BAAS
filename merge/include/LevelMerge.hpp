#include <iostream>

#include <cstdlib>
#include <cstring>

#include "LevelSet.hpp"
#include "CSR.hpp"

using namespace SPM;

namespace Merge
{
    // get executing threads in a level
    int get_nthreads_per_level(int row_beg, int row_end, const CSR *csrA, const int ncore);

    // get threads in all levels
    void level_nthreads(const LevelSet *level, const CSR *csrA, const int ncore, int *&nthreads_per_level);

    // after getting executing threads in all level, achieve level_merge
    void level_merge(const LevelSet *levelset, const CSR *csrA, int *&nthread_per_level_merge, LevelSet *&levelset_merge, const int ncore);
    inline int get_nthreads_per_level_group_IC0(int node_num, int groups, const CSR *csrA, const int ncore);
    inline int get_nthreads_per_level_group(int node_num, int groups, const CSR *csrA, const int ncore);
    inline int get_nthreads_per_level_group_ILU0(int node_num, int groups, const CSR *csrA, const int ncore);
    inline int get_nthreads_per_level_group_TRSV(int node_num, int groups, const int nnz, const int ncore);

    // void level_merge_group(const LevelSet *levelset, const CSR *csrA, const int *cost, int *nthread_per_level_merge, LevelSet *&levelset_merge, const int ncore);
    void level_nthreads_group(const LevelSet *level, const CSR *csrA, const int *group_ptr, const int *group_set, const int ncore, int *&nthreads_per_level);
    void level_nthreads_group_TRSV(const LevelSet *level, const CSR *csrA, const int *group_ptr, const int *group_set, const int ncore, int *&nthreads_per_level);
    void level_nthreads_group_IC0(const LevelSet *level, const CSR *csrA, const int *group_ptr, const int *group_set, const int ncore, int *&nthreads_per_level);
    void level_nthreads_group_ILU0(const LevelSet *level, const CSR *csrA, const int *group_ptr, const int *group_set, const int ncore, int *&nthreads_per_level);

    void level_nthreads_base(const LevelSet *level, const CSR *csrA, const int ncore, int *&nthreads_per_level);

    void level_merge_group_TRSV(const LevelSet *levelset, const CSR *csrA, const double *cost, int *&nthread_per_level_merge,
                                LevelSet *&levelset_merge, const int ncore, const int *group_ptr, const int *group_set, const int ngroups);

    void level_merge_group(const LevelSet *levelset, const CSR *csrA, const double *cost, int *&nthread_per_level_merge,
                           LevelSet *&levelset_merge, const int ncore, const int *group_ptr, const int *group_set, const int ngroups);
    void constructLevelMerge(const LevelSet *levelset, LevelSet *&levelset_merge);

} // namespace Merge
