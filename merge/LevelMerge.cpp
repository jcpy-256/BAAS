#include <iostream>
#include <cmath>

#include "LevelMerge.hpp"
#include "MarcoUtils.hpp"
#include "spmUtils.hpp"

namespace Merge
{
    // get executing threads in a level
    int get_nthreads_per_level(int row_beg, int row_end, const CSR *csrA, const int ncore)
    {
        if (row_end - row_beg < 40)
        {
            return 1;
        }
        else
        {
            int avg_nthreads = std::ceil(1.0 * (row_end - row_beg) / 10);
            return std::min(avg_nthreads, ncore);
        }
    }

    // get threads in all levels
    void level_nthreads(const LevelSet *level, const CSR *csrA, const int ncore, int *&nthreads_per_level)
    {
        int nlevel = level->getLevels();
        int *level_ptr = level->level_ptr;
        nthreads_per_level = MALLOC(int, nlevel);
        if (nlevel == 1) // one level
        {
            int row_beg = level_ptr[0];
            int row_end = level_ptr[1];
            nthreads_per_level[0] = std::min(row_end - row_beg, ncore);
            return;
        }
#pragma omp parallel for
        for (int l = 0; l < nlevel; l++)
        {
            int row_beg = level_ptr[l];
            int row_end = level_ptr[l + 1];
            int n = get_nthreads_per_level(row_beg, row_end, csrA, ncore);
            nthreads_per_level[l] = n;
        }
    }

    // after getting executing threads in all level, achieve level_merge
    void level_merge(const LevelSet *levelset, const CSR *csrA, int *&nthread_per_level_merge, LevelSet *&levelset_merge, const int ncore)
    {
        int nlevels = levelset->getLevels();
        int *level_ptr = levelset->level_ptr;

        // processing non-merge nthreads_per_level, merge the level whose thread is 1.
        int *nthreads_per_level_tmp;
        level_nthreads(levelset, csrA, ncore, nthreads_per_level_tmp);

        // scance the nthreads_per_level_tmp
        int *level_merge_ptr = MALLOC(int, nlevels + 1);
        int *nodeToLevel_merge = MALLOC(int, levelset->getNodeNum());
        // int *nthread_per_level_alias = MALLOC(int , nlevels);
        CHECK_POINTER(level_merge_ptr);
        CHECK_POINTER(nodeToLevel_merge);
        // CHECK_POINTER(nthread_per_level_alias)
        vectorInit(level_merge_ptr, 0, nlevels + 1);

        // // two pointer to delete duplicate elements
        int level_m = 0;
        for (int i = 1; i < nlevels;)
        {
            if (nthreads_per_level_tmp[i] == 1 && nthreads_per_level_tmp[i] == nthreads_per_level_tmp[level_m])
            {
                i++;
                // nthreads_per_level_tmp[level_p] = nthreads_per_level_tmp[i];
            }
            else
            {
                level_m++;
                nthreads_per_level_tmp[level_m] = nthreads_per_level_tmp[i];
                level_merge_ptr[level_m] = level_ptr[i]; // copy the level_ptr( begin )
                i++;
            }
        }
        level_m++; //
        level_merge_ptr[level_m] = level_ptr[nlevels];
        nthread_per_level_merge = MALLOC(int, level_m);
        int *level_merge_ptr_alias = MALLOC(int, level_m + 1);
        memcpy(nthread_per_level_merge, nthreads_per_level_tmp, level_m * sizeof(int));
        memcpy(level_merge_ptr_alias, level_merge_ptr, (level_m + 1) * sizeof(int));
        FREE(level_merge_ptr);
        FREE(nthreads_per_level_tmp);

        levelset_merge->setNodeNum(levelset->getNodeNum());
        levelset_merge->setLevels(level_m);

        levelset_merge->level_ptr = level_merge_ptr_alias;
        // level_merge

        constructLevelMerge(levelset, levelset_merge); // generate nodeTolevel and permutation information
    }


    // int n = get_nthreads_per_level(row_beg, row_end, csrA, ncore);
    // get executing threads in a level
    int get_nthreads_per_level_group(int node_num, int groups, const CSR *csrA, const int ncore)
    {
        // printf("ngroups: %d\n", groups);
        if (node_num < 40 || groups <= 12)
        {
            return 1;
        }
        else
        {
            int avg_nthreads = std::ceil(1.0 * (node_num) / 10);
            return std::min(std::max(avg_nthreads, groups), ncore);
        }
    }

    inline int get_nthreads_per_level_group_TRSV(int node_num, int groups, const int nnz, const int ncore)
    {
        // int nnzPerRow = std::ceil(1.0 * nnz / node_num);
        if (node_num < 16 || groups <= 4 || nnz < 32)
        {
            return 1;
        }
        else
        {
            int avg_nthreads = std::ceil(1.0 * (node_num) / 8);
            int max = std::max(nnz / 32, avg_nthreads);
            return std::min(std::min(max, groups), ncore);
        }
    }

    inline int get_nthreads_per_level_group_IC0(int node_num, int groups, const CSR *csrA, const int ncore)
    {
        // printf("ngroups: %d\n", groups);
        if (node_num < 4 || groups <= 2)
        {
            return 1;
        }
        else
        {
            int avg_nthreads = std::ceil(1.0 * (node_num) / 2);
            return std::min(std::min(avg_nthreads, groups), ncore);
        }
    }



    inline int get_nthreads_per_level_group_ILU0(int node_num, int groups, const CSR *csrA, const int ncore)
    {
        // printf("ngroups: %d\n", groups);
        if (node_num < 4 || groups <= 2)
        {
            return 1;
        }
        else
        {
            int avg_nthreads = std::ceil(1.0 * (node_num) / 2);
            return std::min(std::min(avg_nthreads, groups), ncore);
        }
    }


    void level_nthreads_group_TRSV(const LevelSet *level, const CSR *csrA, const int *group_ptr, const int *group_set, const int ncore, int *&nthreads_per_level)
    {
        int nlevel = level->getLevels();
        int *level_ptr = level->level_ptr;
        int *permToOrig = level->permToOrig;
        int *rowptr = csrA->rowptr;
        nthreads_per_level = MALLOC(int, nlevel);
        if (nlevel == 1) // one level
        {
            int row_beg = level_ptr[0];
            int row_end = level_ptr[1];

            nthreads_per_level[0] = std::min(row_end - row_beg, ncore);
            return;
        }
#pragma omp parallel for
        for (int l = 0; l < nlevel; l++)
        {
            int g_beg = level_ptr[l];
            int g_end = level_ptr[l + 1];
            int node_num = 0;
            int nnz = 0;
            for (; g_beg < g_end; g_beg++)
            {
                int g = permToOrig[g_beg];
                node_num += group_ptr[g + 1] - group_ptr[g];
                for (int node_ptr = group_ptr[g]; node_ptr < group_ptr[g + 1]; node_ptr++)
                {
                    int node = group_set[node_ptr];
                    nnz += rowptr[node + 1] - rowptr[node];
                }
            }
            int n = get_nthreads_per_level_group_TRSV(node_num, level_ptr[l + 1] - level_ptr[l], nnz, ncore);
            // int n  = 1;
            nthreads_per_level[l] = n;
        }
    }

    void level_nthreads_group_ILU0(const LevelSet *level, const CSR *csrA, const int *group_ptr, const int *group_set, const int ncore, int *&nthreads_per_level)
    {
        int nlevel = level->getLevels();
        int *level_ptr = level->level_ptr;
        int *permToOrig = level->permToOrig;
        nthreads_per_level = MALLOC(int, nlevel);
        if (nlevel == 1) // one level
        {
            int row_beg = level_ptr[0];
            int row_end = level_ptr[1];

            nthreads_per_level[0] = std::min(row_end - row_beg, ncore);
            return;
        }
#pragma omp parallel for
        for (int l = 0; l < nlevel; l++)
        {
            int g_beg = level_ptr[l];
            int g_end = level_ptr[l + 1];
            int node_num = 0;
            for (; g_beg < g_end; g_beg++)
            {
                int g = permToOrig[g_beg];
                node_num += group_ptr[g + 1] - group_ptr[g];
            }
            int n = get_nthreads_per_level_group_ILU0(node_num, level_ptr[l + 1] - level_ptr[l], csrA, ncore);
            // int n  = 1;
            nthreads_per_level[l] = n;
        }
    }

    void level_nthreads_group_IC0(const LevelSet *level, const CSR *csrA, const int *group_ptr, const int *group_set, const int ncore, int *&nthreads_per_level)
    {
        int nlevel = level->getLevels();
        int *level_ptr = level->level_ptr;
        int *permToOrig = level->permToOrig;
        nthreads_per_level = MALLOC(int, nlevel);
        if (nlevel == 1) // one level
        {
            int row_beg = level_ptr[0];
            int row_end = level_ptr[1];

            nthreads_per_level[0] = std::min(row_end - row_beg, ncore);
            return;
        }
#pragma omp parallel for
        for (int l = 0; l < nlevel; l++)
        {
            int g_beg = level_ptr[l];
            int g_end = level_ptr[l + 1];
            int node_num = 0;
            for (; g_beg < g_end; g_beg++)
            {
                int g = permToOrig[g_beg];
                node_num += group_ptr[g + 1] - group_ptr[g];
            }
            int n = get_nthreads_per_level_group_IC0(node_num, level_ptr[l + 1] - level_ptr[l], csrA, ncore);
            // int n  = 1;
            nthreads_per_level[l] = n;
        }
    }

    
    void level_nthreads_base(const LevelSet *level, const CSR *csrA, const int ncore, int *&nthreads_per_level)
    {
        int nlevel = level->getLevels();
        int *level_ptr = level->level_ptr;
        int *permToOrig = level->permToOrig;
        nthreads_per_level = MALLOC(int, nlevel);
#pragma omp parallel for
        for (int l = 0; l < nlevel; l++)
        {
            int g_beg = level_ptr[l];
            int g_end = level_ptr[l + 1];
            int node_num = level_ptr[l+1] - level_ptr[l];
            nthreads_per_level[l] = std::min(node_num, ncore);
        }
    }


    void level_nthreads_group(const LevelSet *level, const CSR *csrA, const int *group_ptr, const int *group_set, const int ncore, int *&nthreads_per_level)
    {
        int nlevel = level->getLevels();
        int *level_ptr = level->level_ptr;
        int *permToOrig = level->permToOrig;
        nthreads_per_level = MALLOC(int, nlevel);
        if (nlevel == 1) // one level
        {
            int row_beg = level_ptr[0];
            int row_end = level_ptr[1];

            nthreads_per_level[0] = std::min(row_end - row_beg, ncore);
            return;
        }
#pragma omp parallel for
        for (int l = 0; l < nlevel; l++)
        {
            int g_beg = level_ptr[l];
            int g_end = level_ptr[l + 1];
            int node_num = 0;
            for (; g_beg < g_end; g_beg++)
            {
                int g = permToOrig[g_beg];
                node_num += group_ptr[g + 1] - group_ptr[g];
            }
            int n = get_nthreads_per_level_group(node_num, level_ptr[l + 1] - level_ptr[l], csrA, ncore);
            // int n  = 1;
            nthreads_per_level[l] = n;
        }
    }

    void level_merge_group_TRSV(const LevelSet *levelset, const CSR *csrA, const double *cost, int *&nthread_per_level_merge,
                                LevelSet *&levelset_merge, const int ncore, const int *group_ptr, const int *group_set, const int ngroups)
    {
        int nlevels = levelset->getLevels();
        int *level_ptr = levelset->level_ptr;

        // processing non-merge nthreads_per_level, merge the level whose thread is 1.
        int *nthreads_per_level_tmp;
        // level_nthreads(levelset, csrA, ncore, nthreads_per_level_tmp);
        level_nthreads_group_TRSV(levelset, csrA, group_ptr, group_set, ncore, nthreads_per_level_tmp);

        // scance the nthreads_per_level_tmp
        int *level_merge_ptr = MALLOC(int, nlevels + 1);
        int *nodeToLevel_merge = MALLOC(int, levelset->getNodeNum());
        // int *nthread_per_level_alias = MALLOC(int , nlevels);
        CHECK_POINTER(level_merge_ptr);
        CHECK_POINTER(nodeToLevel_merge);
        // CHECK_POINTER(nthread_per_level_alias)
        vectorInit(level_merge_ptr, 0, nlevels + 1);

        // // two pointer to delete duplicate elements
        int level_m = 0;
        for (int i = 1; i < nlevels;)
        {
            if (nthreads_per_level_tmp[i] == 1 && nthreads_per_level_tmp[i] == nthreads_per_level_tmp[level_m])
            {
                i++;
                // nthreads_per_level_tmp[level_p] = nthreads_per_level_tmp[i];
            }
            else
            {
                level_m++;
                nthreads_per_level_tmp[level_m] = nthreads_per_level_tmp[i];
                level_merge_ptr[level_m] = level_ptr[i]; // copy the level_ptr( begin )
                i++;
            }
        }
        level_m++; //
        level_merge_ptr[level_m] = level_ptr[nlevels];
        nthread_per_level_merge = MALLOC(int, level_m);
        int *level_merge_ptr_alias = MALLOC(int, level_m + 1);
        memcpy(nthread_per_level_merge, nthreads_per_level_tmp, level_m * sizeof(int));
        memcpy(level_merge_ptr_alias, level_merge_ptr, (level_m + 1) * sizeof(int));
        FREE(level_merge_ptr);
        FREE(nthreads_per_level_tmp);

        levelset_merge->setNodeNum(levelset->getNodeNum());
        levelset_merge->setLevels(level_m);

        levelset_merge->level_ptr = level_merge_ptr_alias;

        constructLevelMerge(levelset, levelset_merge); // generate nodeTolevel and permutation information
    }

    void level_merge_group(const LevelSet *levelset, const CSR *csrA, const double *cost, int *&nthread_per_level_merge,
                           LevelSet *&levelset_merge, const int ncore, const int *group_ptr, const int *group_set, const int ngroups)
    {
        int nlevels = levelset->getLevels();
        int *level_ptr = levelset->level_ptr;

        // processing non-merge nthreads_per_level, merge the level whose thread is 1.
        int *nthreads_per_level_tmp;
        // level_nthreads(levelset, csrA, ncore, nthreads_per_level_tmp);
        level_nthreads_group(levelset, csrA, group_ptr, group_set, ncore, nthreads_per_level_tmp);

        // scance the nthreads_per_level_tmp
        int *level_merge_ptr = MALLOC(int, nlevels + 1);
        int *nodeToLevel_merge = MALLOC(int, levelset->getNodeNum());
        // int *nthread_per_level_alias = MALLOC(int , nlevels);
        CHECK_POINTER(level_merge_ptr);
        CHECK_POINTER(nodeToLevel_merge);
        // CHECK_POINTER(nthread_per_level_alias)
        vectorInit(level_merge_ptr, 0, nlevels + 1);

        // // two pointer to delete duplicate elements
        int level_m = 0;
        for (int i = 1; i < nlevels;)
        {
            if (nthreads_per_level_tmp[i] == 1 && nthreads_per_level_tmp[i] == nthreads_per_level_tmp[level_m])
            {
                i++;
                // nthreads_per_level_tmp[level_p] = nthreads_per_level_tmp[i];
            }
            else
            {
                level_m++;
                nthreads_per_level_tmp[level_m] = nthreads_per_level_tmp[i];
                level_merge_ptr[level_m] = level_ptr[i]; // copy the level_ptr( begin )
                i++;
            }
        }
        level_m++; //
        level_merge_ptr[level_m] = level_ptr[nlevels];
        nthread_per_level_merge = MALLOC(int, level_m);
        int *level_merge_ptr_alias = MALLOC(int, level_m + 1);
        memcpy(nthread_per_level_merge, nthreads_per_level_tmp, level_m * sizeof(int));
        memcpy(level_merge_ptr_alias, level_merge_ptr, (level_m + 1) * sizeof(int));
        FREE(level_merge_ptr);
        FREE(nthreads_per_level_tmp);

        levelset_merge->setNodeNum(levelset->getNodeNum());
        levelset_merge->setLevels(level_m);

        levelset_merge->level_ptr = level_merge_ptr_alias;

        constructLevelMerge(levelset, levelset_merge); // generate nodeTolevel and permutation information
    }

    // construct the permutation, nodeToLevel, and permToOrig
    void constructLevelMerge(const LevelSet *levelset, LevelSet *&levelset_merge)
    {
        int n = levelset->getNodeNum();
        int nlevels_merge = levelset_merge->getLevels();
        int *nodeToLevel_merge_tmp = MALLOC(int, n);
        int *permutation_merge_tmp = MALLOC(int, n);
        int *permToOrig_merge_tmp = MALLOC(int, n);

        int *level_merge_ptr = levelset_merge->level_ptr;
        // int *nodeToLevel_merge = MALLOC(int, n);

        // int *nodeToLevel = levelset->nodeToLevel;
        int *permToOrig = levelset->permToOrig;   // node grouped by level (level set reverse perm)
        int *permutation = levelset->permutation; // perm

// perm and reverse perm is not influenced, and only nodeToLevel is need to rectify
// find new nodeToLevel from permToOrig
#pragma omp parallel for
        for (int i = 0; i < nlevels_merge; i++)
        {
            for (int j = level_merge_ptr[i]; j < level_merge_ptr[i + 1]; j++)
            {
                nodeToLevel_merge_tmp[permToOrig[j]] = i;
            }
        }
        copyVector(permToOrig_merge_tmp, permToOrig, n);
        copyVector(permutation_merge_tmp, permutation, n);

        levelset_merge->nodeToLevel = nodeToLevel_merge_tmp;
        levelset_merge->permToOrig = permToOrig_merge_tmp;
        levelset_merge->permutation = permutation_merge_tmp;
        // isPerm
    }
} // namespace Merge
