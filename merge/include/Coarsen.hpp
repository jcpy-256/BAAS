
#pragma once
/**
 * created in 2025-06-19
 * @file: this file aims to implement vertex coarsening and level coarsening in SpTRSV DAG
 */

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <list>

#include <vector>

#include "DAG.hpp"

using namespace SPM;

namespace Coarsen
{
    enum Kernel
    {
        SpTRSV_LL,
        SpTRSV_LU,
        ILU0,
        IC0
    };

    bool isSingleForwardTree(int node, int &nchild, std::vector<int> &child_node, std::vector<bool> &visited, const std::vector<int> &DAG_ptr, const std::vector<int> &DAG_set, const std::vector<int> &DAG_inv_ptr, const std::vector<int> &DAG_inv_set);

    bool isSingleReverseTree(int node, int &nparent, std::vector<int> &parent_node, std::vector<bool> &visited, const std::vector<int> &DAG_ptr, const std::vector<int> &DAG_set, const std::vector<int> &DAG_inv_ptr, const std::vector<int> &DAG_inv_set);

    /**
     * @note: this function implements the forward tree coarsening (chain + divergent pattern), first version without any restriction
     *
     * @param n: the vertex number of DAG
     * @param DAG_ptr: the colptr of DAG in CSC format
     * @param DAG_set: the rowidx of DAG in CSC format
     * @param ngroup: the number of vertex after coarsening
     * @param group_ptr: the range of original vertex index included in a coarsened vertex within group_set
     * @param group_set: original vertex indices arranged by coarse nodes
     */
    bool forwardTreeCoarseningBFS_all(const int n, const std::vector<int> &DAG_ptr, const std::vector<int> &DAG_set,
                                      int &ngroups, std::vector<int> &group_ptr, std::vector<int> &group_set, bool restriction = false);

    /**
     * @note: this function implements the reverse tree coarsening (chain + convergent pattern), first version without any restriction
     *
     * @param n: the vertex number of DAG
     * @param DAG_ptr: the colptr of DAG in CSC format
     * @param DAG_set: the rowidx of DAG in CSC format
     * @param ngroup: the number of vertex after coarsening
     * @param group_ptr: the range of original vertex index included in a coarsened vertex within group_set
     * @param group_set: original vertex indices arranged by coarse nodes
     */
    bool reverseTreeCoarseningBFS_all(const int n, const std::vector<int> &DAG_ptr, const std::vector<int> &DAG_set, int &ngroups, std::vector<int> &group_ptr, std::vector<int> &group_set, bool restriction = false);

    /**
     * @note: this function implements the forward tree coarsening (chain + divergent pattern), first version without any restriction
     *
     * @param n: the vertex number of DAG
     * @param DAG_ptr: the colptr of DAG in CSC format
     * @param DAG_set: the rowidx of DAG in CSC format
     * @param ngroup: the number of vertex after coarsening
     * @param group_ptr: the range of original vertex index included in a coarsened vertex within group_set
     * @param group_set: original vertex indices arranged by coarse nodes
     */
    bool forwardTreeCoarseningBFS(int n, std::vector<int> &DAG_ptr, std::vector<int> &DAG_set,
                                  int &ngroups, std::vector<int> &group_ptr, std::vector<int> &group_set, bool restriction = false);
    /**
     * @note: this function implements the reverse tree coarsening (chain + convergent pattern), first version without any restriction
     *
     * @param n: the vertex number of DAG
     * @param DAG_ptr: the colptr of DAG in CSC format
     * @param DAG_set: the rowidx of DAG in CSC format
     * @param ngroup: the number of vertex after coarsening
     * @param group_ptr: the range of original vertex index included in a coarsened vertex within group_set
     * @param group_set: original vertex indices arranged by coarse nodes
     */
    bool reverseTreeCoarseningBFS(int n, std::vector<int> &DAG_ptr, std::vector<int> &DAG_set, int &ngroups, std::vector<int> &group_ptr, std::vector<int> &group_set, bool restriction = false);
    
    
    void buildGroupDAGParallel(const int &n, const int &ngroups, const int *group_ptr, const int *group_set,
                               const int *DAG_ptr, const int *DAG_set, std::vector<int> &group_DAG_ptr, std::vector<int> &group_DAG_set);
    void buildGroupDAG(const int &n, const int &ngroups, const int *group_ptr, const int *group_set,
                       const int *DAG_ptr, const int *DAG_set, std::vector<int> &group_DAG_ptr, std::vector<int> &group_DAG_set);

    /**
     * @note: this function implement the caclucation of cost for grouped or non-grouped DAG in different kernel context.
     *
     * @param nodes: the node number of DAG
     * @param colptr: the column offset pointer array in CSC format
     * @param rowidx: the row index in CSC format
     * @param rowptr: the row offset pointer array in CSR format
     * @param colidx: the column idx in CSR format
     * @param kernel: computation kernel，such as SpTRSV, ILU, IC
     * @param group_ptr: coarsened DAG group offser pointer array
     * @param group_set: coarsened DAG group node array
     * @param grouped: whether the DAG is coarsened
     * @param cost: the cost vector (return)
     */
    void costComputation(int nodes, const int *colptr, int *rowidx, const int *rowptr, const int *colidx,
                         Kernel kernel, const int *group_ptr, const int *group_set, bool grouped, std::vector<double> &cost);
    void groupRemapping(std::vector<int> &group_ptr, std::vector<int> &group_set, const std::vector<int> group_ptr_f, const std::vector<int> group_set_f, const int ngroups_f);

} // namespace Coarsen
