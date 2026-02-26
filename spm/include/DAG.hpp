#pragma once

/**
 * @file graph opertation base on these basic fotmats in this folder
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <omp.h>
#include <cassert>
#include <list>

#include "LevelSet.hpp"

using namespace std;

namespace SPM
{

    enum class DAG_MAT
    {
        DAG_CSC,
        DAG_CSR
    };

    class DAG
    {
    private:
        /** check this DAG whether is a null DAG, don't detect the weight vector  */
        bool isNullDAG();
        // private:
        //     /* data */
    public:
        DAG_MAT format = DAG_MAT::DAG_CSC; // csc default
        std::vector<int> DAG_set;          // idx
        std::vector<int> DAG_ptr;          // ptr
        std::vector<double> DAG_vw;        // vertex weight
        int n, edges;
        bool prealloc = false;
        void alloc();
        DAG(int n, int edges, const DAG_MAT mat = DAG_MAT::DAG_CSC); // initialize and prealloc vector
        // n :nodes edges: edge count, set: idx array, ptr: idx offset
        // DAG(int n, int edges, const int *set, const int *ptr); // initialize and copy data
        DAG(int n, int edges, const int *set, const int *ptr, const DAG_MAT mat = DAG_MAT::DAG_CSC); // initialize and copy data
        // DAG(int n, int edges, const int *set, const int *ptr, const double *weight);
        DAG(int n, int edges, const int *set, const int *ptr, const double *weight, const DAG_MAT mat = DAG_MAT::DAG_CSC);
        DAG(DAG &dag); // deep copy
        DAG();
        ~DAG();
        // 自定义函数

        void dealloc();
        void findLevels(LevelSet *&level);
        void findLevelsPostOrder(LevelSet *&level);
        void updateEdges(); // 根据ptr更新edge的数量
        int getEdges() const;     // 获取edges
        int getNodeNum() const;

        void setNodeNum(const int nodeNum);

        // 对DAG CSC格式 进行 two-hop  transitive reduction 并返回一个pruned DAG
        static void partialSparsification_CSC(const int n, const int edges_not_prune, const int *DAG_ptr_not_prune, const int *DAG_set_not_prune,
                                              std::vector<int> &DAG_ptr, std::vector<int> &DAG_set, bool cliqueSimplification = true);
        // inverse the edge direction of a dag
        static DAG *inverseDAG(const DAG &dag, bool isSort = true);
    };

    // DAG::DAG(/* args */)
    // {
    // }

    // 0-based indexing
    void DAG_levelSet_CSC(int n, const int *colptr, const int *rowidx, int &nlevel, int *level_ptr, int *node_to_level, int *node_grouped_by_level);

    // 0-based indexing 
    void DAG_levelSet_CSR(int n, const int *rowptr, const int *colidx, int &nlevel, int *level_ptr, int *node_to_level, int *node_grouped_by_level);

} // namespace SPM