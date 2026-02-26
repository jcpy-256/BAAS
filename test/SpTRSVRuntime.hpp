#pragma once
#include <iostream>
#include "mkl.h"
#include "SPM.hpp"

#include "Merge.hpp"
// #include "SpTRSV.hpp"
#include "TimeMeasure.hpp"
#include "FusionDemo.hpp"
#include "MarcoUtils.hpp"
#include "MathUtils.hpp"

using namespace Merge;
// using

namespace SpTRSVRT
{

    int build_levelSet_CSC(size_t n, const int *Lp, const int *Li,
                           int *levelPtr, int *levelSet)
    {
        int begin = 0, end = n - 1;
        int cur_level = 0, cur_levelCol = 0;
        int *inDegree = new int[n]();
        bool *visited = new bool[n]();
        for (int i = 0; i < Lp[n]; ++i)
        { // O(nnz) -> but not catch efficient. This code should work well
            // on millions of none zeros to enjoy a gain in the parformance. Maybe we can find another way. Behrooz
            inDegree[Li[i]]++; // Isn't it the nnz in each row? or the rowptr[x + 1] - rowptr[x] in CSR?
        }
        // print_vec("dd\n",0,n,inDegree);
        while (begin <= end)
        {
            for (int i = begin; i <= end; ++i)
            { // For level cur_level
                if (inDegree[i] == 1 && !visited[i])
                { // if no incoming edge
                    visited[i] = true;
                    levelSet[cur_levelCol] = i; // add it to current level
                    cur_levelCol++;             // Adding to level-set - This is a cnt for the current level. Behrooz
                }
            }
            cur_level++; // all nodes_ with zero indegree are processed.
            // assert(cur_level < n);
            if (cur_level >= n)
                return -1;                      // The input graph has a cycle
            levelPtr[cur_level] = cur_levelCol; // The levelPtr starts from level 1. Behrooz
            while (inDegree[begin] == 1)        // Why? Behrooz
            {
                begin++;
                if (begin >= n)
                    break;
            }
            while (inDegree[end] == 1 && begin <= end) // The same why as above. Behrooz
                end--;
            // Updating degrees after removing the nodes_
            for (int l = levelPtr[cur_level - 1]; l < levelPtr[cur_level]; ++l) // I don't get this part. Behrooz
            {
                int cc = levelSet[l];
                for (int j = Lp[cc]; j < Lp[cc + 1]; ++j)
                {
                    if (Li[j] != cc)       // skip diagonals
                        inDegree[Li[j]]--; // removing corresponding edges
                }
            }
            // print_vec("dd\n",0,n,inDegree);
        }
        delete[] inDegree;
        delete[] visited;
        return cur_level; // return number of levels
    }

    class SpRTSV_Serial : public FusionDemo
    {
    protected:
        TimeMeasure fused_code() override
        {
            // this->analysis_time = 0;
            TimeMeasure t1;
            const CSR csrc = *csrA;
            t1.start_timer();
            sptrsv_serial_csr(csrc, x, y);
            t1.measure_elasped_time();
            return t1;
        }

    public:
        SpRTSV_Serial(CSR *csrA, double *correct_x, int n, int nnz, double alpha, std::string algName)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_x = correct_x;
            this->alpha = alpha;
        }
        ~SpRTSV_Serial() = default;
    };

    class SpTRSV_Wavefront : public FusionDemo
    {
    protected:
        // CSC *cscA;
        LevelSet *levelset;
        TimeMeasure fused_code() override
        {
            // double *tmp = MALLOC(double, n);
            // vectorReorder(y, tmp, levelset->permutation, n);
            TimeMeasure t1;
            // const CSR csrc = *csrA;
            t1.start_timer();
            sptrsv_level_csr_no_perm(csrA, x, y, levelset);
            t1.measure_elasped_time();
            return t1;
        }

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            DAG *dag = new DAG(n, nnz, csrA->colidx, csrA->rowptr, DAG_MAT::DAG_CSR);
            // levelset = new LevelSet();
            dag->findLevels(levelset);
            delete dag;
            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
            // omp_set_num_threads(this->num_thread);
        }

    public:
        SpTRSV_Wavefront(CSR *csrA, double *correct_x, int n, int nnz, double alpha, std::string algName, int num_thread)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_x = correct_x;
            this->alpha = alpha;
            this->num_thread = num_thread;
            levelset = new LevelSet();
            // this->cscA = cscA;
        }
        ~SpTRSV_Wavefront()
        {
            delete levelset;
        }

        int getWavefront() const { return levelset->getLevels(); }
    };

    class SpTRSV_WF_Merge : public FusionDemo
    {
    protected:
        LevelSet *levelset_merge;
        int *nthread_per_level;

        TimeMeasure fused_code() override
        {
            TimeMeasure t1;
            t1.start_timer();
            sptrsv_level_csr_merge_no_perm(csrA, x, y, levelset_merge, nthread_per_level);
            t1.measure_elasped_time();
            return t1;
        }

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            DAG *dag = new DAG(n, nnz, csrA->colidx, csrA->rowptr, DAG_MAT::DAG_CSR);
            LevelSet *levelset = new LevelSet();
            dag->findLevels(levelset);
            delete dag;

            level_merge(levelset, csrA, nthread_per_level, levelset_merge, this->num_thread);
            delete levelset;
            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
        }

    public:
        SpTRSV_WF_Merge(CSR *csrA, double *correct_x, int n, int nnz, double alpha, std::string algName, int num_thread)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_x = correct_x;
            // this->n = n;
            // this->nnz = nnz;
            this->alpha = alpha;
            this->num_thread = num_thread;
            levelset_merge = new LevelSet();
        }
        ~SpTRSV_WF_Merge()
        {
            // FREE(nthread_per_level);
            FREE(nthread_per_level)

            delete levelset_merge;
        }

        int getWavefront() const { return levelset_merge->getLevels(); }
    };

    class SpTRSV_MKL : public FusionDemo
    {
    protected:
        bool opt = false;
        sparse_matrix_t matrixA;
        struct matrix_descr descA;

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            MKL_Set_Num_Threads(num_thread);
            sparse_status_t ret = mkl_sparse_d_create_csr(&matrixA, SPARSE_INDEX_BASE_ZERO, csrA->m, csrA->n,
                                                          csrA->rowptr, csrA->rowptr + 1, csrA->colidx, csrA->values);
            if (ret != SPARSE_STATUS_SUCCESS)
            {
                printf("MKL Error in %s, %n", __FILE__, __LINE__);
                return;
            }

            descA.diag = SPARSE_DIAG_NON_UNIT;
            descA.mode = SPARSE_FILL_MODE_LOWER;
            descA.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
            if (this->opt)
            {
                ret = mkl_sparse_set_sv_hint(matrixA, SPARSE_OPERATION_NON_TRANSPOSE, descA, num_test);
                if (ret != SPARSE_STATUS_SUCCESS)
                {
                    printf("MKL Error in %s, %s", __FILE__, __LINE__);
                    return;
                }
                ret = mkl_sparse_set_memory_hint(matrixA, SPARSE_MEMORY_AGGRESSIVE);
                if (ret != SPARSE_STATUS_SUCCESS)
                {
                    printf("MKL Error in %s, %s", __FILE__, __LINE__);
                    return;
                }
                ret = mkl_sparse_optimize(matrixA);
                if (ret != SPARSE_STATUS_SUCCESS)
                {
                    printf("MKL Error in %s, %s", __FILE__, __LINE__);
                    return;
                }
            }
            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
        }

        TimeMeasure fused_code() override
        {
            // Ay = x
            TimeMeasure t1;
            t1.start_timer();
            sparse_status_t ret = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, matrixA, descA, y, x);
            t1.measure_elasped_time();
            if (ret != SPARSE_STATUS_SUCCESS)
            {
                printf("MKL Error in %s, %d", __FILE__, __LINE__);
                return t1;
            }
            return t1;
        }

    public:
        SpTRSV_MKL(CSR *csrA, double *correct_x, int n, int nnz, double alpha, std::string algName, int num_thread, bool opt = false)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_x = correct_x;
            this->alpha = alpha;
            this->num_thread = num_thread;
            this->opt = opt;
        }
        ~SpTRSV_MKL()
        {
            mkl_sparse_destroy(matrixA);
        }
    };

    /********************************* Reverse and Forward Tree with Barrier ****************************** */
    class SpTRSV_DUAL_TREE_BFS_WF_Merge : public FusionDemo
    {
    protected:
        LevelSet *levelset_merge;
        int *nthread_per_level;
        int ngroups;
        CSC *cscA;

        DAG *dag_group;
        std::vector<int> group_set, group_ptr;
        std::vector<double> cost;

        void orderingGroupSet(const int ngroups, const vector<int> &group_ptr, vector<int> &group_set)
        {
#pragma omp parallel for schedule(static)
            for (int g = 0; g < ngroups; g++)
            {
                std::sort(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1]);
            }
        }

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            // printf("dag node number: %d\n", n);
            DAG *dag = new DAG(n, nnz);
            dag->DAG_ptr.clear();
            dag->DAG_set.clear();
            DAG::partialSparsification_CSC(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);
            // HDAGG::partialSparsification(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);

            /************************** reverse coarsen ************************** */
            Coarsen::reverseTreeCoarseningBFS_all(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);
            // HDAGG::treeBasedGroupingBFS(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);

            // printf("reverse ngroups: %d\n", ngroups);
            Coarsen::buildGroupDAG(n, ngroups, group_ptr.data(), group_set.data(),
                                   dag->DAG_ptr.data(), dag->DAG_set.data(), dag_group->DAG_ptr, dag_group->DAG_set);
            dag_group->setNodeNum(ngroups);
            delete dag;

            /************************** forward coarsen ************************** */
            std::vector<int> group_ptr_f, group_set_f;
            DAG *dag_group_f = new DAG();
            int ngroups_f = 0;
            Coarsen::forwardTreeCoarseningBFS_all(ngroups, dag_group->DAG_ptr, dag_group->DAG_set, ngroups_f, group_ptr_f, group_set_f, false);

            Coarsen::groupRemapping(group_ptr, group_set, group_ptr_f, group_set_f, ngroups_f);

            Coarsen::buildGroupDAG(ngroups, ngroups_f, group_ptr_f.data(), group_set_f.data(),
                                   dag_group->DAG_ptr.data(), dag_group->DAG_set.data(), dag_group_f->DAG_ptr, dag_group_f->DAG_set);

            ngroups = ngroups_f;
            // printf("after two coarsening ngroups: %d\n", ngroups);
            dag_group_f->setNodeNum(ngroups);
            dag_group_f->updateEdges();
            LevelSet *levelset = new LevelSet();
            dag_group_f->findLevelsPostOrder(levelset);
            // dag_group->findLevelsPostOrder(levelset);

            delete dag_group;
            dag_group = dag_group_f;

            // level_merge_group(levelset, csrA, cost.data(), nthread_per_level, levelset_merge, this->num_thread, group_ptr.data(), group_set.data(), ngroups);
            level_merge_group(levelset, csrA, cost.data(), nthread_per_level, levelset_merge, this->num_thread, group_ptr.data(), group_set.data(), ngroups);
            // level_merge(levelset, csrA, nthread_per_level, levelset_merge, this->num_thread);
            delete levelset;

            orderingGroupSet(ngroups, group_ptr, group_set);

            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
        }

        TimeMeasure fused_code() override
        {
            TimeMeasure t1;
            t1.start_timer();
            sptrsv_level_csr_group_merge_no_perm(csrA, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data());
            t1.measure_elasped_time();
            return t1;
        }

    public:
        SpTRSV_DUAL_TREE_BFS_WF_Merge(CSR *csrA, CSC *cscA, double *correct_x, int n, int nnz, double alpha, std::string algName, int num_thread)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->cscA = cscA;
            this->correct_x = correct_x;
            this->alpha = alpha;
            this->num_thread = num_thread;

            levelset_merge = new LevelSet();
            this->dag_group = new DAG();
        }

        ~SpTRSV_DUAL_TREE_BFS_WF_Merge()
        {
            delete levelset_merge;
            FREE(nthread_per_level);
            delete dag_group;
        }

        int getWavefront() const { return levelset_merge->getLevels(); }
    };

    /********************************* Reverse and Forward Tree with P2P (BAASWOS) *******************************/
    class SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P : public FusionDemo
    {
    protected:
        LevelSet *levelset_merge;
        int *nthread_per_level;
        int ngroups;
        CSC *cscA;

        DAG *dag_group;
        DAG *taskDAG;
        Merge::Task *task;
        Merge::TaskSchedule *schedule = nullptr;
        std::vector<int> group_set, group_ptr;
        std::vector<double> cost; // cost for grouped node

        void orderingGroupSet(const int ngroups, const vector<int> &group_ptr, vector<int> &group_set)
        {
#pragma omp parallel for schedule(static)
            for (int g = 0; g < ngroups; g++)
            {
                std::sort(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1]);
            }
        }

        void groupPerm(const int ngroups, const int *origToThreadContPerm, const vector<int> &group_set, const vector<int> &group_ptr, vector<int> &group_set_p, vector<int> &group_ptr_p)
        {
            group_set_p.reserve(group_set.size());
            group_set_p.resize(group_set.size(), 0);
            group_ptr_p.reserve(group_ptr.size());
            group_ptr_p.resize(group_ptr.size(), 0);

#pragma omp parallel
            {
#pragma omp for
                for (int g = 0; g < ngroups; g++)
                {
                    // for(int )
                    int perm_group = origToThreadContPerm[g];
                    group_ptr_p[perm_group + 1] = group_ptr[g + 1] - group_ptr[g];
                }
                // #pragma omp barrier

#pragma omp single
                {
                    prefixSumSingle(group_ptr_p.data(), ngroups + 1);
                    assert(group_ptr_p.back() == group_ptr.back());
                }

#pragma omp for
                for (int g = 0; g < ngroups; g++)
                {
                    int perm_group = origToThreadContPerm[g];
                    std::copy(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1], group_set_p.begin() + group_ptr_p[perm_group]);
                }
            }
        }

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            // printf("dag node number: %d\n", n);
            DAG *dag = new DAG(n, nnz);
            dag->DAG_ptr.clear();
            dag->DAG_set.clear();
            DAG::partialSparsification_CSC(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);
            // HDAGG::partialSparsification(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);

            /************************** reverse coarsen ************************** */
            Coarsen::reverseTreeCoarseningBFS_all(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);
            // HDAGG::treeBasedGroupingBFS(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);

            // printf("reverse ngroups: %d\n", ngroups);
            Coarsen::buildGroupDAG(n, ngroups, group_ptr.data(), group_set.data(),
                                   dag->DAG_ptr.data(), dag->DAG_set.data(), dag_group->DAG_ptr, dag_group->DAG_set);
            dag_group->setNodeNum(ngroups);
            delete dag;

            /************************** forward coarsen ************************** */
            std::vector<int> group_ptr_f, group_set_f;
            DAG *dag_group_f = new DAG();
            int ngroups_f = 0;
            Coarsen::forwardTreeCoarseningBFS_all(ngroups, dag_group->DAG_ptr, dag_group->DAG_set, ngroups_f, group_ptr_f, group_set_f, false);

            // mapping group_f into group_ptr,
            Coarsen::groupRemapping(group_ptr, group_set, group_ptr_f, group_set_f, ngroups_f);

            // build new DAG based on group_DAG, with group_f
            Coarsen::buildGroupDAG(ngroups, ngroups_f, group_ptr_f.data(), group_set_f.data(),
                                   dag_group->DAG_ptr.data(), dag_group->DAG_set.data(), dag_group_f->DAG_ptr, dag_group_f->DAG_set);

            ngroups = ngroups_f;
            // printf("after two coarsening ngroups: %d\n", ngroups);
            dag_group_f->setNodeNum(ngroups);
            dag_group_f->updateEdges();
            LevelSet *levelset = new LevelSet();
            dag_group_f->findLevelsPostOrder(levelset);
            // dag_group->findLevelsPostOrder(levelset);
            delete dag_group;
            dag_group = dag_group_f;
            Coarsen::costComputation(ngroups, nullptr, nullptr, csrA->rowptr, csrA->colidx, Coarsen::Kernel::SpTRSV_LL, group_ptr.data(), group_set.data(), true, cost);

            // level_merge_group(levelset, csrA, cost.data(), nthread_per_level, levelset_merge, this->num_thread, group_ptr.data(), group_set.data(), ngroups);
            // level_merge_group(levelset, csrA, cost.data(), nthread_per_level, levelset_merge, this->num_thread, group_ptr.data(), group_set.data(), ngroups);
            level_merge_group_TRSV(levelset, csrA, cost.data(), nthread_per_level, levelset_merge, this->num_thread, group_ptr.data(), group_set.data(), ngroups);
            // level_merge(levelset, csrA, nthread_per_level, levelset_merge, this->num_thread);
            delete levelset;

            orderingGroupSet(ngroups, group_ptr, group_set);

            // 转p2p
            task = new Merge::Task(ngroups, false);
            // task alloc
            task->constructTask(levelset_merge, cost.data(), this->num_thread, nthread_per_level);
            // constrcut task DAG
            DAG *dag_group_inv = DAG::inverseDAG(*dag_group);
            task->constructTaskDAGParallel(dag_group, dag_group_inv, taskDAG, this->num_thread, false);
            delete dag_group_inv;

            // printf("taskNUm: %d\n", taskDAG->getNodeNum());
            // printf("cparents: %d\n", taskDAG->DAG_ptr.back());
            // fflush(stdout);

            // construct task schedule
            schedule = new TaskSchedule(taskDAG->getNodeNum(), taskDAG->DAG_ptr.back());
            schedule->constructTaskSchedule(taskDAG);

            if (this->hasPerm)
            {
                // 对稀疏矩阵进行重排
                std::vector<int> group_set_p, group_ptr_p; // 记录经过perm后的ptr和set集合，随后将类中的group_ptr和set进行std::move
                groupPerm(ngroups, task->origToThreadContPerm, group_set, group_ptr, group_set_p, group_ptr_p);
                std::vector<int> origToPerm(group_set_p.size(), 0);
                getInversePerm(origToPerm.data(), group_set_p.data(), group_set_p.size());

                assert(isPerm(group_set.data(), group_set.size()));
                assert(isPerm(group_set_p.data(), group_set_p.size()));
                assert(isPerm(origToPerm.data(), origToPerm.size()));
                // 对CSR进行深度拷贝
                // this->csrA_perm = new CSR(*csrA);
                // csrA_perm.
                csrA_perm = csrA->permute(origToPerm.data(), group_set_p.data()); // not sort to keep the diag element
                this->permToOrig = group_set_p;
                this->origToPerm = std::move(origToPerm);

                group_ptr = std::move(group_ptr_p);
                group_set = std::move(group_set_p); // inverse permuation : perm -> orig
            }

            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();

            // for (int tid = 0; tid < this->num_thread; tid++)
            // {
            //     for (int t = task->threadBoundaries[tid]; t < task->threadBoundaries[tid + 1]; t++)
            //     {
            //         for (int i = task->taskBoundaries[t]; i < task->taskBoundaries[t + 1]; i++)
            //         {
            //             int group_idx = task->threadContToOrigPerm[i];
            //             printf("tid: %d, group_idx: %d\n", tid, group_idx);
            //         }
            //     }
            // }
            // for (int i = 0; i < taskDAG->getNodeNum(); i++)
            // {
            //     printf("node %d, rowbegin: %d, rowend: %d\n", i, taskDAG->DAG_ptr[i], taskDAG->DAG_ptr[i + 1]);
            //     for (int j = taskDAG->DAG_ptr[i]; j < taskDAG->DAG_ptr[i + 1]; j++)
            //     {
            //         printf("dependency: %d\n", taskDAG->DAG_set[j]);
            //     }
            // }
        }

        TimeMeasure fused_code() override
        {
            TimeMeasure t1;
            if (this->hasPerm)
            {
                t1.start_timer();
                sptrsv_p2p_csr_group_merge_perm_alloc(csrA_perm, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                t1.measure_elasped_time();
            }
            else
            {
                t1.start_timer();
                sptrsv_p2p_csr_group_merge_no_perm_alloc(csrA, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                t1.measure_elasped_time();
            }
            return t1;
        }

    public:
        SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P(CSR *csrA, CSC *cscA, double *correct_x, int n, int nnz, double alpha, std::string algName, int num_thread, bool hasPerm)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->cscA = cscA;
            this->correct_x = correct_x;
            this->alpha = alpha;
            this->num_thread = num_thread;
            this->hasPerm = hasPerm;

            levelset_merge = new LevelSet();
            this->dag_group = new DAG();
            this->taskDAG = new DAG();
        }

        ~SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P()
        {
            delete levelset_merge;
            FREE(nthread_per_level);
            delete dag_group;
            delete task;
            delete taskDAG;
            if (schedule)
            {
                delete schedule;
            }
            if (csrA_perm)
            {
                delete csrA_perm;
            }
        }

        int getWavefront() const { return schedule->getP2PNum(); }
    };

    /********************************* Reverse and Forward Tree with P2P and node Maping in Four rule (BAAS) ****************************** */
    class SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule : public FusionDemo
    {
    protected:
        LevelSet *levelset_merge;
        int *nthread_per_level;
        int ngroups;
        CSC *cscA;
        // CSR *csrA_perm;

        DAG *dag_group;
        DAG *taskDAG;
        Merge::Task *task;
        Merge::TaskSchedule *schedule = nullptr;
        std::vector<int> group_set, group_ptr;
        std::vector<double> cost; // cost for grouped node

        void orderingGroupSet(const int ngroups, const vector<int> &group_ptr, vector<int> &group_set)
        {
#pragma omp parallel for schedule(static)
            for (int g = 0; g < ngroups; g++)
            {
                std::sort(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1]);
            }
        }

        void groupPerm(const int ngroups, const int *origToThreadContPerm, const vector<int> &group_set, const vector<int> &group_ptr, vector<int> &group_set_p, vector<int> &group_ptr_p)
        {
            group_set_p.reserve(group_set.size());
            group_set_p.resize(group_set.size(), 0);
            group_ptr_p.reserve(group_ptr.size());
            group_ptr_p.resize(group_ptr.size(), 0);

#pragma omp parallel
            {
#pragma omp for
                for (int g = 0; g < ngroups; g++)
                {
                    // for(int )
                    int perm_group = origToThreadContPerm[g];
                    group_ptr_p[perm_group + 1] = group_ptr[g + 1] - group_ptr[g];
                }
                // #pragma omp barrier

#pragma omp single
                {
                    prefixSumSingle(group_ptr_p.data(), ngroups + 1);
                    assert(group_ptr_p.back() == group_ptr.back());
                }

#pragma omp for
                for (int g = 0; g < ngroups; g++)
                {
                    int perm_group = origToThreadContPerm[g];
                    std::copy(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1], group_set_p.begin() + group_ptr_p[perm_group]);
                }
            }
        }

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            // printf("dag node number: %d\n", n);
            DAG *dag = new DAG(n, nnz);
            dag->DAG_ptr.clear();
            dag->DAG_set.clear();
            DAG::partialSparsification_CSC(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);
            // HDAGG::partialSparsification(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);

            /************************** reverse coarsen ************************** */
            Coarsen::reverseTreeCoarseningBFS_all(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);
            // HDAGG::treeBasedGroupingBFS(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);

            // printf("reverse ngroups: %d\n", ngroups);
            Coarsen::buildGroupDAGParallel(n, ngroups, group_ptr.data(), group_set.data(),
                                           dag->DAG_ptr.data(), dag->DAG_set.data(), dag_group->DAG_ptr, dag_group->DAG_set);
            dag_group->setNodeNum(ngroups);
            delete dag;

            /************************** forward coarsen ************************** */
            std::vector<int> group_ptr_f, group_set_f;
            DAG *dag_group_f = new DAG();
            int ngroups_f = 0;
            Coarsen::forwardTreeCoarseningBFS_all(ngroups, dag_group->DAG_ptr, dag_group->DAG_set, ngroups_f, group_ptr_f, group_set_f, false);
            // printf("forward ngroups: %d\n", ngroups_f);
            // mapping group_f into group_ptr,
            Coarsen::groupRemapping(group_ptr, group_set, group_ptr_f, group_set_f, ngroups_f);

            // build new DAG based on group_DAG, with group_f
            Coarsen::buildGroupDAGParallel(ngroups, ngroups_f, group_ptr_f.data(), group_set_f.data(),
                                           dag_group->DAG_ptr.data(), dag_group->DAG_set.data(), dag_group_f->DAG_ptr, dag_group_f->DAG_set);

            ngroups = ngroups_f;
            // printf("after two coarsening ngroups: %d\n", ngroups);
            dag_group_f->setNodeNum(ngroups);
            dag_group_f->updateEdges();
            LevelSet *levelset = new LevelSet();
            dag_group_f->findLevelsPostOrder(levelset);

            levelset_merge = levelset;
            // dag_group->findLevelsPostOrder(levelset);
            delete dag_group;
            dag_group = dag_group_f;
            Coarsen::costComputation(ngroups, nullptr, nullptr, csrA->rowptr, csrA->colidx, Coarsen::Kernel::SpTRSV_LL, group_ptr.data(), group_set.data(), true, cost);

            // 在p2p模式下这里不应该进行直接的merge 而是根据线程数量进行分配 intra-transitive会自动消除
            level_nthreads_group_TRSV(levelset_merge, csrA, group_ptr.data(), group_set.data(), this->num_thread, this->nthread_per_level);

            // 确保小的节点编号先执行
            orderingGroupSet(ngroups, group_ptr, group_set);

            // 转p2p
            task = new Merge::Task(ngroups, false);

            DAG *dag_group_inv = DAG::inverseDAG(*dag_group);
            // task alloc
            // task->constructTask(levelset_merge, cost.data(), this->num_thread, nthread_per_level);
            // task->constructMappingSerial(dag_group, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            // task->constructMappingByFinishTime(dag_group, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            // task->constructMappingByFourRule(dag_group,dag_group_inv, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            task->constructMappingByFourRuleParallel(dag_group, dag_group_inv, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());

            // constrcut task DAG
            task->constructTaskDAG(dag_group, dag_group_inv, taskDAG, this->num_thread, false);
            delete dag_group_inv;

            // // printf("taskNUm: %d\n", taskDAG->getNodeNum());
            // // printf("cparents: %d\n", taskDAG->DAG_ptr.back());
            // // fflush(stdout);

            // construct task schedule
            schedule = new TaskSchedule(taskDAG->getNodeNum(), taskDAG->DAG_ptr.back());
            schedule->constructTaskSchedule(taskDAG);

            if (this->hasPerm)
            {
                // 对稀疏矩阵进行重排
                std::vector<int> group_set_p, group_ptr_p; // 记录经过perm后的ptr和set集合，随后将类中的group_ptr和set进行std::move
                groupPerm(ngroups, task->origToThreadContPerm, group_set, group_ptr, group_set_p, group_ptr_p);
                std::vector<int> origToPerm(group_set_p.size(), 0);
                getInversePerm(origToPerm.data(), group_set_p.data(), group_set_p.size());

                assert(isPerm(group_set.data(), group_set.size()));
                assert(isPerm(group_set_p.data(), group_set_p.size()));
                assert(isPerm(origToPerm.data(), origToPerm.size()));
                // 对CSR进行深度拷贝
                // this->csrA_perm = new CSR(*csrA);
                // csrA_perm.
                csrA_perm = csrA->permute(origToPerm.data(), group_set_p.data()); // not sort to keep the diag element
                this->permToOrig = group_set_p;
                this->origToPerm = std::move(origToPerm);

                group_ptr = std::move(group_ptr_p);
                group_set = std::move(group_set_p); // inverse permuation : perm -> orig
            }
            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
            // omp_set_num_threads(this->num_thread);
        }

        TimeMeasure fused_code() override
        {
            TimeMeasure t1;
            if (this->hasPerm)
            {
                // if (getWavefront() == 0)
                // {
                //     t1.start_timer();
                //     lower_csr_trsv_serial(*csrA_perm, x, y);
                //     t1.measure_elasped_time();
                // }
                // else
                {
                    t1.start_timer();
                    sptrsv_p2p_csr_group_merge_perm_alloc(csrA_perm, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                    t1.measure_elasped_time();
                }
            }
            else
            {

                // if (getWavefront() == 0)
                // {
                //     t1.start_timer();
                //     lower_csr_trsv_serial(*csrA, x, y);
                //     t1.measure_elasped_time();
                // }
                // else
                {
                    t1.start_timer();
                    sptrsv_p2p_csr_group_merge_no_perm_alloc(csrA, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                    t1.measure_elasped_time();
                }
                // t1.start_timer();
                // sptrsv_p2p_csr_group_merge_no_perm_alloc(csrA, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                // t1.measure_elasped_time();
            }

            return t1;
        }

    public:
        SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule(CSR *csrA, CSC *cscA, double *correct_x, int n, int nnz, double alpha, std::string algName, int num_thread, bool hasPerm)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->cscA = cscA;
            this->correct_x = correct_x;
            this->alpha = alpha;
            this->num_thread = num_thread;
            this->hasPerm = hasPerm;

            levelset_merge = new LevelSet();
            this->dag_group = new DAG();
            this->taskDAG = new DAG();
        }

        ~SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule()
        {
            delete levelset_merge;
            FREE(nthread_per_level);
            delete dag_group;
            delete task;
            delete taskDAG;
            if (schedule)
            {
                delete schedule;
            }
            if (csrA_perm)
            {
                delete csrA_perm;
            }
        }

        int getWavefront() const { return schedule->getP2PNum(); }
    };

    /*********************************** Reverse and Forward Tree with P2P and node Maping in Four rule (BAASWOAS) **********************************/
    class SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationBase : public FusionDemo
    {
    protected:
        LevelSet *levelset_merge;
        int *nthread_per_level;
        int ngroups;
        CSC *cscA;
        // CSR *csrA_perm;

        DAG *dag_group;
        DAG *taskDAG;
        Merge::Task *task;
        Merge::TaskSchedule *schedule = nullptr;
        std::vector<int> group_set, group_ptr;
        std::vector<double> cost; // cost for grouped node

        void orderingGroupSet(const int ngroups, const vector<int> &group_ptr, vector<int> &group_set)
        {
#pragma omp parallel for schedule(static)
            for (int g = 0; g < ngroups; g++)
            {
                std::sort(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1]);
            }
        }

        void groupPerm(const int ngroups, const int *origToThreadContPerm, const vector<int> &group_set, const vector<int> &group_ptr, vector<int> &group_set_p, vector<int> &group_ptr_p)
        {
            group_set_p.reserve(group_set.size());
            group_set_p.resize(group_set.size(), 0);
            group_ptr_p.reserve(group_ptr.size());
            group_ptr_p.resize(group_ptr.size(), 0);

#pragma omp parallel
            {
#pragma omp for
                for (int g = 0; g < ngroups; g++)
                {
                    // for(int )
                    int perm_group = origToThreadContPerm[g];
                    group_ptr_p[perm_group + 1] = group_ptr[g + 1] - group_ptr[g];
                }
                // #pragma omp barrier

#pragma omp single
                {
                    prefixSumSingle(group_ptr_p.data(), ngroups + 1);
                    assert(group_ptr_p.back() == group_ptr.back());
                }

#pragma omp for
                for (int g = 0; g < ngroups; g++)
                {
                    int perm_group = origToThreadContPerm[g];
                    std::copy(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1], group_set_p.begin() + group_ptr_p[perm_group]);
                }
            }
        }

        void buildset() override
        {
            printf("%s is begin\n",this->algName.c_str());
            TimeMeasure t1;
            t1.start_timer();
            // printf("dag node number: %d\n", n);
            DAG *dag = new DAG(n, nnz);
            dag->DAG_ptr.clear();
            dag->DAG_set.clear();
            DAG::partialSparsification_CSC(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);

            // create group ptr and group set
            group_ptr.resize(n + 1, 0);
            group_set.resize(n, 1);
#pragma omp for
            for (int i = 0; i < n; i++)
            {
                group_ptr[i + 1] = i + 1;
                group_set[i] = i;
            }

            // ngroups = ngroups_f;
            ngroups = n;
            delete dag_group;
            dag_group = dag;
            // printf("after two coarsening ngroups: %d\n", ngroups);
            dag_group->setNodeNum(ngroups);
            dag_group->updateEdges();
            LevelSet *levelset = new LevelSet();
            dag_group->findLevelsPostOrder(levelset);

            levelset_merge = levelset;
            // dag_group->findLevelsPostOrder(levelset);
            // dag_group = dag_group_f;
            Coarsen::costComputation(ngroups, nullptr, nullptr, csrA->rowptr, csrA->colidx, Coarsen::Kernel::SpTRSV_LL, group_ptr.data(), group_set.data(), true, cost);

            // // 在p2p模式下这里不应该进行直接的merge 而是根据线程数量进行分配 intra-transitive会自动消除
            // level_nthreads_group_TRSV(levelset, csrA, group_ptr.data(), group_set.data(), this->num_thread, this->nthread_per_level);
            level_nthreads_base(levelset_merge, csrA, this->num_thread, this->nthread_per_level);

            // 确保小的节点编号先执行
            orderingGroupSet(ngroups, group_ptr, group_set);

            // 转p2p
            task = new Merge::Task(ngroups, false);

            DAG *dag_group_inv = DAG::inverseDAG(*dag_group);
            // task alloc
            task->constructTask(levelset_merge, cost.data(), this->num_thread, nthread_per_level);

            // constrcut task DAG
            task->constructTaskDAG(dag_group, dag_group_inv, taskDAG, this->num_thread, false);
            delete dag_group_inv;

            // printf("taskNUm: %d\n", taskDAG->getNodeNum());
            // printf("cparents: %d\n", taskDAG->DAG_ptr.back());
            // fflush(stdout);

            // construct task schedule
            schedule = new TaskSchedule(taskDAG->getNodeNum(), taskDAG->DAG_ptr.back());
            schedule->constructTaskSchedule(taskDAG);

            if (this->hasPerm)
            {
                // 对稀疏矩阵进行重排
                std::vector<int> group_set_p, group_ptr_p; // 记录经过perm后的ptr和set集合，随后将类中的group_ptr和set进行std::move
                groupPerm(ngroups, task->origToThreadContPerm, group_set, group_ptr, group_set_p, group_ptr_p);
                std::vector<int> origToPerm(group_set_p.size(), 0);
                getInversePerm(origToPerm.data(), group_set_p.data(), group_set_p.size());

                assert(isPerm(group_set.data(), group_set.size()));
                assert(isPerm(group_set_p.data(), group_set_p.size()));
                assert(isPerm(origToPerm.data(), origToPerm.size()));
                // 对CSR进行深度拷贝
                // this->csrA_perm = new CSR(*csrA);
                // csrA_perm.
                csrA_perm = csrA->permute(origToPerm.data(), group_set_p.data()); // not sort to keep the diag element
                this->permToOrig = group_set_p;
                this->origToPerm = std::move(origToPerm);

                group_ptr = std::move(group_ptr_p);
                group_set = std::move(group_set_p); // inverse permuation : perm -> orig
            }
            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
            // omp_set_num_threads(this->num_thread);
        }

        TimeMeasure fused_code() override
        {
            TimeMeasure t1;
            if (this->hasPerm)
            {
                t1.start_timer();
                sptrsv_p2p_csr_group_merge_perm_alloc(csrA_perm, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                t1.measure_elasped_time();
            }
            else
            {
                t1.start_timer();
                sptrsv_p2p_csr_group_merge_no_perm_alloc(csrA, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                t1.measure_elasped_time();
            }

            return t1;
        }

    public:
        SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationBase(CSR *csrA, CSC *cscA, double *correct_x, int n, int nnz, double alpha, std::string algName, int num_thread, bool hasPerm)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->cscA = cscA;
            this->correct_x = correct_x;
            this->alpha = alpha;
            this->num_thread = num_thread;
            this->hasPerm = hasPerm;

            levelset_merge = new LevelSet();
            this->dag_group = new DAG();
            this->taskDAG = new DAG();
        }

        ~SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationBase()
        {
            delete levelset_merge;
            FREE(nthread_per_level);
            delete dag_group;
            delete task;
            delete taskDAG;
            if (schedule)
            {
                delete schedule;
            }
            if (csrA_perm)
            {
                delete csrA_perm;
            }
        }

        int getWavefront() const { return schedule->getP2PNum(); }
    };
    /*********************************** Reverse and Forward Tree with P2P and node Maping in Four rule (BAASWOA) **********************************/
    class SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationWOAggregation : public FusionDemo
    {
    protected:
        LevelSet *levelset_merge;
        int *nthread_per_level;
        int ngroups;
        CSC *cscA;
        // CSR *csrA_perm;

        DAG *dag_group;
        DAG *taskDAG;
        Merge::Task *task;
        Merge::TaskSchedule *schedule = nullptr;
        std::vector<int> group_set, group_ptr;
        std::vector<double> cost; // cost for grouped node

        void orderingGroupSet(const int ngroups, const vector<int> &group_ptr, vector<int> &group_set)
        {
#pragma omp parallel for schedule(static)
            for (int g = 0; g < ngroups; g++)
            {
                std::sort(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1]);
            }
        }

        void groupPerm(const int ngroups, const int *origToThreadContPerm, const vector<int> &group_set, const vector<int> &group_ptr, vector<int> &group_set_p, vector<int> &group_ptr_p)
        {
            group_set_p.reserve(group_set.size());
            group_set_p.resize(group_set.size(), 0);
            group_ptr_p.reserve(group_ptr.size());
            group_ptr_p.resize(group_ptr.size(), 0);

#pragma omp parallel
            {
#pragma omp for
                for (int g = 0; g < ngroups; g++)
                {
                    // for(int )
                    int perm_group = origToThreadContPerm[g];
                    group_ptr_p[perm_group + 1] = group_ptr[g + 1] - group_ptr[g];
                }
                // #pragma omp barrier

#pragma omp single
                {
                    prefixSumSingle(group_ptr_p.data(), ngroups + 1);
                    assert(group_ptr_p.back() == group_ptr.back());
                }

#pragma omp for
                for (int g = 0; g < ngroups; g++)
                {
                    int perm_group = origToThreadContPerm[g];
                    std::copy(group_set.begin() + group_ptr[g], group_set.begin() + group_ptr[g + 1], group_set_p.begin() + group_ptr_p[perm_group]);
                }
            }
        }

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            // printf("dag node number: %d\n", n);
            DAG *dag = new DAG(n, nnz);
            dag->DAG_ptr.clear();
            dag->DAG_set.clear();
            DAG::partialSparsification_CSC(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);
            // HDAGG::partialSparsification(n, nnz, cscA->colptr, cscA->rowidx, dag->DAG_ptr, dag->DAG_set, false);

            // /************************** reverse coarsen ************************** */
            // Coarsen::reverseTreeCoarseningBFS_all(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);
            // // HDAGG::treeBasedGroupingBFS(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);

            // // printf("reverse ngroups: %d\n", ngroups);
            // Coarsen::buildGroupDAGParallel(n, ngroups, group_ptr.data(), group_set.data(),
            //                                dag->DAG_ptr.data(), dag->DAG_set.data(), dag_group->DAG_ptr, dag_group->DAG_set);
            // dag_group->setNodeNum(ngroups);
            // delete dag;

            // /************************** forward coarsen ************************** */
            // std::vector<int> group_ptr_f, group_set_f;
            // DAG *dag_group_f = new DAG();
            // int ngroups_f = 0;
            // Coarsen::forwardTreeCoarseningBFS_all(ngroups, dag_group->DAG_ptr, dag_group->DAG_set, ngroups_f, group_ptr_f, group_set_f, false);
            // printf("forward ngroups: %d\n", ngroups_f);
            // // mapping group_f into group_ptr,
            // Coarsen::groupRemapping(group_ptr, group_set, group_ptr_f, group_set_f, ngroups_f);

            // // build new DAG based on group_DAG, with group_f
            // Coarsen::buildGroupDAGParallel(ngroups, ngroups_f, group_ptr_f.data(), group_set_f.data(),
            //                                dag_group->DAG_ptr.data(), dag_group->DAG_set.data(), dag_group_f->DAG_ptr, dag_group_f->DAG_set);

            // create group ptr and group set
            group_ptr.resize(n + 1, 0);
            group_set.resize(n, 0);
#pragma omp for
            for (int i = 0; i < n; i++)
            {
                group_set[i] = i;
                group_ptr[i + 1] = i + 1;
            }
            delete dag_group;
            dag_group = dag;

            ngroups = n;
            // printf("after two coarsening ngroups: %d\n", ngroups);
            dag_group->setNodeNum(ngroups);
            dag_group->updateEdges();
            LevelSet *levelset = new LevelSet();
            dag_group->findLevelsPostOrder(levelset);

            levelset_merge = levelset;
            // dag_group->findLevelsPostOrder(levelset);
            // delete dag_group;
            // dag_group = dag_group_f;
            Coarsen::costComputation(ngroups, nullptr, nullptr, csrA->rowptr, csrA->colidx, Coarsen::Kernel::SpTRSV_LL, group_ptr.data(), group_set.data(), true, cost);

            // 在p2p模式下这里不应该进行直接的merge 而是根据线程数量进行分配 intra-transitive会自动消除
            level_nthreads_group_TRSV(levelset, csrA, group_ptr.data(), group_set.data(), this->num_thread, this->nthread_per_level);

            // 确保小的节点编号先执行
            orderingGroupSet(ngroups, group_ptr, group_set);

            // 转p2p
            task = new Merge::Task(ngroups, false);

            DAG *dag_group_inv = DAG::inverseDAG(*dag_group);
            // task alloc
            // task->constructTask(levelset_merge, cost.data(), this->num_thread, nthread_per_level);
            // task->constructMappingSerial(dag_group, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            // task->constructMappingByFinishTime(dag_group, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            // task->constructMappingByFourRule(dag_group,dag_group_inv, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            task->constructMappingByFourRuleParallel(dag_group, dag_group_inv, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());

            // constrcut task DAG
            task->constructTaskDAG(dag_group, dag_group_inv, taskDAG, this->num_thread, false);
            delete dag_group_inv;

            // // printf("taskNUm: %d\n", taskDAG->getNodeNum());
            // // printf("cparents: %d\n", taskDAG->DAG_ptr.back());
            // // fflush(stdout);

            // construct task schedule
            schedule = new TaskSchedule(taskDAG->getNodeNum(), taskDAG->DAG_ptr.back());
            schedule->constructTaskSchedule(taskDAG);

            if (this->hasPerm)
            {
                // 对稀疏矩阵进行重排
                std::vector<int> group_set_p, group_ptr_p; // 记录经过perm后的ptr和set集合，随后将类中的group_ptr和set进行std::move
                groupPerm(ngroups, task->origToThreadContPerm, group_set, group_ptr, group_set_p, group_ptr_p);
                std::vector<int> origToPerm(group_set_p.size(), 0);
                getInversePerm(origToPerm.data(), group_set_p.data(), group_set_p.size());

                assert(isPerm(group_set.data(), group_set.size()));
                assert(isPerm(group_set_p.data(), group_set_p.size()));
                assert(isPerm(origToPerm.data(), origToPerm.size()));
                // 对CSR进行深度拷贝
                // this->csrA_perm = new CSR(*csrA);
                // csrA_perm.
                csrA_perm = csrA->permute(origToPerm.data(), group_set_p.data()); // not sort to keep the diag element
                this->permToOrig = group_set_p;
                this->origToPerm = std::move(origToPerm);

                group_ptr = std::move(group_ptr_p);
                group_set = std::move(group_set_p); // inverse permuation : perm -> orig
            }
            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
            // omp_set_num_threads(this->num_thread);
        }

        TimeMeasure fused_code() override
        {
            TimeMeasure t1;
            if (this->hasPerm)
            {
                t1.start_timer();
                sptrsv_p2p_csr_group_merge_perm_alloc(csrA_perm, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                t1.measure_elasped_time();
            }
            else
            {
                t1.start_timer();
                sptrsv_p2p_csr_group_merge_no_perm_alloc(csrA, x, y, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                t1.measure_elasped_time();
            }

            return t1;
        }

    public:
        SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationWOAggregation(CSR *csrA, CSC *cscA, double *correct_x, int n, int nnz, double alpha, std::string algName, int num_thread, bool hasPerm)
            : FusionDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->cscA = cscA;
            this->correct_x = correct_x;
            this->alpha = alpha;
            this->num_thread = num_thread;
            this->hasPerm = hasPerm;

            levelset_merge = new LevelSet();
            this->dag_group = new DAG();
            this->taskDAG = new DAG();
        }

        ~SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationWOAggregation()
        {
            delete levelset_merge;
            FREE(nthread_per_level);
            delete dag_group;
            delete task;
            delete taskDAG;
            if (schedule)
            {
                delete schedule;
            }
            if (csrA_perm)
            {
                delete csrA_perm;
            }
        }

        int getWavefront() const { return schedule->getP2PNum(); }
    };

} // namespace SpTRSVRT
