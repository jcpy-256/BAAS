#pragma once

#include <iostream>
#include <vector>
#include <utility>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <list>
#include <set>
#include <unordered_set>
#include <limits>

#include "SPM.hpp"
#include "MarcoUtils.hpp"
using namespace SPM;
using namespace std;

namespace Merge
{

    /**
     * this class is used for grouping tasks by level
     */
    class Task
    {
    private:
        int taskNum;

    public:
        bool isBarrier; // is using barrier synchronization
        int n;          // node num;

        vector<int> taskBoundaries; // begin and end rows for( threadcntToOrigPerm)
        vector<int> threadBoundaries;
        vector<int> taskToLevel; // responding to levels
        // vector<vector<int>> taskParents;    // taks parents to construct task DAG
        vector<int> origNodeToTask; // node idx map to task id
        int *origToThreadContPerm;
        int *threadContToOrigPerm;

        Task(int n, bool isBarrier = true);
        ~Task();

        int getTaskNum() const;
        void setTaskNum(int taskNum);

        void dealloc()
        {
            taskBoundaries.clear();
            taskBoundaries.shrink_to_fit();

            threadBoundaries.clear();
            threadBoundaries.shrink_to_fit();

            taskToLevel.clear();
            taskToLevel.shrink_to_fit();

            origNodeToTask.clear();
            origNodeToTask.shrink_to_fit();
            FREE(origToThreadContPerm);
            FREE(threadContToOrigPerm);
        }
        /**
         * @note: construct Task level-by-level
         * @param leveset: the levelset class
         * @param cost: cost per node
         * @param nthread: the max thread num
         * @param nthread_per_level: the thread num assigned for each level
         */
        void constructTask(const LevelSet *levelset, const double *cost, const int nthread, const int *nthread_per_level);

        /**
         * @note: convert taskRows structure to threadBoundaries and taskBoundaries
         * @param taskRows: the task organized in level-by-level and the node are arranged according to threads
         * @param nlevels: the number of levels
         * @param nodes: the number of nodes needed to execute.
         * @param inversePerm: the nodes are arranged by threads (mapping from exectuation order to original node order)
         * @param nthread: the maximum of execution thread.
         */
        void taskMapToBoundaries(const std::vector<std::pair<int, int>> &taskRows, const int nlevels, const int nodes, const int *inversePerm, const int nthread);

        void constructTaskDAG(DAG *&dag_group, DAG *&dag_csr, DAG *&taskDAG, const int nthread, bool transitiveReduction = false);

        // void constructTaskDAGParallel(DAG *&dag_group, DAG *&taskDAG, const int nthread, bool transitiveReduction);
        void constructTaskDAGParallel(DAG *&dag_group, DAG *&dag_csr, DAG *&taskDAG, const int nthread, bool transitiveReduction);

        void deleteIntraThreadEdge();

        void twoHopTransitiveRedcution();
        /**
         * @note: construct Task with node mapping method. In this case, we will assign node based on parents assignment to premote the locality,
         * load balance and reduction of intra-thread dependency. For node in each level, we compute the priority function which is determined by
         * the level interval and thread index of parent nodes.
         * @param dag_group: the DAG class of grouped lower triangular matrix in CSC format
         * @param leveset: the levelset class
         * @param cost: cost per node
         * @param nthread: the max thread num
         * @param nthread_per_level: the thread num assigned for each level
         * @param group_ptr: the group ptr for group_set
         * @param group_set: the node set listed by group ptr
         */
        void constructMapping(DAG *&dag_group, const LevelSet *levelset, const double *cost, const int nthread, const int *nthread_per_level, const int *group_ptr, const int *groupset);

        /**
         * @note: (serial) construct Task with node mapping method. In this case, we will assign node based on parents assignment to premote the locality,
         * load balance and reduction of intra-thread dependency. For node in each level, we compute the priority function which is determined by
         * the level interval and thread index of parent nodes.
         * @param dag_group: the DAG class of grouped lower triangular matrix in CSC format
         * @param leveset: the levelset class
         * @param cost: cost per node
         * @param nthread: the max thread num
         * @param nthread_per_level: the thread num assigned for each level
         * @param group_ptr: the group ptr for group_set
         * @param group_set: the node set listed by group ptr
         */
        void constructMappingSerial(DAG *&dag_group, const LevelSet *levelset, const double *cost, const int nthread, const int *nthread_per_level, const int *group_ptr, const int *groupset);

        /**
         * @note: (serial) construct Task with node mapping method. In this case, we will assign node based on parents assignment to premote the locality,
         * load balance and reduction of intra-thread dependency. For node in each level, we compute the priority function which is determined by
         * the level interval and thread index of parent nodes. Additionally, we merge the level with continuous one executation thread. By this tactic,
         * the task generated in Task DAG is reduced and the p2p synchronizations decreases simultaneously.
         * @param dag_group: the DAG class of grouped lower triangular matrix in CSC format
         * @param leveset: the levelset class
         * @param cost: cost per node grouped
         * @param nthread: the max thread num
         * @param nthread_per_level: the thread num assigned for each level
         * @param group_ptr: the group ptr for group_set
         * @param group_set: the node set listed by group ptr
         */
        void constructMappingMergeSerial(DAG *&dag_group, DAG *&dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset);

        /**
         * @note: (serial) construct Task with node mapping method. In this pattern, we will assign node to the thread that finished earliest.
         * @param dag_group: the DAG class of grouped lower triangular matrix in CSC format
         * @param leveset: the levelset class
         * @param cost: cost per node grouped
         * @param nthread: the max thread num
         * @param nthread_per_level: the thread num assigned for each level
         * @param group_ptr: the group ptr for group_set
         * @param group_set: the node set listed by group ptr
         */
        void constructMappingByFinishTime(DAG *&dag_group, DAG *&dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset);

        /**
         * @note: (serial) construct Task with node mapping method. We will assign the node in level-by-level. The node in the same level will be
         * assign according to a complicated (complex) score formula. It consists of four components: thread affinity, load penalty, overload penalty, and global balance
         * @param dag_group: the DAG class of grouped lower triangular matrix in CSC format
         * @param leveset: the levelset class
         * @param cost: cost per node grouped
         * @param nthread: the max thread num
         * @param nthread_per_level: the thread num assigned for each level
         * @param group_ptr: the group ptr for group_set
         * @param group_set: the node set listed by group ptr
         */
        void constructMappingByFourRule(DAG *&dag_group, DAG *& dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset);

        void constructMappingByFourRuleParallel(DAG *&dag_group, DAG *&dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset);

         void constructMappingByFourRuleParallelBack(DAG *&dag_group, DAG *&dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset);
    };

    struct TaskSchedule
    {
        int cparents;
        int ntasks;
        int **parents;
        int *nparents;
        int *parentsBuf;

        int **parentsBackward = nullptr;
        int *nparentsBackward = nullptr;
        int *parentsBufBackward = nullptr;

        volatile int *taskFinsished;

        TaskSchedule(int ntasks, int cparents);
        ~TaskSchedule();

        void constructTaskSchedule(const DAG *taskDAG);
        void constructInverseSchedule(const DAG *taskDAG);
        int getP2PNum();
    };

} // namespace Merge
