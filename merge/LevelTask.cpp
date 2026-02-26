#include <iostream>
#include <vector>
#include <utility>

#include "LevelTask.hpp"
#include "MergeMarco.hpp"

using namespace std;

namespace Merge
{

    Task::Task(int n, bool isBarrier)
    {
        this->n = n;
        this->isBarrier = isBarrier;
        this->origToThreadContPerm = MALLOC(int, n);
        this->threadContToOrigPerm = MALLOC(int, n);
        CHECK_POINTER(this->origToThreadContPerm);
        CHECK_POINTER(this->threadContToOrigPerm);
    }

    Task::~Task()
    {
        dealloc();
        // FREE(origToThreadContPerm);
        // FREE(threadContToOrigPerm);
    }

    int Task::getTaskNum() const
    {
        return this->taskNum;
    }

    void Task::setTaskNum(int taskNum)
    {

        this->taskNum = taskNum;
        assert(taskBoundaries.size() - 1 == taskNum);
    }

    void Task::constructTask(const LevelSet *levelset, const double *cost, const int nthread, const int *nthread_per_level)
    {
        int *level_ptr = levelset->level_ptr;
        int *permToOrig = levelset->permToOrig;
        int nlevels = levelset->getLevels();
        int nodes = levelset->getNodeNum();
        // taskBoundaries.reserve();
        std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        // taskRows.resize(nlevels * nthread);

        double *costBuffer = MALLOC(double, nodes + nthread);
        CHECK_POINTER(costBuffer);

// #ifdef _OPENMP
#pragma omp parallel for
        // #endif
        for (int l = 0; l < nlevels; l++)
        {
            int exec_thread = nthread_per_level[l];
            int levelBegin = level_ptr[l], levelEnd = level_ptr[l + 1];
            double load = 0;
            // compute load
            for (int i = levelBegin; i < levelEnd; i++)
            {
                // int node = permToOrig[i];
                costBuffer[i] = load;
                load += cost[permToOrig[i]];
            }
            double loadPerThread = (load + exec_thread - 1) / exec_thread;

            int preEnd = levelBegin;
            int r = levelBegin;
            load = 0;
            int t;
            // int max_thread = 0;
            for (t = 0; t < exec_thread; t++)
            {
                int newr = std::lower_bound(&costBuffer[r], &costBuffer[levelEnd], (t + 1) * loadPerThread) - costBuffer;
                r = newr;
                int begin = preEnd;
                int end = std::min(r, levelEnd);
                preEnd = end;
                taskRows[t * nlevels + l] = make_pair(begin, end);
                ++r;
            }
            for (t = exec_thread; t < nthread; t++)
            {
                taskRows[t * nlevels + l] = make_pair(levelEnd, levelEnd);
            }
        }
        FREE(costBuffer);

        std::vector<int> rowPartialSum(nthread + 1);
        rowPartialSum[0] = 0;
        threadBoundaries.resize(nthread + 1);
        threadBoundaries[0] = 0;

        if (isBarrier)
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int sum = 0;
                int cnt = 0;
                for (int i = 0; i < nlevels; i++)
                {
                    int diff = taskRows[tid * nlevels + i].second - taskRows[tid * nlevels + i].first;
                    sum += diff;
                    ++cnt;
                }
                rowPartialSum[tid + 1] = sum;
                threadBoundaries[tid + 1] = cnt;
            }
        }
        else
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int sum = 0;
                int cnt = 0;
                for (int i = 0; i < nlevels; i++)
                {
                    int diff = taskRows[tid * nlevels + i].second - taskRows[tid * nlevels + i].first;
                    sum += diff;
                    if (diff)
                        ++cnt;
                }
                rowPartialSum[tid + 1] = sum;
                threadBoundaries[tid + 1] = cnt;
            }
        }

        for (int tid = 0; tid < nthread; tid++)
        {

            rowPartialSum[tid + 1] += rowPartialSum[tid];
            threadBoundaries[tid + 1] += threadBoundaries[tid];
            // printf("rowPartialsum: %d\n", rowPartialSum[tid + 1]);
        }

        assert(rowPartialSum[nthread] == nodes);

        taskBoundaries.resize(threadBoundaries[nthread] + 1);
        if (isBarrier)
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++) // thread
            {
                int rowOffset = rowPartialSum[tid];
                int taskOffset = threadBoundaries[tid];
                for (int i = 0; i < nlevels; i++) // level
                {
                    taskBoundaries[taskOffset] = rowOffset;
                    ++taskOffset;
                    for (int j = taskRows[tid * nlevels + i].first; j < taskRows[tid * nlevels + i].second; j++)
                    {
                        origToThreadContPerm[permToOrig[j]] = rowOffset;
                        threadContToOrigPerm[rowOffset] = permToOrig[j];
                        rowOffset++;
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int rowOffset = rowPartialSum[tid];
                int taskOffset = threadBoundaries[tid];
                for (int i = 0; i < nlevels; i++)
                {
                    if (taskRows[tid * nlevels + i].second > taskRows[tid * nlevels + i].first)
                    {
                        taskBoundaries[taskOffset] = rowOffset;
                        ++taskOffset;

                        for (int j = taskRows[tid * nlevels + i].first; j < taskRows[tid * nlevels + i].second; j++)
                        {
                            origToThreadContPerm[permToOrig[j]] = rowOffset;
                            threadContToOrigPerm[rowOffset] = permToOrig[j];
                            assert(permToOrig[j] >= 0);
                            ++rowOffset;
                        }
                    }
                }
            }
        }
        // assert(rowPartialSum[nthread + 1] == nodes);
        taskBoundaries[threadBoundaries[nthread]] = nodes;
        assert(isPerm(origToThreadContPerm, nodes));
        assert(isPerm(threadContToOrigPerm, nodes));
        this->setTaskNum(threadBoundaries.back());
    }

    void Task::taskMapToBoundaries(const std::vector<std::pair<int, int>> &taskRows, const int nlevels, const int nodes, const int *inversePerm, const int nthread)
    {


        std::vector<int> rowPartialSum(nthread + 1);
        rowPartialSum[0] = 0;
        threadBoundaries.resize(nthread + 1);
        threadBoundaries[0] = 0;

        if (isBarrier)
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int sum = 0;
                int cnt = 0;
                for (int i = 0; i < nlevels; i++)
                {
                    int diff = taskRows[tid * nlevels + i].second - taskRows[tid * nlevels + i].first;
                    sum += diff;
                    ++cnt;
                }
                rowPartialSum[tid + 1] = sum;
                threadBoundaries[tid + 1] = cnt;
            }
        }
        else
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int sum = 0;
                int cnt = 0;
                for (int i = 0; i < nlevels; i++)
                {
                    int diff = taskRows[tid * nlevels + i].second - taskRows[tid * nlevels + i].first;
                    sum += diff;
                    if (diff)
                        ++cnt;
                }
                rowPartialSum[tid + 1] = sum;
                threadBoundaries[tid + 1] = cnt;
            }
        }

        for (int tid = 0; tid < nthread; tid++)
        {

            rowPartialSum[tid + 1] += rowPartialSum[tid];
            threadBoundaries[tid + 1] += threadBoundaries[tid];
            // printf("rowPartialsum: %d\n", rowPartialSum[tid + 1]);
        }

        assert(rowPartialSum[nthread] == nodes);

        taskBoundaries.resize(threadBoundaries[nthread] + 1);
        if (isBarrier)
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++) // thread
            {
                int rowOffset = rowPartialSum[tid];
                int taskOffset = threadBoundaries[tid];
                for (int i = 0; i < nlevels; i++) // level
                {
                    taskBoundaries[taskOffset] = rowOffset;
                    ++taskOffset;
                    for (int j = taskRows[tid * nlevels + i].first; j < taskRows[tid * nlevels + i].second; j++)
                    {
                        origToThreadContPerm[inversePerm[j]] = rowOffset;
                        threadContToOrigPerm[rowOffset] = inversePerm[j];
                        rowOffset++;
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int rowOffset = rowPartialSum[tid];
                int taskOffset = threadBoundaries[tid];
                for (int i = 0; i < nlevels; i++)
                {
                    if (taskRows[tid * nlevels + i].second > taskRows[tid * nlevels + i].first)
                    {
                        taskBoundaries[taskOffset] = rowOffset;
                        ++taskOffset;

                        for (int j = taskRows[tid * nlevels + i].first; j < taskRows[tid * nlevels + i].second; j++)
                        {
                            origToThreadContPerm[inversePerm[j]] = rowOffset;
                            threadContToOrigPerm[rowOffset] = inversePerm[j];
                            assert(inversePerm[j] >= 0);
                            ++rowOffset;
                        }
                    }
                }
            }
        }
        // assert(rowPartialSum[nthread + 1] == nodes);
        taskBoundaries[threadBoundaries[nthread]] = nodes;
        assert(isPerm(origToThreadContPerm, nodes));
        assert(isPerm(threadContToOrigPerm, nodes));
        this->setTaskNum(threadBoundaries.back());
    }
    void Task::constructTaskDAG(DAG *&dag_group, DAG *&dag_csr ,DAG *&taskDAG, const int nthread, bool transitiveReduction)
    {
        int nnz = dag_group->edges;
        int node_grouped = dag_group->getNodeNum();
        int ntask = this->getTaskNum();
        // const int
        // DAG *dag_csr = DAG::inverseDAG(*dag_group, true);
        const int *rowptr = dag_csr->DAG_ptr.data();
        const int *colidx = dag_csr->DAG_set.data();

        // taskDAG = new DAG();
        taskDAG->format = DAG_MAT::DAG_CSR;
        taskDAG->DAG_ptr.resize(ntask + 1);
        taskDAG->DAG_set.resize(nnz);
        taskDAG->setNodeNum(ntask);

        std::vector<int> &taskRowPtr = taskDAG->DAG_ptr;
        std::vector<int> &taskColIdx = taskDAG->DAG_set;

        int *perThreadOrigRowPtrSum = MALLOC(int, nthread + 1);
        perThreadOrigRowPtrSum[0] = 0;

        std::vector<int> taskAdjLength(ntask + 1, 0);
        //         // taskDag.

        // pre origNodeToTask
        origNodeToTask.resize(node_grouped);
        int maxSize = 0;
        std::vector<int> maxSizeVec(nthread, 0);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int v1PerThread = (ntask + nthread - 1) / nthread;
            int v1Begin = min(tid * v1PerThread, ntask);
            int v1End = min(v1Begin + v1PerThread, ntask);

            int origRowPtrCnt = 0;
            int localMaxSize = 0;
            for (int v1 = v1Begin; v1 < v1End; ++v1)
            {
                int size = 0;
                for (int i1Perm = taskBoundaries[v1]; i1Perm < taskBoundaries[v1 + 1]; ++i1Perm)
                {
                    int i1 = threadContToOrigPerm[i1Perm];
                    size += rowptr[i1 + 1] - rowptr[i1];
                }
                taskRowPtr[v1] = origRowPtrCnt;
                origRowPtrCnt += size;
                // maxSizeVec[tid] =
                localMaxSize = std::max(localMaxSize, size);
            }
            maxSizeVec[tid] = localMaxSize;
            perThreadOrigRowPtrSum[tid + 1] = origRowPtrCnt;

#pragma omp barrier
#pragma omp single
            {
                for (int tid = 0; tid < nthread; ++tid)
                {
                    perThreadOrigRowPtrSum[tid + 1] += perThreadOrigRowPtrSum[tid];
                    // perThreadOrigInvRowPtrSum[tid + 1] += perThreadOrigInvRowPtrSum[tid];
                }

                // for (int i = 0; i < ntask; i++)
                // {
                //     maxSize = std::max(taskRowPtr[i], maxSize);
                // }
                for (int tid = 0; tid < nthread; tid++)
                {
                    maxSize = std::max(maxSize, maxSizeVec[tid]);
                }
            }

            // get task ptr
            for (int v1 = v1Begin; v1 < v1End; ++v1)
            {
                taskRowPtr[v1] += perThreadOrigRowPtrSum[tid];
            }

            // get origNodeToTask
            for (int task = v1Begin; task < v1End; task++)
            {
                for (int row = taskBoundaries[task]; row < taskBoundaries[task + 1]; row++)
                {
                    origNodeToTask[threadContToOrigPerm[row]] = task;
                }
            }
        }
#ifndef NDEBUG
        for (int task = 0; task < ntask; task++)
        {
            assert(taskRowPtr[task] <= nnz);
        }
#endif

        // assert(perThreadOrigRowPtrSum[nthread] == nnz);

        // #pragma omp parallel for

        std::vector<int> dependency(maxSize);

        for (int tid = 0; tid < nthread; tid++) // thread
        {
            int boundBeg = threadBoundaries[tid];
            int boundEnd = threadBoundaries[tid + 1];
            for (int task = boundBeg; task < boundEnd; task++) // task
            {
                int size = 0;
                for (int row = taskBoundaries[task]; row < taskBoundaries[task + 1]; row++) // row
                {
                    // 遍历task的所有row加入colidx对应的行对应的task到本task的rowPtr和colidx中
                    int orgNode = threadContToOrigPerm[row];
                    int begin = rowptr[orgNode];
                    int end = rowptr[orgNode + 1];

                    for (int j = begin; j < end; j++)
                    {
                        int v = origNodeToTask[colidx[j]];
                        if (v < boundBeg || v >= boundEnd) // 非本线程的依赖添加
                        {
                            dependency[size] = v;
                            size++;
                        }
                    }
                }

                std::sort(dependency.begin(), dependency.begin() + size);
                // int l = 0;
                int oldV = -1;
                bool intraAdded = task == boundEnd - 1; // 不是最后一个task
                for (int idx = 0; idx < size; idx++)
                {
                    int v = dependency[idx];
                    if (v == oldV)
                        continue;
                    oldV = v;
                    // l++;

                    if (!intraAdded && transitiveReduction && v >= task + 1)
                    {
                        if (v > task + 1) // intra add task -> task + 1
                        {
                            int tempL = ++taskAdjLength[task + 1];
                            taskDAG->DAG_set[taskDAG->DAG_ptr[task + 1] + tempL] = task;
                        }
                        intraAdded = true;
                    }
                    // add depedency
                    // int tempL = ++taskAdjLength[task];
                    int tempL = taskAdjLength[task]++;
                    // assert(taskDAG->DAG_ptr)
                    assert(taskRowPtr[task] + tempL < nnz);
                    taskColIdx[taskRowPtr[task] + tempL] = v;
                }
                if (!intraAdded && transitiveReduction)
                {
                    int tempL = ++taskAdjLength[task + 1];
                    assert(taskDAG->DAG_ptr[task + 1] + tempL >= 0);
                    assert(taskDAG->DAG_ptr[task + 1] + tempL < nnz);
                    taskDAG->DAG_set[taskDAG->DAG_ptr[task + 1] + tempL] = task;
                }
            }
        }
        dependency.clear();
        dependency.shrink_to_fit();

        // 对taskDAG进行空间的收缩
        exclusive_scan(taskAdjLength.data(), ntask + 1);
        std::vector<int> task_set(taskAdjLength.back());

#pragma omp parallel for
        for (int task = 0; task < ntask; task++)
        {
            auto beg = taskColIdx.begin() + taskRowPtr[task];
            std::copy(beg, beg + taskAdjLength[task + 1] - taskAdjLength[task], task_set.begin() + taskAdjLength[task]);
        }
        taskDAG->DAG_ptr = std::move(taskAdjLength);
        taskDAG->DAG_set = std::move(task_set);
        FREE(perThreadOrigRowPtrSum);
    }

    void Task::constructTaskDAGParallel(DAG *&dag_group, DAG *&dag_csr, DAG *&taskDAG, const int nthread, bool transitiveReduction)
    {
        int nnz = dag_group->edges;
        int node_grouped = dag_group->getNodeNum();
        int ntask = this->getTaskNum();
        // const int
        // DAG *dag_csr = DAG::inverseDAG(*dag_group, true);
        const int *rowptr = dag_csr->DAG_ptr.data();
        const int *colidx = dag_csr->DAG_set.data();

        // taskDAG = new DAG();
        taskDAG->format = DAG_MAT::DAG_CSR;
        taskDAG->DAG_ptr.resize(ntask + 1);
        taskDAG->DAG_set.resize(nnz);
        taskDAG->setNodeNum(ntask);

        std::vector<int> &taskRowPtr = taskDAG->DAG_ptr;
        std::vector<int> &taskColIdx = taskDAG->DAG_set;

        int *perThreadOrigRowPtrSum = MALLOC(int, nthread + 1);
        perThreadOrigRowPtrSum[0] = 0;

        std::vector<int> taskAdjLength(ntask + 1, 0);
        //         // taskDag.

        // pre origNodeToTask
        origNodeToTask.resize(node_grouped);
        int maxSize = 0;
        std::vector<int> maxSizeVec(nthread, 0);
        std::vector<int> task_set;

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int v1PerThread = (ntask + nthread - 1) / nthread;
            int v1Begin = min(tid * v1PerThread, ntask);
            int v1End = min(v1Begin + v1PerThread, ntask);

            int origRowPtrCnt = 0;
            int localMaxSize = 0;
            for (int v1 = v1Begin; v1 < v1End; ++v1)
            {
                int size = 0;
                for (int i1Perm = taskBoundaries[v1]; i1Perm < taskBoundaries[v1 + 1]; ++i1Perm)
                {
                    int i1 = threadContToOrigPerm[i1Perm];
                    size += rowptr[i1 + 1] - rowptr[i1];
                }
                taskRowPtr[v1] = origRowPtrCnt;
                origRowPtrCnt += size;
                // maxSizeVec[tid] =
                localMaxSize = std::max(localMaxSize, size);
            }
            maxSizeVec[tid] = localMaxSize;
            perThreadOrigRowPtrSum[tid + 1] = origRowPtrCnt;

#pragma omp barrier
#pragma omp single
            {
                for (int tid = 0; tid < nthread; ++tid)
                {
                    perThreadOrigRowPtrSum[tid + 1] += perThreadOrigRowPtrSum[tid];
                    // perThreadOrigInvRowPtrSum[tid + 1] += perThreadOrigInvRowPtrSum[tid];
                }
                for (int tid = 0; tid < nthread; tid++)
                {
                    maxSize = std::max(maxSize, maxSizeVec[tid]);
                }
            }

            // get task ptr
            for (int v1 = v1Begin; v1 < v1End; ++v1)
            {
                taskRowPtr[v1] += perThreadOrigRowPtrSum[tid];
            }

            // get origNodeToTask
            for (int task = v1Begin; task < v1End; task++)
            {
                for (int row = taskBoundaries[task]; row < taskBoundaries[task + 1]; row++)
                {
                    origNodeToTask[threadContToOrigPerm[row]] = task;
                }
            }
        }
#ifndef NDEBUG
        for (int task = 0; task < ntask; task++)
        {
            assert(taskRowPtr[task] <= nnz);
        }
#endif

        // assert(perThreadOrigRowPtrSum[nthread] == nnz);

        // #pragma omp parallel for

#pragma omp parall
        {
            std::vector<int> dependency(maxSize);
#pragma omp for schedule(static)
            for (int tid = 0; tid < nthread; tid++) // thread
            {
                int boundBeg = threadBoundaries[tid];
                int boundEnd = threadBoundaries[tid + 1];
                for (int task = boundBeg; task < boundEnd; task++) // task
                {
                    int size = 0;
                    for (int row = taskBoundaries[task]; row < taskBoundaries[task + 1]; row++) // row
                    {
                        // 遍历task的所有row加入colidx对应的行对应的task到本task的rowPtr和colidx中
                        int orgNode = threadContToOrigPerm[row];
                        int begin = rowptr[orgNode];
                        int end = rowptr[orgNode + 1];

                        for (int j = begin; j < end; j++)
                        {
                            int v = origNodeToTask[colidx[j]];
                            if (v < boundBeg || v >= boundEnd) // 非本线程的依赖添加
                            {
                                dependency[size] = v;
                                size++;
                            }
                        }
                    }

                    std::sort(dependency.begin(), dependency.begin() + size);
                    // int l = 0;
                    int oldV = -1;
                    bool intraAdded = task == boundEnd - 1; // 不是最后一个task
                    for (int idx = 0; idx < size; idx++)
                    {
                        int v = dependency[idx];
                        if (v == oldV)
                            continue;
                        oldV = v;
                        // l++;

                        // if (!intraAdded && transitiveReduction && v >= task + 1)
                        // {
                        //     if (v > task + 1) // intra add task -> task + 1
                        //     {
                        //         //note: for inverse DAG
                        //         // int tempL = ++taskAdjLength[task + 1];
                        //         // int tempL =  __sync_fetch_and_add(&taskAdjLength[task + 1], 1);
                        //         // int tempL = taskAdjLength[task + 1];
                        //         // taskDAG->DAG_set[taskDAG->DAG_ptr[task + 1] + tempL] = task;
                        //     }
                        //     intraAdded = true;
                        // }
                        // add depedency
                        // int tempL = ++taskAdjLength[task];
                        int tempL = taskAdjLength[task];
                        __sync_fetch_and_add(&taskAdjLength[task], 1);
                        // assert(taskDAG->DAG_ptr)
                        assert(taskRowPtr[task] + tempL < nnz);
                        taskColIdx[taskRowPtr[task] + tempL] = v;
                    }
         
                }
            }
            dependency.clear();
            dependency.shrink_to_fit();

// 对taskDAG进行空间的收缩
#pragma omp barrier
#pragma omp single
            {
                exclusive_scan(taskAdjLength.data(), ntask + 1);
                // std::vector<int> task_set(taskAdjLength.back());
                task_set.resize(taskAdjLength.back());
            }

#pragma omp for
            for (int task = 0; task < ntask; task++)
            {
                auto beg = taskColIdx.begin() + taskRowPtr[task];
                std::copy(beg, beg + taskAdjLength[task + 1] - taskAdjLength[task], task_set.begin() + taskAdjLength[task]);
            }
        }

        // #pragma omp parallel for

        taskDAG->DAG_ptr = std::move(taskAdjLength);
        taskDAG->DAG_set = std::move(task_set);
        FREE(perThreadOrigRowPtrSum);
    }

    void Task::constructMapping(DAG *&dag_group, const LevelSet *levelset, const double *cost, const int nthread, const int *nthread_per_level, const int *group_ptr, const int *groupset)
    {
        int *level_ptr = levelset->level_ptr;
        int *permToOrig = levelset->permToOrig;
        int nlevels = levelset->getLevels();
        int nodes = levelset->getNodeNum();

        // taskBoundaries.reserve();
        // std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        std::vector<int> nodeByLevelThreadTask(nodes);

        // 作为所有线程在进行socre table扫描的时候暂存每个线程负责的node
        std::vector<int> nodePerThread(nodes + nthread - 1);
        // 每个线程中暂存的node数量
        std::vector<int> nNodeThreadSum(nthread + 1);
        // 每个线程中的node的总cost，依据costBuffer计算
        std::vector<double> nCostPerThread(nthread + 1);

        double *costBuffer = MALLOC(double, nodes + nthread);
        CHECK_POINTER(costBuffer);

        int tLimit = std::sqrt((double)nthread);
        // 找到nthread_per_level的最大值，分段进行并行查找
        int socreWidth = std::max_element(nthread_per_level, nthread_per_level + nlevels) - nthread_per_level;
        socreWidth = std::ceil(std::sqrt(std::max(socreWidth, nthread))); // 将最大的线程数量开方向上取整

        // 每个thread 平均的处理的最大节点数量，因此每个线程的scoreTable的开始位置为
        int maxAvgNodeNum = (nodes + nthread - 1) / nthread;

    }

    void Task::constructMappingSerial(DAG *&dag_group, const LevelSet *levelset, const double *cost, const int nthread, const int *nthread_per_level, const int *group_ptr, const int *groupset)
    {
        typedef struct NodeScore
        {
            int node;
            int thread;
            double score;
        } NodeScore;

        int *level_ptr = levelset->level_ptr;
        int *permToOrig = levelset->permToOrig;
        int *origToPerm = levelset->permutation;
        int *nodeToLevel = levelset->nodeToLevel;
        int nlevels = levelset->getLevels();
        int nodes = levelset->getNodeNum();

        DAG *dag_csr = DAG::inverseDAG(*dag_group, true);
        const int *DAGrowptr = dag_csr->DAG_ptr.data();
        const int *DAGcolidx = dag_csr->DAG_set.data();


        // taskBoundaries.reserve();
        // std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        // 将node按照映射后的线程进行排列，每个线程都是紧邻的并且线程内部按照node从小到大的顺序
        std::vector<int> nodeByLevelThreadTask(nodes);
        std::vector<int> origNodeToThread(nodes, -1);


        std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        std::vector<int> nodeBack(nodes);

        double *costBuffer = MALLOC(double, nodes + nthread);
        CHECK_POINTER(costBuffer);

        // process level one
        {
            int l = 0;
            int exec_thread = nthread_per_level[0];
            int levelBegin = level_ptr[0], levelEnd = level_ptr[1];
            // printf("level one node num: %d, exec_thread: %d\n", levelEnd - levelBegin, exec_thread);
            double load = 0.0;
            for (int i = levelBegin; i < levelEnd; i++)
            {
                costBuffer[i] = load;
                load += cost[permToOrig[i]];
            }
            double loadPerThread = (load + exec_thread - 1) / exec_thread;
            int preEnd = levelBegin;
            int r = levelBegin;
            load = 0;
            int t = 0;
            for (t = 0; t < exec_thread; t++)
            {
                int newr = std::lower_bound(&costBuffer[r], &costBuffer[levelEnd], (t + 1) * loadPerThread) - costBuffer;
                r = newr;
                int begin = preEnd;
                int end = std::min(r, levelEnd);
                preEnd = end;
                taskRows[t * nlevels + l] = make_pair(begin, end);
                for (int k = begin; k < end; k++)
                {
                    int nodeOrig = permToOrig[k];
                    // nodeBack[k] = nodeOrig;
                    nodeByLevelThreadTask[k] = nodeOrig;
                    origNodeToThread[nodeOrig] = t; // mapping reverse perm node to thread
                }
                ++r;
            }
            for (t = exec_thread; t < nthread; t++)
            {
                taskRows[t * nlevels + l] = make_pair(levelEnd, levelEnd);
            }
        }
        FREE(costBuffer);
        // process the remain
        std::vector<double> nodeScore(nthread);
        std::vector<int> threadIdxVec(nthread);
        // assign assist
        std::vector<std::list<int>> threadNode(nthread); // 每个线程分配的节点
        std::vector<double> threadLoad(nthread);         // 每个线程的负载
        std::vector<int> disperseNode;                   // 未被分配的节点
        std::vector<bool> threadAllocated(nthread);      // 线程是否已经任务分配完毕
        std::unordered_set<int> threadAllocating;        // 还没有分配完毕的线程

        std::vector<int> threadIndcies(nthread);  // 线程索引
        std::vector<double> threadScore(nthread); // 每个线程的得分，用来根据得分获得选中的线程
        std::unordered_set<int> threadChoose;     // 被选中的执行的线程
        threadChoose.reserve(nthread);
        threadAllocating.reserve(nthread);
        disperseNode.reserve(nodes);

        int checkNode = level_ptr[1] - level_ptr[0];
        // int pre_exec_thread = -1;
        // int pre_thread_num = nthread_per_level[0];
        for (int lvl = 1; lvl < nlevels; lvl++)
        {

            int levelBegin = level_ptr[lvl], levelEnd = level_ptr[lvl + 1];
            int nodeIn = levelEnd - levelBegin;
            int exec_thread = nthread_per_level[lvl];
            int nodeNumLevel = levelEnd - levelBegin;


            // 准备清空下面需要使用的数据
            threadChoose.clear();
            disperseNode.clear();
            threadAllocating.clear();
            // 清空上次的得分

            std::fill_n(threadLoad.begin(), nthread, 0);
            std::fill_n(threadScore.begin(), nthread, 0);
            std::fill_n(threadAllocated.begin(), nthread, false);

            int threadInd = 0;
            std::generate(threadIndcies.begin(), threadIndcies.end(), [&threadInd]()
                          { return threadInd++; });
            // threadIndcies

            for (auto &item : threadNode)
            {
                item.clear();
            }

            // if(exec_thread == 1)
            // {
            //     taskRows
            // }

            // 取两者最小值开方，并且向上取整
            int scoreWidth = std::ceil(std::sqrt(std::min(exec_thread, nthread)));
            std::vector<NodeScore> scoreTable(scoreWidth * nodeIn); // 按照列布局，每个node对应的score紧邻

            // 计算每个node的score
            for (int i = levelBegin; i < levelEnd; i++)
            {
                std::fill_n(nodeScore.begin(), nthread, 0);
                int nodeOrig = permToOrig[i];
                // nodeBack[i] = nodeOrig;
                int nodeLevel = nodeToLevel[nodeOrig];
                // printf("parents:%d\n", DAGrowptr[nodeOrig + 1] - DAGrowptr[nodeOrig] - 1);
                // 移除对角位置
                for (int parentPtr = DAGrowptr[nodeOrig]; parentPtr < DAGrowptr[nodeOrig + 1] - 1; parentPtr++)
                {
                    int parent = DAGcolidx[parentPtr];
                    // if (parent == nodeOrig)
                    //     printf("node error!!!\n");
                    int pThread = origNodeToThread[parent];
                    int levelDiff = nodeLevel - nodeToLevel[parent];
                    // double score = std::log2(1.0 * (group_ptr[parent + 1] - group_ptr[parent])) / std::log2(1.0 * (levelDiff + 1));
                    double score = 1 / std::log(1.0 * (levelDiff + 1));
                    // printf("nodeLevel:%d,levelDiff:%d, score: %lf\n", nodeLevel, levelDiff, score);
                    // double score = 1 / std::log(1.0 * (levelDiff + 1));
                    // double score =i;
                    nodeScore[pThread] += score;
                }
                // replace with for-loop
                int threadIdx = 0;
                std::generate(threadIdxVec.begin(), threadIdxVec.end(), [&threadIdx]()
                              { return threadIdx++; });
                // 对index 按照score降序
                std::partial_sort(threadIdxVec.begin(), threadIdxVec.begin() + scoreWidth, threadIdxVec.end(), [&nodeScore](int a, int b)
                                  { return nodeScore[a] > nodeScore[b]; });

                // 统计本node对应的线程的得分
                // 选取执行的线程
                double threadScoreTotal = 0.0;
                double alpha = 1.2;
                for (int m = 0; m < scoreWidth; m++)
                {
                    int tIdx = threadIdxVec[m];
                    // 将前score写入到scoreTable中
                    auto &entry = scoreTable[i - levelBegin + m * nodeNumLevel];
                    entry.node = nodeOrig, entry.thread = tIdx, entry.score = nodeScore[tIdx];
                    double scoreM = nodeScore[tIdx] * std::exp(-alpha * (m + 1));
                    threadScoreTotal += scoreM;
                }
                for (int m = 0; m < scoreWidth; m++)
                {
                    int tIdx = threadIdxVec[m];
                    double scoreM = nodeScore[tIdx] * std::exp(-alpha * (m + 1));
                    threadScore[tIdx] += scoreM / threadScoreTotal;
                }
            } // node for-loop
            // 对第一行的score table进行排序, 相同的线程排列在一起，得分高的在前
            std::sort(scoreTable.begin(), scoreTable.begin() + nodeNumLevel, [](NodeScore &a, NodeScore &b)
                      {if(a.thread != b.thread){return a.thread < b.thread; }else {return a.score > b.score;} });
            // 降序排列threadScore
            std::sort(threadIndcies.begin(), threadIndcies.end(), [&threadScore](int a, int b)
                      { return threadScore[a] > threadScore[b]; });

            threadChoose.insert(threadIndcies.begin(), threadIndcies.begin() + exec_thread);
            assert(threadChoose.size() == exec_thread);

            // 筛选节点到线程中
            double loadLevel = 0;
            for (int i = 0; i < nodeNumLevel; i++)
            {
                NodeScore &item = scoreTable[i];
                int threadIdx = item.thread;
                int node = item.node;
                int costNode = cost[node];
                loadLevel += costNode;
                if (threadChoose.find(threadIdx) != threadChoose.end())
                {
                    threadNode[threadIdx].push_back(node);
                    threadLoad[threadIdx] += costNode;
                    // printf("threadIdx: %d, node: %d, score:%lf\n", threadIdx, node, item.score);
                }
                else
                {
                    disperseNode.push_back(node);
                }
            }
            assert(!(disperseNode.size() == nodeNumLevel));
            // scoreTable总的node 没有错误
            // for(int b= levelBegin; b < levelEnd; b++)
            // {
            //     nodeBack[b] = scoreTable[b - levelBegin].node;
            // }

            double avgLoadPerThread = (loadLevel + exec_thread - 1) / exec_thread;
            // 这个循环存在问题
            for (auto &t : threadChoose)
            {
                if (threadLoad[t] >= avgLoadPerThread)
                {
                    threadAllocated[t] = true;
                    int node = threadNode[t].back();
                    // while()
                    while (threadLoad[t] - cost[node] >= avgLoadPerThread)
                    {
                        disperseNode.push_back(node);
                        threadNode[t].pop_back();
                        threadLoad[t] -= cost[node];
                        node = threadNode[t].back();
                    }
                }
                else
                {
                    threadAllocating.insert(t);
                }
            }

#ifndef NDEBUG
            int total = 0;
            for (auto &t : threadChoose)
            {
                total += threadNode[t].size();
            }
            assert(total + disperseNode.size() == nodeNumLevel);
#endif
            // 将未分配的负载按照scoreTable依次向下进行分配
            int disperseNum = 0;
            for (int i = 0; i < disperseNode.size(); i++)
            {
                int node = disperseNode[i];
                bool flag = false;
                for (int w = 1; w < scoreWidth; w++)
                {
                    int p = scoreTable[w * nodeNumLevel + origToPerm[node] - levelBegin].thread;
                    if (threadAllocating.find(p) != threadAllocating.end())
                    {
                        threadLoad[p] += cost[node];
                        threadNode[p].push_back(node);
                        if (threadLoad[p] >= avgLoadPerThread)
                        {
                            threadAllocated[p] = true;
                            threadAllocating.erase(p);
                        }
                        flag = true;
                        break; // 插入到了一个线程队列则，break循环
                    }
                }
                if (flag == false)
                {
                    disperseNode[disperseNum++] = node;
                }
            }
            assert(threadAllocating.size() > 0 || disperseNum == 0);
#ifndef NDEBUG
            // std::unordered_set<int> nodeCheckList;
            // nodeCheckList.reserve(nodeNumLevel);
            int totalNode = 0;
            for (auto &t : threadChoose)
            {
                totalNode += threadNode[t].size();
                // nodeCheckList.insert(threadNode[t].begin(), threadNode[t].end());
            }
            assert(totalNode + disperseNum == nodeNumLevel);
#endif
            // 将未被分配的任务均分到没有分满的线程中
            auto iter = threadAllocating.begin();
            int allocIdx = 0;
            int t = 0;
            while (allocIdx < disperseNum && iter != threadAllocating.end())
            {
                t = *iter;
                while (threadLoad[t] <= avgLoadPerThread && allocIdx < disperseNum) // 确保不超过disperse number
                {
                    int node = disperseNode[allocIdx++];
                    // idx++;
                    threadLoad[t] += cost[node];
                    threadNode[t].push_back(node);
                }
                iter++;
            }

            // 如果所有的线程均满，但是仍旧还有节点未进行分配
            while (allocIdx < disperseNum)
            {
                int node = disperseNode[allocIdx++];
                // int t = scoreTable[nodeNumLevel + origToPerm[node] - levelBegin].thread;
                threadNode[t].push_back(node);
            }
            // printf("allocIdx:%d\n", allocIdx);
            // printf("dispernum:%d\n", disperseNum);
            fflush(stdout);
            assert(allocIdx == disperseNum);
// assert(iter == threadAllocating.end());
#ifndef NDEBUG
            std::unordered_set<int> nodeCheckList;
            nodeCheckList.reserve(nodeNumLevel);
            // int totalNode = 0;
            for (auto &t : threadChoose)
            {
                // totalNode += threadNode[t].size();
                nodeCheckList.insert(threadNode[t].begin(), threadNode[t].end());
            }
            assert(nodeCheckList.size() == nodeNumLevel);
#endif

            // 释放空间
            scoreTable.clear();
            scoreTable.shrink_to_fit();

            // 将组织好的threadChoose, threadNode映射到taskRows, origNodeToThread 以及 nodeByLevelThreadTask
            int index = levelBegin;
            for (int et = 0; et < nthread; et++)
            {
                int taskIdx = et * nlevels + lvl;
                if (threadChoose.find(et) != threadChoose.end()) // 该线程是执行线程
                {
                    // int taskBeg = index;
                    int nodeNum = threadNode[et].size();
                    taskRows[taskIdx].first = index;
                    taskRows[taskIdx].second = index + nodeNum;
                    checkNode += nodeNum;

                    // 将节点放置到nodeByLevelThreadTask 中
                    for (auto &item : threadNode[et])
                    {
                        nodeByLevelThreadTask[index++] = item;
                        origNodeToThread[item] = et;
                    }
                    // 是否对一个节点内的数据进行排序
                    // std::sort(nodeByLevelThreadTask.begin() + taskBeg - nodeNum,
                    //           nodeByLevelThreadTask.begin() + taskBeg);
                }
                else
                {
                    taskRows[taskIdx].first = index;
                    taskRows[taskIdx].second = index;
                }
            }
        } // nlevel loop end
        // assert(isPerm(nodeBack.data(), nodes));
        // printf("nodes: %d\n", nodes);
        // printf("check nodes:%d\n", checkNode);
        fflush(stdout);
        assert(checkNode == nodes);

        // clear memory
        nodeScore.clear();
        nodeScore.shrink_to_fit();
        threadIdxVec.clear();
        threadIdxVec.shrink_to_fit();
        threadNode.clear();
        threadNode.shrink_to_fit();
        threadLoad.clear();
        threadLoad.shrink_to_fit();
        disperseNode.clear();
        disperseNode.shrink_to_fit();
        threadAllocated.clear();
        threadAllocated.shrink_to_fit();
        threadAllocating.clear();
        // threadAllocat
        threadIndcies.clear();
        threadIndcies.shrink_to_fit();
        threadScore.clear();
        threadScore.shrink_to_fit();
        threadChoose.clear();
        delete dag_csr;

        // 组织Task类中的threadBoundaries
        std::vector<int> rowPartialSum(nthread + 1);
        rowPartialSum[0] = 0;
        threadBoundaries.resize(nthread + 1);
        threadBoundaries[0] = 0;
        if (isBarrier)
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int sum = 0;
                int cnt = 0;
                for (int lvl = 0; lvl < nlevels; lvl++)
                {
                    int diff = taskRows[tid * nlevels + lvl].second - taskRows[tid * nlevels + lvl].first;
                    sum += diff;
                    cnt++;
                }
                rowPartialSum[tid + 1] = sum;
                threadBoundaries[tid + 1] = cnt;
            }
        }
        else
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int sum = 0;
                int cnt = 0;
                for (int lvl = 0; lvl < nlevels; lvl++)
                {
                    int diff = taskRows[tid * nlevels + lvl].second - taskRows[tid * nlevels + lvl].first;
                    sum += diff;
                    if (diff)
                        cnt++;
                }
                rowPartialSum[tid + 1] = sum;
                threadBoundaries[tid + 1] = cnt;
            }
        }
        // prefix sum
        for (int tid = 0; tid < nthread; tid++)
        {
            rowPartialSum[tid + 1] += rowPartialSum[tid];
            threadBoundaries[tid + 1] += threadBoundaries[tid];
        }
        // printf("nodes； %d\n", nodes);
        // printf("rowPartialSum total:%d\n", rowPartialSum[nthread]);
        fflush(stdout);

        assert(rowPartialSum[nthread] == nodes);
        // 任务的数量边界 threadBoundaries
        taskBoundaries.resize(threadBoundaries[nthread] + 1);
        assert(isPerm(nodeByLevelThreadTask.data(), nodes));

        if (isBarrier)
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int rowOffset = rowPartialSum[tid];
                int taskOffset = threadBoundaries[tid];
                for (int i = 0; i < nlevels; i++)
                {
                    taskBoundaries[taskOffset] = rowOffset;
                    ++taskOffset;
                    for (int j = taskRows[tid * nlevels + i].first; j < taskRows[tid * nlevels + i].second; j++)
                    {
                        origToThreadContPerm[nodeByLevelThreadTask[j]] = rowOffset;
                        threadContToOrigPerm[rowOffset] = nodeByLevelThreadTask[j];
                        rowOffset++;
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (int tid = 0; tid < nthread; tid++)
            {
                int rowOffset = rowPartialSum[tid];
                int taskOffset = threadBoundaries[tid];
                for (int i = 0; i < nlevels; i++)
                {
                    if (taskRows[tid * nlevels + i].second > taskRows[tid * nlevels + i].first)
                    {
                        taskBoundaries[taskOffset] = rowOffset;
                        ++taskOffset;

                        for (int j = taskRows[tid * nlevels + i].first; j < taskRows[tid * nlevels + i].second; j++)
                        {
                            origToThreadContPerm[nodeByLevelThreadTask[j]] = rowOffset;
                            threadContToOrigPerm[rowOffset] = nodeByLevelThreadTask[j];
                            assert(nodeByLevelThreadTask[j] >= 0);
                            ++rowOffset;
                        }
                    }
                }
            }
        } // taskBoundaries
        taskBoundaries[threadBoundaries[nthread]] = nodes;
        // std::copy(nodeByLevelThreadTask.begin(), nodeByLevelThreadTask.begin(), levelset->permToOrig);
        assert(isPerm(origToThreadContPerm, nodes));
        assert(isPerm(threadContToOrigPerm, nodes));
        this->setTaskNum(threadBoundaries.back());
        // printf("taskNum: %d\n", threadBoundaries.back());
    }

    void Task::constructMappingMergeSerial(DAG *&dag_group, DAG *&dag_csr,  const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset)
    {
        typedef struct NodeScore
        {
            int node;
            int thread;
            double score;
        } NodeScore;

        int *level_ptr = levelset->level_ptr;
        int *permToOrig = levelset->permToOrig;
        int *origToPerm = levelset->permutation;
        int *nodeToLevel = levelset->nodeToLevel;
        int nlevels = levelset->getLevels();
        int nodes = levelset->getNodeNum();

        // DAG *dag_csr = DAG::inverseDAG(*dag_group, true);
        const int *DAGrowptr = dag_csr->DAG_ptr.data();
        const int *DAGcolidx = dag_csr->DAG_set.data();

        // 将node按照映射后的线程进行排列，每个线程都是紧邻的并且线程内部按照node从小到大的顺序
        std::vector<int> nodeByLevelThreadTask(nodes);
        std::vector<int> origNodeToThread(nodes, -1);

        std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        std::vector<int> levelThreadOne(nlevels, -1); // record the thread choosed for level with one execution thread
        std::vector<int> nodeBack(nodes);

        double *costBuffer = MALLOC(double, nodes + nthread);
        CHECK_POINTER(costBuffer);

        // process level one
        {
            int l = 0;
            int exec_thread = nthread_per_level[0];
            int levelBegin = level_ptr[0], levelEnd = level_ptr[1];
            // printf("level one node num: %d, exec_thread: %d\n", levelEnd - levelBegin, exec_thread);
            double load = 0.0;
            for (int i = levelBegin; i < levelEnd; i++)
            {
                costBuffer[i] = load;
                load += cost[permToOrig[i]];
            }
            double loadPerThread = (load + exec_thread - 1) / exec_thread;
            int preEnd = levelBegin;
            int r = levelBegin;
            load = 0;
            int t = 0;
            if (exec_thread == 1)
            {
                levelThreadOne[l] = 0;
            }

            int avgThreadSkip = nthread / exec_thread;
            for (t = 0; t < exec_thread; t++)
            // for (t = 0; t < nthread; t+= avgThreadSkip)
            {
                int newr = std::lower_bound(&costBuffer[r], &costBuffer[levelEnd], (t + 1) * loadPerThread) - costBuffer;
                r = newr;
                int begin = preEnd;
                int end = std::min(r, levelEnd);
                preEnd = end;
                taskRows[t * nlevels + l] = make_pair(begin, end);
                for (int k = begin; k < end; k++)
                {
                    int nodeOrig = permToOrig[k];
                    nodeByLevelThreadTask[k] = nodeOrig;
                    origNodeToThread[nodeOrig] = t; // mapping reverse perm node to thread
                }
                ++r;
            }
            for (t = exec_thread; t < nthread; t++)
            {
                taskRows[t * nlevels + l] = make_pair(levelEnd, levelEnd);
            }
        }
        FREE(costBuffer);
        // process the remain
        std::vector<double> nodeScore(nthread);
        std::vector<int> threadIdxVec(nthread);
        // assign assist
        std::vector<std::list<int>> threadNode(nthread); // 每个线程分配的节点
        std::vector<double> threadLoad(nthread);         // 每个线程的负载
        std::vector<int> disperseNode;                   // 未被分配的节点
        // std::vector<bool> threadAllocated(nthread);      // 线程是否已经任务分配完毕
        std::unordered_set<int> threadAllocating; // 还没有分配完毕的线程

        std::vector<int> threadIndcies(nthread);  // 线程索引
        std::vector<double> threadScore(nthread); // 每个线程的得分，用来根据得分获得选中的线程
        std::unordered_set<int> threadChoose;     // 被选中的执行的线程
        threadChoose.reserve(nthread);
        threadAllocating.reserve(nthread);
        disperseNode.reserve(nodes);

        int checkNode = level_ptr[1] - level_ptr[0];
        for (int lvl = 1; lvl < nlevels; lvl++)
        {

            int levelBegin = level_ptr[lvl], levelEnd = level_ptr[lvl + 1];
            // int nodeNumLevel = levelEnd - levelBegin;
            int exec_thread = nthread_per_level[lvl];
            int nodeNumLevel = levelEnd - levelBegin;

            // 准备清空下面需要使用的数据
            threadChoose.clear();
            disperseNode.clear();
            threadAllocating.clear();
            // 清空上次的得分

            std::fill_n(threadLoad.begin(), nthread, 0);
            std::fill_n(threadScore.begin(), nthread, 0);
            // std::fill_n(threadAllocated.begin(), nthread, false);

            int threadInd = 0;
            std::generate(threadIndcies.begin(), threadIndcies.end(), [&threadInd]()
                          { return threadInd++; });
            // threadIndcies
            for (auto &item : threadNode)
            {
                item.clear();
            }

            // 取两者最小值开方，并且向上取整
            int scoreWidth = std::ceil(std::sqrt(std::min(exec_thread, nthread)));
            std::vector<NodeScore> scoreTable(scoreWidth * nodeNumLevel); // 按照列布局，每个node对应的score紧邻

            // 计算每个node的score
            for (int i = levelBegin; i < levelEnd; i++)
            {
                std::fill_n(nodeScore.begin(), nthread, 0);
                int nodeOrig = permToOrig[i];
                int nodeLevel = nodeToLevel[nodeOrig];
                // 移除对角位置
                for (int parentPtr = DAGrowptr[nodeOrig]; parentPtr < DAGrowptr[nodeOrig + 1] - 1; parentPtr++)
                {
                    int parent = DAGcolidx[parentPtr];
                    int pThread = origNodeToThread[parent];
                    int levelDiff = nodeLevel - nodeToLevel[parent];
                    if (levelDiff < LEVEL_WINDOW_SIZE)
                    {
                        // double score = std::log2(1.0 * (group_ptr[parent + 1] - group_ptr[parent])) / std::log2(1.0 * (levelDiff + 1));
                        double score = AMPLIFY_FACTOR * std::log2(1.0 * (group_ptr[parent + 1] - group_ptr[parent] + 1)) / std::log2(1.0 * (levelDiff + 1));

                        nodeScore[pThread] += score;
                    }
                }
                // replace with for-loop
                int threadIdx = 0;
                std::generate(threadIdxVec.begin(), threadIdxVec.end(), [&threadIdx]()
                              { return threadIdx++; });
                // 对index 按照score降序
                std::partial_sort(threadIdxVec.begin(), threadIdxVec.begin() + scoreWidth, threadIdxVec.end(), [&nodeScore](int a, int b)
                                  { return nodeScore[a] > nodeScore[b]; });

                // 统计本node对应的线程的得分
                // 选取执行的线程
                double threadScoreTotal = 0.0;
                double alpha = 1.2;
                for (int m = 0; m < scoreWidth; m++)
                {
                    int tIdx = threadIdxVec[m];
                    // 将前score写入到scoreTable中
                    auto &entry = scoreTable[i - levelBegin + m * nodeNumLevel];
                    entry.node = nodeOrig, entry.thread = tIdx, entry.score = nodeScore[tIdx];
                    double scoreM = nodeScore[tIdx] * std::exp(-alpha * (m + 1));
                    threadScoreTotal += scoreM;
                }
                for (int m = 0; m < scoreWidth; m++)
                {
                    int tIdx = threadIdxVec[m];
                    double scoreM = nodeScore[tIdx] * std::exp(-alpha * (m + 1));
                    threadScore[tIdx] += scoreM / threadScoreTotal;
                }
            } // node for-loop
            // 对第一行的score table进行排序, 相同的线程排列在一起，得分高的在前
            std::sort(scoreTable.begin(), scoreTable.begin() + nodeNumLevel, [](NodeScore &a, NodeScore &b)
                      {if(a.thread != b.thread){return a.thread < b.thread; }else {return a.score > b.score;} });
            // 降序排列threadScore
            std::sort(threadIndcies.begin(), threadIndcies.end(), [&threadScore](int a, int b)
                      { return threadScore[a] > threadScore[b]; });

            threadChoose.insert(threadIndcies.begin(), threadIndcies.begin() + exec_thread);
            assert(threadChoose.size() == exec_thread);
            if (exec_thread == 1)
            {
                assert(threadIndcies[0] >= 0);
                levelThreadOne[lvl] = threadIndcies[0];
            }

            // 筛选节点到线程中
            double loadLevel = 0;
            for (int i = 0; i < nodeNumLevel; i++)
            {
                NodeScore &item = scoreTable[i];
                int threadIdx = item.thread;
                int node = item.node;
                int costNode = cost[node];
                loadLevel += costNode;
                if (threadChoose.find(threadIdx) != threadChoose.end())
                {
                    threadNode[threadIdx].push_back(node);
                    threadLoad[threadIdx] += costNode;
                }
                else
                {
                    disperseNode.push_back(node);
                }
            }
            assert(!(disperseNode.size() == nodeNumLevel));
            double avgLoadPerThread = (loadLevel + exec_thread - 1) / exec_thread;
            // 这个循环存在问题
            for (auto &t : threadChoose)
            {
                if (threadLoad[t] >= avgLoadPerThread)
                {
                    int node = threadNode[t].back();
                    while (threadLoad[t] - cost[node] >= avgLoadPerThread)
                    {
                        disperseNode.push_back(node);
                        threadNode[t].pop_back();
                        threadLoad[t] -= cost[node];
                        node = threadNode[t].back();
                    }
                }
                else
                {
                    threadAllocating.insert(t);
                }
            }

#ifndef NDEBUG
            int total = 0;
            for (auto &t : threadChoose)
            {
                total += threadNode[t].size();
            }
            assert(total + disperseNode.size() == nodeNumLevel);
#endif
            // 将未分配的负载按照scoreTable依次向下进行分配
            int disperseNum = 0;
            for (int i = 0; i < disperseNode.size(); i++)
            {
                int node = disperseNode[i];
                bool flag = false;
                for (int w = 1; w < scoreWidth; w++)
                {
                    int p = scoreTable[w * nodeNumLevel + origToPerm[node] - levelBegin].thread;
                    if (threadAllocating.find(p) != threadAllocating.end())
                    {
                        threadLoad[p] += cost[node];
                        threadNode[p].push_back(node);
                        if (threadLoad[p] >= avgLoadPerThread)
                        {
                            // threadAllocated[p] = true;
                            threadAllocating.erase(p);
                        }
                        flag = true;
                        break; // 插入到了一个线程队列则，break循环
                    }
                }
                if (flag == false)
                {
                    disperseNode[disperseNum++] = node;
                }
            }
            assert(threadAllocating.size() > 0 || disperseNum == 0);
#ifndef NDEBUG
            int totalNode = 0;
            for (auto &t : threadChoose)
            {
                totalNode += threadNode[t].size();
            }
            assert(totalNode + disperseNum == nodeNumLevel);
#endif
            // 将未被分配的任务均分到没有分满的线程中
            auto iter = threadAllocating.begin();
            int allocIdx = 0;
            int t = 0;
            while (allocIdx < disperseNum && iter != threadAllocating.end())
            {
                t = *iter;
                while (threadLoad[t] <= avgLoadPerThread && allocIdx < disperseNum) // 确保不超过disperse number
                {
                    int node = disperseNode[allocIdx++];
                    threadLoad[t] += cost[node];
                    threadNode[t].push_back(node);
                }

                iter++;
            }

            // printf("remain node:%d\n", disperseNum - allocIdx);
            // printf("levelNodeNum:%d\n", nodeNumLevel);
            // 如果所有的线程均满，但是仍旧还有节点未进行分配
            while (allocIdx < disperseNum)
            {
                int node = disperseNode[allocIdx++];
                threadNode[t].push_back(node);
            }
            // printf("allocIdx:%d\n", allocIdx);
            // printf("dispernum:%d\n", disperseNum);
            fflush(stdout);
            assert(allocIdx == disperseNum);
#ifndef NDEBUG
            std::unordered_set<int> nodeCheckList;
            nodeCheckList.reserve(nodeNumLevel);
            for (auto &t : threadChoose)
            {
                nodeCheckList.insert(threadNode[t].begin(), threadNode[t].end());
            }
            assert(nodeCheckList.size() == nodeNumLevel);
#endif

            // 释放空间
            scoreTable.clear();
            scoreTable.shrink_to_fit();

            // 将组织好的threadChoose, threadNode映射到taskRows, origNodeToThread 以及 nodeByLevelThreadTask
            int index = levelBegin;
            for (int et = 0; et < nthread; et++)
            {
                int taskIdx = et * nlevels + lvl;
                if (threadChoose.find(et) != threadChoose.end()) // 该线程是执行线程
                {
                    int nodeNum = threadNode[et].size();
                    taskRows[taskIdx].first = index;
                    taskRows[taskIdx].second = index + nodeNum;

                    // 将节点放置到nodeByLevelThreadTask 中
                    for (auto &item : threadNode[et])
                    {
                        nodeByLevelThreadTask[index++] = item;
                        origNodeToThread[item] = et;
                    }
                    // 是否对一个节点内的数据进行排序
                    // std::sort(nodeByLevelThreadTask.begin() + taskBeg - nodeNum,
                    //           nodeByLevelThreadTask.begin() + taskBeg);
                }
                else
                {
                    taskRows[taskIdx].first = index;
                    taskRows[taskIdx].second = index;
                }
            }
        } // nlevel loop end

        // clear memory
        nodeScore.clear();
        nodeScore.shrink_to_fit();
        threadIdxVec.clear();
        threadIdxVec.shrink_to_fit();
        threadNode.clear();
        threadNode.shrink_to_fit();
        threadLoad.clear();
        threadLoad.shrink_to_fit();
        disperseNode.clear();
        disperseNode.shrink_to_fit();
        threadAllocating.clear();
        threadIndcies.clear();
        threadIndcies.shrink_to_fit();
        threadScore.clear();
        threadScore.shrink_to_fit();
        threadChoose.clear();
        // delete dag_csr;
        /***************************** merge level task with continuous one execution thread *******************/
        int *level_ptr_merged = MALLOC(int, nlevels + 1);
        int *nodeToLevel_merged = MALLOC(int, nodes);
        int *merge_ptr = MALLOC(int, nlevels + 1);
        CHECK_POINTER(level_ptr_merged);
        CHECK_POINTER(nodeToLevel_merged);
        CHECK_POINTER(merge_ptr);
        std::fill_n(level_ptr_merged, nlevels + 1, 0);
        std::fill_n(merge_ptr, nlevels + 1, 0);
#ifndef NDEBUG
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            if (nthread_per_level[lvl] == 1)
            {
                assert(levelThreadOne[lvl] != -1);
                // printf(" %d ", levelThreadOne[lvl]);
            }
        }
        // printf("\n");
#endif
        int *nthread_per_level_tmp = MALLOC(int, nlevels);
        CHECK_POINTER(nthread_per_level_tmp);
        std::copy(nthread_per_level, nthread_per_level + nlevels, nthread_per_level_tmp);

        int level_m = 0;
        for (int lvl = 1; lvl < nlevels;)
        {
            if (nthread_per_level_tmp[lvl] == 1 && nthread_per_level_tmp[level_m] == 1)
            {

                // 合并taskRows 到level_m的线程
                int preThread = levelThreadOne[level_m];
                int postThread = levelThreadOne[lvl];
                assert(preThread != -1);
                assert(postThread != -1);
                // 找到taskRows
                // 这里taskRow原本的位置似乎不是level_m，应该是merge_ptr[level_m]
                std::pair<int, int> &preTask = taskRows[preThread * nlevels + merge_ptr[level_m]];
                // assert(level_ptr[lvl] - level_ptr_merged[level_m] == preTask.second - preTask.first);

                std::pair<int, int> &postTask = taskRows[postThread * nlevels + lvl];
                // assert(level_ptr[lvl + 1] - level_ptr[lvl] == postTask.second - postTask.first);
                assert(preTask.second == postTask.first);

                preTask.second = postTask.second;
                postTask.first = 0;
                postTask.second = 0;
                lvl++;
            }
            else
            {
                level_m++;
                nthread_per_level_tmp[level_m] = nthread_per_level_tmp[lvl];
                levelThreadOne[level_m] = levelThreadOne[lvl];
                level_ptr_merged[level_m] = level_ptr[lvl];
                merge_ptr[level_m] = lvl;
                lvl++;
            }
        }

        level_m++;
        level_ptr_merged[level_m] = level_ptr[nlevels];
        merge_ptr[level_m] = nlevels;

        int *level_ptr_merged_alias = MALLOC(int, level_m + 1);
        CHECK_POINTER(level_ptr_merged_alias);
        memcpy(level_ptr_merged_alias, level_ptr_merged, (level_m + 1) * sizeof(int));
        FREE(level_ptr_merged);
        FREE(nodeToLevel_merged);
        FREE(merge_ptr);
        FREE(level_ptr_merged_alias);
        FREE(nthread_per_level_tmp);
        // levelset->level_ptr = level_ptr_merged_alias;
        // 组织Task类中的threadBoundaries 和 taskBoundaries
        this->taskMapToBoundaries(taskRows, nlevels, nodes, nodeByLevelThreadTask.data(), nthread);
    }

    void Task::constructMappingByFinishTime(DAG *&dag_group, DAG *&dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset)
    {
        int *level_ptr = levelset->level_ptr;
        int *permToOrig = levelset->permToOrig;
        int *origToPerm = levelset->permutation;
        int *nodeToLevel = levelset->nodeToLevel;
        int nlevels = levelset->getLevels();
        int nodes = levelset->getNodeNum();

        // DAG *dag_csr = DAG::inverseDAG(*dag_group, true);
        const int *DAGrowptr = dag_csr->DAG_ptr.data();
        const int *DAGcolidx = dag_csr->DAG_set.data();

        // 将node按照level内线程分配好的顺序进行排列，确保每个线程的task中node是紧密排列的，这样可以根据task找到对应的node
        std::vector<int> nodeByLevelThreadTask(nodes);
        std::vector<int> origNodeToThread(nodes, -1);

        // nodeFinishTime作为每个节点的计算完成时间，刚开始根据父节点的完成时间，计算出最早的可以开始计算的时间。根据最早的开始时间去匹配对应的线程
        std::vector<double> nodeFinishTime(nodes, 0);
        std::vector<double> threadTime(nthread, 0);

        std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        std::vector<int> levelThreadOne(nlevels, -1); // record the thread choosed for level with one execution thread

        // 分配level 1按照负载进行分配
        double *costBuffer = MALLOC(double, nodes + nthread);
        CHECK_POINTER(costBuffer);

        // process level one
        {
            int l = 0;
            int exec_thread = nthread_per_level[0];
            int levelBegin = level_ptr[0], levelEnd = level_ptr[1];
            // printf("level one node num: %d, exec_thread: %d\n", levelEnd - levelBegin, exec_thread);
            double load = 0.0;
            for (int i = levelBegin; i < levelEnd; i++)
            {
                costBuffer[i] = load;
                load += cost[permToOrig[i]];
            }
            double loadPerThread = (load + exec_thread - 1) / exec_thread;
            int preEnd = levelBegin;
            int r = levelBegin;
            load = 0;
            int t = 0;
            if (exec_thread == 1)
            {
                levelThreadOne[l] = 0;
            }

            // int avgThreadSkip = nthread / exec_thread;
            for (t = 0; t < exec_thread; t++)
            // for (t = 0; t < nthread; t+= avgThreadSkip)
            {
                int newr = std::lower_bound(&costBuffer[r], &costBuffer[levelEnd], (t + 1) * loadPerThread) - costBuffer;
                r = newr;
                int begin = preEnd;
                int end = std::min(r, levelEnd);
                preEnd = end;
                taskRows[t * nlevels + l] = make_pair(begin, end);
                for (int k = begin; k < end; k++)
                {
                    int nodeOrig = permToOrig[k];
                    nodeByLevelThreadTask[k] = nodeOrig;
                    origNodeToThread[nodeOrig] = t;           // mapping reverse perm node to thread
                    threadTime[t] += cost[nodeOrig];          // update threadTime by adding node cost time
                    nodeFinishTime[nodeOrig] = threadTime[t]; // update nodeFinishTime using threadTime
                }
                ++r;
            }
            for (t = exec_thread; t < nthread; t++)
            {
                taskRows[t * nlevels + l] = make_pair(levelEnd, levelEnd);
            }
        }
        FREE(costBuffer);

        std::vector<int> threadIdxVec(nthread);
        std::vector<std::list<int>> threadNode(nthread); // 每个线程分配的节点
        std::unordered_set<int> threadChoose;            // copy threadIdxVec to a set for qucik data search
        threadChoose.reserve(nthread);

        // process remain
        for (int lvl = 1; lvl < nlevels; lvl++)
        {
            int levelBegin = level_ptr[lvl], levelEnd = level_ptr[lvl + 1];
            int exec_thread = nthread_per_level[lvl];
            int nodeNumLevel = levelEnd - levelBegin;
            threadChoose.clear();
            for (auto &item : threadNode)
            {
                item.clear();
            }
            int threadInd = 0;
            std::generate(threadIdxVec.begin(), threadIdxVec.end(), [&threadInd]()
                          { return threadInd++; });

            // 找到最早finsih 的几个线程作为执行线程
            std::nth_element(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread, threadIdxVec.end(), [&threadTime](int a, int b)
                             { return threadTime[a] < threadTime[b]; });
            if (exec_thread == 1)
            {
                assert(threadIdxVec[0] >= 0);
                levelThreadOne[lvl] = threadIdxVec[0];
            }

            // 计算每个节点的最早开始时间，并按照时间进行排序，将最早可以执行的节点分配给最早结束任务的线程
            for (int i = levelBegin; i < levelEnd; i++)
            {
                int nodeOrig = permToOrig[i];
                int maxParentFinsihTime = 0;
                for (int parentPtr = DAGrowptr[nodeOrig]; parentPtr < DAGrowptr[nodeOrig + 1] - 1; parentPtr++)
                {
                    int parent = DAGcolidx[parentPtr];
                    maxParentFinsihTime = maxParentFinsihTime < nodeFinishTime[parent] ? nodeFinishTime[parent] : maxParentFinsihTime;
                }
                nodeFinishTime[nodeOrig] = maxParentFinsihTime;
            }
            // 对nodeFinishTime进行排序，按照最早的分配到最早开始运算的线程上
            std::vector<int> nodeIdx(nodeNumLevel); // 这里存储的是origIdx
            int s = levelBegin;
            std::generate(nodeIdx.begin(), nodeIdx.end(), [&s, permToOrig]()
                          { return permToOrig[s++]; });
            std::sort(nodeIdx.begin(), nodeIdx.end(), [&nodeFinishTime](int a, int b)
                      { return nodeFinishTime[a] < nodeFinishTime[b]; });
            // 根据选择出来的几个最早结束的线程进行任务的分配
            for (int i = 0; i < nodeNumLevel; i++)
            {
                // int nodeAllocating = ;
                int nodeOrig = nodeIdx[i];
                // 在threadIdx中找到一个最小的threadTime，将node分配给这个线程
                auto minItem = std::min_element(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread, [threadTime](int ta, int tb)
                                                { return threadTime[ta] < threadTime[tb]; });
                int t = *minItem;
                nodeFinishTime[nodeOrig] = threadTime[t] + cost[nodeOrig];
                threadTime[t] = nodeFinishTime[nodeOrig];
                threadNode[t].push_back(nodeOrig);
            }

#ifndef NDEBUG
            int total = 0;
            for (auto iter = threadIdxVec.begin(); iter != threadIdxVec.begin() + exec_thread; iter++)
            {
                // int t = threadIdxVec[i];
                total += threadNode[*iter].size();
            }
            assert(total == nodeNumLevel);
#endif

            nodeIdx.clear();
            nodeIdx.shrink_to_fit();

            // 将组织好的数据映射到taskRows, origNodeToThread 以及 nodeByLevelThreadTask
            // std::unordered_set<int> threadChoose
            threadChoose.insert(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread);
            int index = levelBegin;
            for (int et = 0; et < nthread; et++)
            {
                int taskIdx = et * nlevels + lvl;
                if (threadChoose.find(et) != threadChoose.end())
                {
                    int nodeNum = threadNode[et].size();
                    taskRows[taskIdx].first = index;
                    taskRows[taskIdx].second = index + nodeNum;

                    // 将节点放置到nodeByLevelThreadTask 中
                    for (auto &item : threadNode[et])
                    {
                        nodeByLevelThreadTask[index++] = item;
                        origNodeToThread[item] = et;
                    }
                }
                else
                {
                    taskRows[taskIdx].first = index;
                    taskRows[taskIdx].second = index;
                }
            }
        } // nlevel loop end

        // free memory
        threadIdxVec.clear();
        threadIdxVec.shrink_to_fit();
        threadNode.clear();
        threadNode.shrink_to_fit();
        nodeFinishTime.clear();
        nodeFinishTime.shrink_to_fit();
        threadTime.clear();
        threadTime.shrink_to_fit();
        threadChoose.clear();
        // delete dag_csr;

        /***************************** merge level task with continuous one execution thread *******************/
        int *level_ptr_merged = MALLOC(int, nlevels + 1);
        int *nodeToLevel_merged = MALLOC(int, nodes);
        int *merge_ptr = MALLOC(int, nlevels + 1);
        CHECK_POINTER(level_ptr_merged);
        CHECK_POINTER(nodeToLevel_merged);
        CHECK_POINTER(merge_ptr);
        std::fill_n(level_ptr_merged, nlevels + 1, 0);
        std::fill_n(merge_ptr, nlevels + 1, 0);
#ifndef NDEBUG
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            if (nthread_per_level[lvl] == 1)
            {
                assert(levelThreadOne[lvl] != -1);
                // printf(" %d ", levelThreadOne[lvl]);
            }
        }
        // printf("\n");
#endif
        int *nthread_per_level_tmp = MALLOC(int, nlevels);
        CHECK_POINTER(nthread_per_level_tmp);
        std::copy(nthread_per_level, nthread_per_level + nlevels, nthread_per_level_tmp);

        int level_m = 0;
        for (int lvl = 1; lvl < nlevels;)
        {
            if (nthread_per_level_tmp[lvl] == 1 && nthread_per_level_tmp[level_m] == 1)
            {
                // 合并taskRows 到level_m的线程
                int preThread = levelThreadOne[level_m];
                int postThread = levelThreadOne[lvl];
                assert(preThread != -1);
                assert(postThread != -1);
                // 找到taskRows
                // 这里taskRow原本的位置似乎不是level_m，应该是merge_ptr[level_m]
                std::pair<int, int> &preTask = taskRows[preThread * nlevels + merge_ptr[level_m]];
                // assert(level_ptr[lvl] - level_ptr_merged[level_m] == preTask.second - preTask.first);
                std::pair<int, int> &postTask = taskRows[postThread * nlevels + lvl];
                // assert(level_ptr[lvl + 1] - level_ptr[lvl] == postTask.second - postTask.first);
                assert(preTask.second == postTask.first);
                preTask.second = postTask.second;
                postTask.first = 0;
                postTask.second = 0;
                lvl++;
            }
            else
            {
                level_m++;
                nthread_per_level_tmp[level_m] = nthread_per_level_tmp[lvl];
                levelThreadOne[level_m] = levelThreadOne[lvl];
                level_ptr_merged[level_m] = level_ptr[lvl];
                merge_ptr[level_m] = lvl;
                lvl++;
            }
        }
        level_m++;
        level_ptr_merged[level_m] = level_ptr[nlevels];
        merge_ptr[level_m] = nlevels;
        int *level_ptr_merged_alias = MALLOC(int, level_m + 1);
        CHECK_POINTER(level_ptr_merged_alias);
        memcpy(level_ptr_merged_alias, level_ptr_merged, (level_m + 1) * sizeof(int));
        FREE(level_ptr_merged);
        FREE(nodeToLevel_merged);
        FREE(merge_ptr);
        FREE(level_ptr_merged_alias);
        FREE(nthread_per_level_tmp);
        // levelset->level_ptr = level_ptr_merged_alias;
        // 组织task类中的数据到threadBoundaries
        // 组织Task类中的threadBoundaries
        this->taskMapToBoundaries(taskRows, nlevels, nodes, nodeByLevelThreadTask.data(), nthread);
    }

    void Task::constructMappingByFourRule(DAG *&dag_group, DAG *& dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset)
    {

        constexpr double alpha = 1.0, beta = 0.8, gamma = 6.0;
        constexpr int affinityWindowSize = 5;

        int *level_ptr = levelset->level_ptr;
        int *permToOrig = levelset->permToOrig;
        int *origToPerm = levelset->permutation;
        int *nodeToLevel = levelset->nodeToLevel;
        int nlevels = levelset->getLevels();
        int nodes = levelset->getNodeNum();

        // DAG *dag_csr = DAG::inverseDAG(*dag_group, true);
        const int *DAGrowptr = dag_csr->DAG_ptr.data();
        const int *DAGcolidx = dag_csr->DAG_set.data();

        // 将node按照level内线程分配好的顺序进行排列，确保每个线程的task中node是紧密排列的，这样可以根据task找到对应的node
        std::vector<int> nodeByLevelThreadTask(nodes);
        std::vector<int> origNodeToThread(nodes, -1);

        std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        std::vector<int> levelThreadOne(nlevels, -1); // record the thread choosed for level with one execution thread

        // 分配level 1按照负载进行分配
        double *costBuffer = MALLOC(double, nodes + nthread);
        CHECK_POINTER(costBuffer);

        // process level one
        {
            int l = 0;
            int exec_thread = nthread_per_level[0];
            int levelBegin = level_ptr[0], levelEnd = level_ptr[1];
            // printf("level one node num: %d, exec_thread: %d\n", levelEnd - levelBegin, exec_thread);
            double load = 0.0;
            for (int i = levelBegin; i < levelEnd; i++)
            {
                costBuffer[i] = load;
                load += cost[permToOrig[i]];
            }
            double loadPerThread = (load + exec_thread - 1) / exec_thread;
            int preEnd = levelBegin;
            int r = levelBegin;
            load = 0;
            int t = 0;
            if (exec_thread == 1)
            {
                levelThreadOne[l] = 0;
            }

            // int avgThreadSkip = nthread / exec_thread;
            for (t = 0; t < exec_thread; t++)
            // for (t = 0; t < nthread; t+= avgThreadSkip)
            {
                int newr = std::lower_bound(&costBuffer[r], &costBuffer[levelEnd], (t + 1) * loadPerThread) - costBuffer;
                r = newr;
                int begin = preEnd;
                int end = std::min(r, levelEnd);
                // printf("thread %d begin: %d, end: %d\n",t, begin, end);
                preEnd = end;
                taskRows[t * nlevels + l] = make_pair(begin, end);
                for (int k = begin; k < end; k++)
                {
                    int nodeOrig = permToOrig[k];
                    nodeByLevelThreadTask[k] = nodeOrig;
                    origNodeToThread[nodeOrig] = t; // mapping reverse perm node to thread
                    // threadTime[t] += cost[nodeOrig];          // update threadTime by adding node cost time
                    // nodeFinishTime[nodeOrig] = threadTime[t]; // update nodeFinishTime using threadTime
                }
                ++r;
            }
            for (t = exec_thread; t < nthread; t++)
            {
                taskRows[t * nlevels + l] = make_pair(levelEnd, levelEnd);
            }
        }
        FREE(costBuffer);

        std::vector<int> threadIdxVec(nthread);          // 作为线程的候选idx，方便进行排序
        std::vector<std::list<int>> threadNode(nthread); // nthread lists to record nodes assigned to it
        std::unordered_set<int> threadChoose;
        std::vector<double> threadSelectScore(nthread);
        threadChoose.reserve(nthread);

        // the memmory for thread score computing
        std::vector<double> threadScore(nthread);
        std::vector<double> affinity(nthread);
        std::vector<double> threadLocalLoad(nthread);
        std::vector<double> threadPenalty(nthread);
        std::vector<double> nodeAffinityScore(nodes, 0.0);
        // std::vector<double>

        for (int lvl = 1; lvl < nlevels; lvl++)
        {
            int levelBegin = level_ptr[lvl], levelEnd = level_ptr[lvl + 1];
            int exec_thread = nthread_per_level[lvl];
            int nodeNumLevel = levelEnd - levelBegin;
            double avgLoad = 0;
            threadChoose.clear();
            for (auto &item : threadNode)
            {
                item.clear();
            }
            int threadInd = 0;
            std::generate(threadIdxVec.begin(), threadIdxVec.end(), [&threadInd]()
                          { return threadInd++; });
            // threadSelectScore
            std::fill(threadSelectScore.begin(), threadSelectScore.end(), 0.0);
            std::fill(affinity.begin(), affinity.end(), 0.0);
            std::fill(threadLocalLoad.begin(), threadLocalLoad.end(), 0.0);

            // initialize penalty
            for (int i = 0; i < nthread; i++)
            {
                constexpr double util = 0.0;
                const double remainUtil = 1.0 - util; // [0.0, 1.0]
                threadPenalty[i] = LOAD_REGULARIZATION_BETA_PARMETER * std::log(1.0 + remainUtil);
            }

            // select the executing thread
            for (int i = levelBegin; i < levelEnd; i++)
            {
                int nodeOrig = permToOrig[i];
                int nodeLevel = nodeToLevel[nodeOrig];
                avgLoad += cost[nodeOrig];
                for (int parentPtr = DAGrowptr[nodeOrig]; parentPtr < DAGrowptr[nodeOrig + 1] - 1; parentPtr++)
                {
                    int parent = DAGcolidx[parentPtr];
                    int runThread = origNodeToThread[parent];
                    int levelDiff = nodeLevel - nodeToLevel[parent];
                    assert(levelDiff > 0);
                    // int nodeLevel = nodeToLevel[parent];
                    // int levelDiff = lvl - levelDiff;
                    // constexpr double factor = 2.0; // 增大临近的level的方法效应

                    if (levelDiff <= LEVEL_WINDOW_SIZE)
                    {
                        double vote = AMPLIFY_FACTOR * std::log2(1.0 * (group_ptr[parent + 1] - group_ptr[parent] + 1)) / std::log2(levelDiff + 1);
                        threadSelectScore[runThread] += vote;
                        nodeAffinityScore[i] += vote;
                    }
                }
                // printf("node %d affinity: %lf\n", i, nodeAffinityScore[i]);
            }

            fflush(stdout);
            avgLoad /= exec_thread;

            // sort threadIdx by threadSelectScore
            std::nth_element(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread, threadIdxVec.end(),
                             [&threadSelectScore](int a, int b)
                             { return threadSelectScore[a] > threadSelectScore[b]; });

            if (exec_thread == 1)
            {
                assert(threadIdxVec[0] >= 0.0);
                levelThreadOne[lvl] = threadIdxVec[0];
            }

            threadChoose.insert(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread);

            // computing threadScore, and assign node to thread
            for (int i = levelBegin; i < levelEnd; i++)
            {
                std::fill(affinity.begin(), affinity.end(), 0.0);
                int nodeOrig = permToOrig[i];
                int nodeLevel = nodeToLevel[nodeOrig];
                int parentNum = DAGrowptr[nodeOrig + 1] - DAGrowptr[nodeOrig] - 1;
                for (int parentPtr = DAGrowptr[nodeOrig]; parentPtr < DAGrowptr[nodeOrig + 1] - 1; parentPtr++)
                {
                    int parent = DAGcolidx[parentPtr];
                    int runThread = origNodeToThread[parent];
                    int levelDiff = nodeLevel - nodeToLevel[parent];
                    assert(levelDiff);
                    if (levelDiff <= LEVEL_WINDOW_SIZE)
                    {
                        double vote = AMPLIFY_FACTOR * std::log2(1.0 * (group_ptr[parent + 1] - group_ptr[parent] + 1)) / std::log2(levelDiff + 1);
                        double scoreRatio = vote / nodeAffinityScore[i];
                        affinity[runThread] += scoreRatio;
                    }
                }
                double maxScore = std::numeric_limits<double>::lowest(); // get the minmum negative number of double type
                int targetThread = -1;
                // compute thread score for each thread and find the maximum score of thread
                for (int t = 0; t < nthread; t++)
                {
                    // double util = threadLocalLoad[t] / avgLoad;
                    // double loadPenalty = 0.0;
                    double loadPenalty = threadPenalty[t];

                    threadScore[t] = AWARD_ALPHA_PARMETER * affinity[t] + loadPenalty;
                    if (threadScore[t] > maxScore && threadChoose.find(t) != threadChoose.end())
                    {
                        maxScore = threadScore[t];
                        targetThread = t;
                    }
                }

                threadNode[targetThread].push_back(nodeOrig);
                threadLocalLoad[targetThread] += cost[nodeOrig];
                double util = threadLocalLoad[targetThread] / avgLoad;
                double loadPenalty = 0.0;
                if (util < 1.0)
                {
                    const double remainUtil = 1.0 - util; // [0.0, 1.0]
                    loadPenalty = LOAD_REGULARIZATION_BETA_PARMETER * std::log(1.0 + remainUtil);
                }
                else if (util > 1.05)
                {
                    loadPenalty = std::numeric_limits<double>::lowest();
                }
                else
                {
                    loadPenalty = -OVER_REGULARIZATION_GMMA_PARMETER * (std::exp(10 * (util - 1)) - 1);
                }
                threadPenalty[targetThread] = loadPenalty;
            }

#ifndef NDEBUG
            int total = 0;
            for (auto &item : threadChoose)
            {
                total += threadNode[item].size();
            }
            assert(total == nodeNumLevel);
#endif

            // maping threadNode to taskRows, origNodeToThread, and nodeByLevelThreadTask
            int index = levelBegin;
            for (int et = 0; et < nthread; et++)
            {
                int taskIdx = et * nlevels + lvl;
                if (threadChoose.find(et) != threadChoose.end())
                {
                    int nodeNum = threadNode[et].size();
                    taskRows[taskIdx].first = index;
                    taskRows[taskIdx].second = index + nodeNum;

                    // put node into nodeByLevelThreadTask
                    for (auto &item : threadNode[et])
                    {
                        nodeByLevelThreadTask[index++] = item;
                        origNodeToThread[item] = et;
                    }
                    // assert(nodeNum > 0);
                }
                else
                {
                    taskRows[taskIdx].first = index;
                    taskRows[taskIdx].second = index;
                }
            }
        } // for-loop: nlevels

#ifndef NDEBUG
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            if (nthread_per_level[lvl] == 1)
            {
                assert(levelThreadOne[lvl] != -1);
                // printf(" %d ", levelThreadOne[lvl]);
            }
        }
        // printf("\n");
#endif

#ifndef NDEBUG
        assert(taskRows.back().second == nodes);
        // check taskRows
        std::pair<int, int> preTask = taskRows[0]; // thread 0 for level 0;
        // std::pair<int, int> &postTask taskRows[0];
        // preTask = ;
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            for (int t = 0; t < nthread; t++)
            {
                if (lvl != 0 || t != 0) // not fisrt Task
                {
                    // std::pair<int, int>
                    auto postTask = taskRows[t * nlevels + lvl];
                    assert(preTask.second == postTask.first);
                    preTask = postTask;
                }
            }
        }

#endif

        threadIdxVec.clear();
        threadIdxVec.shrink_to_fit();
        threadNode.clear();
        threadNode.shrink_to_fit();
        threadChoose.clear();

        threadSelectScore.clear();
        threadSelectScore.shrink_to_fit();
        threadScore.clear();
        threadScore.shrink_to_fit();
        affinity.clear();
        affinity.shrink_to_fit();
        threadLocalLoad.clear();
        threadLocalLoad.shrink_to_fit();
        // delete dag_csr;


        /***************************** merge level task with continuous one execution thread *******************/
        // int *level_ptr_merged = MALLOC(int, nlevels + 1);
        // int *nodeToLevel_merged = MALLOC(int, nodes);
        int *merge_ptr = MALLOC(int, nlevels + 1);
        // CHECK_POINTER(level_ptr_merged);
        // CHECK_POINTER(nodeToLevel_merged);
        CHECK_POINTER(merge_ptr);
        // std::fill_n(level_ptr_merged, nlevels + 1, 0);
        std::fill_n(merge_ptr, nlevels + 1, 0);
#ifndef NDEBUG
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            if (nthread_per_level[lvl] == 1)
            {
                assert(levelThreadOne[lvl] != -1);
                // printf(" %d ", levelThreadOne[lvl]);
            }
        }
        // printf("\n");
#endif
        int *nthread_per_level_tmp = MALLOC(int, nlevels);
        CHECK_POINTER(nthread_per_level_tmp);
        std::copy(nthread_per_level, nthread_per_level + nlevels, nthread_per_level_tmp);

        int level_m = 0;
        for (int lvl = 1; lvl < nlevels;)
        {
            if (nthread_per_level_tmp[lvl] == 1 && nthread_per_level_tmp[level_m] == 1)
            {

                // 合并taskRows 到level_m的线程
                int preThread = levelThreadOne[level_m];
                int postThread = levelThreadOne[lvl];
                assert(preThread != -1);
                assert(postThread != -1);
                // 找到taskRows
                // 这里taskRow原本的位置似乎不是level_m，应该是merge_ptr[level_m]
                std::pair<int, int> &preTask = taskRows[preThread * nlevels + merge_ptr[level_m]];
                // assert(level_ptr[lvl] - level_ptr_merged[level_m] == preTask.second - preTask.first);

                std::pair<int, int> &postTask = taskRows[postThread * nlevels + lvl];
                // assert(level_ptr[lvl + 1] - level_ptr[lvl] == postTask.second - postTask.first);
                // assert(preTask.second == postTask.first);
                // printf("preThread:%d\n", preThread);
                // printf("postThread:%d\n", postThread);
                // printf("level_m:%d\n", level_m);
                // printf("lvl:%d\n", lvl);
                // printf("preTask.first: %d\n", preTask.first);
                // printf("preTask.sencond: %d\n", preTask.second);
                // printf("postTask.first: %d\n", postTask.first);
                // printf("postTask.sencond: %d\n", postTask.second);
                preTask.second = postTask.second;
                postTask.first = 0;
                postTask.second = 0;
                lvl++;
            }
            else
            {
                level_m++;
                nthread_per_level_tmp[level_m] = nthread_per_level_tmp[lvl];
                levelThreadOne[level_m] = levelThreadOne[lvl];
                // level_ptr_merged[level_m] = level_ptr[lvl];
                merge_ptr[level_m] = lvl;
                lvl++;
            }
        }

        level_m++;
        // level_ptr_merged[level_m] = level_ptr[nlevels];
        merge_ptr[level_m] = nlevels;

        FREE(merge_ptr);
        // FREE(level_ptr_merged_alias);
        FREE(nthread_per_level_tmp);

        this->taskMapToBoundaries(taskRows, nlevels, nodes, nodeByLevelThreadTask.data(), nthread);
    }

    void Task::
    constructMappingByFourRuleParallel(DAG *&dag_group, DAG *&dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset)
    {

        constexpr double alpha = 1.0, beta = 0.8, gamma = 6.0;
        constexpr int affinityWindowSize = 5;

        int *level_ptr = levelset->level_ptr;
        int *permToOrig = levelset->permToOrig;
        int *origToPerm = levelset->permutation;
        int *nodeToLevel = levelset->nodeToLevel;
        int nlevels = levelset->getLevels();
        int nodes = levelset->getNodeNum();

        // DAG *dag_csr = DAG::inverseDAG(*dag_group, true);
        const int *DAGrowptr = dag_csr->DAG_ptr.data();
        const int *DAGcolidx = dag_csr->DAG_set.data();

        // 将node按照level内线程分配好的顺序进行排列，确保每个线程的task中node是紧密排列的，这样可以根据task找到对应的node
        std::vector<int> nodeByLevelThreadTask(nodes);
        std::vector<int> origNodeToThread(nodes, -1);

        std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        std::vector<int> levelThreadOne(nlevels, -1); // record the thread choosed for level with one execution thread

        // 分配level 1按照负载进行分配
        double *costBuffer = MALLOC(double, nodes + nthread);
        CHECK_POINTER(costBuffer);

        // process level one
        {
            int l = 0;
            int exec_thread = nthread_per_level[0];
            int levelBegin = level_ptr[0], levelEnd = level_ptr[1];
            // printf("level one node num: %d, exec_thread: %d\n", levelEnd - levelBegin, exec_thread);
            double load = 0.0;
            for (int i = levelBegin; i < levelEnd; i++)
            {
                costBuffer[i] = load;
                load += cost[permToOrig[i]];
            }
            double loadPerThread = (load + exec_thread - 1) / exec_thread;
            int preEnd = levelBegin;
            int r = levelBegin;
            load = 0;
            int t = 0;
            if (exec_thread == 1)
            {
                levelThreadOne[l] = 0;
            }

            // int avgThreadSkip = nthread / exec_thread;
            for (t = 0; t < exec_thread; t++)
            // for (t = 0; t < nthread; t+= avgThreadSkip)
            {
                int newr = std::lower_bound(&costBuffer[r], &costBuffer[levelEnd], (t + 1) * loadPerThread) - costBuffer;
                r = newr;
                int begin = preEnd;
                int end = std::min(r, levelEnd);
                // printf("thread %d begin: %d, end: %d\n",t, begin, end);
                preEnd = end;
                taskRows[t * nlevels + l] = make_pair(begin, end);
                for (int k = begin; k < end; k++)
                {
                    int nodeOrig = permToOrig[k];
                    nodeByLevelThreadTask[k] = nodeOrig;
                    origNodeToThread[nodeOrig] = t; // mapping reverse perm node to thread
                    // threadTime[t] += cost[nodeOrig];          // update threadTime by adding node cost time
                    // nodeFinishTime[nodeOrig] = threadTime[t]; // update nodeFinishTime using threadTime
                }
                ++r;
            }
            for (t = exec_thread; t < nthread; t++)
            {
                taskRows[t * nlevels + l] = make_pair(levelEnd, levelEnd);
            }
        }
        FREE(costBuffer);

        std::vector<int> threadIdxVec(nthread);          // 作为线程的候选idx，方便进行排序
        std::vector<std::list<int>> threadNode(nthread); // nthread lists to record nodes assigned to it
        std::unordered_set<int> threadChoose;
        std::vector<double> threadSelectScore(nthread);
        threadChoose.reserve(nthread);

        // the memmory for thread score computing
        std::vector<double> threadScore(nthread);
        std::vector<double> affinity(nthread);
        std::vector<double> threadLocalLoad(nthread);
        std::vector<double> threadPenalty(nthread);
        std::vector<double> nodeAffinityScore(nodes, 0.0);
        // std::vector<double>

        std::vector<double> threadSelectScoreVec(nthread * nthread);
        std::vector<double> loadTotal(nthread);

        int maxLevel = 0;
        int maxDiffAvg = 0;
#pragma omp parallel for reduction(max : maxLevel) reduction(max : maxDiffAvg)
        for (int i = 0; i < nlevels; i++)
        {
            int diff = level_ptr[i + 1] - level_ptr[i];
            int diffAvg = (diff + nthread - 1) / nthread;
            maxLevel = std::max(maxLevel, diff);
            maxDiffAvg = std::max(maxDiffAvg, diffAvg);
        }
        // 计算每个线程的最大使用内存量, 每个线程都使用这个区间内的数据，不允许操作数据的越界
        int maxAvgThreadLen = maxDiffAvg * nthread;
        std::vector<double> affinityLevelNode(maxLevel * nthread);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int tidNum = omp_get_num_threads();
            for (int lvl = 1; lvl < nlevels; lvl++)
            {
#pragma omp barrier
                int levelBegin = level_ptr[lvl], levelEnd = level_ptr[lvl + 1];
                int exec_thread = nthread_per_level[lvl];
                int nodeNumLevel = levelEnd - levelBegin;

                threadIdxVec[tid] = tid;
                threadSelectScore[tid] = 0;
                affinity[tid] = 0;
                threadLocalLoad[tid] = 0;
                threadNode[tid].clear();
                double util = 0;
                const double remainUtil = 1.0 - util; // [0.0, 1.0]
                threadPenalty[tid] = LOAD_REGULARIZATION_BETA_PARMETER * std::log(1.0 + remainUtil);
                // threadPenalty[tid] = 0;

                std::fill(threadSelectScoreVec.begin() + tid * nthread, threadSelectScoreVec.begin() + (tid + 1) * nthread, 0);
                // std::fill(affinityLevelNode.begin() + tid * maxLevel, affinityLevelNode.begin() + (tid + 1) * maxLevel, 0);
                loadTotal[tid] = 0;
                // #pragma omp barrier

                int tidAvg = (nodeNumLevel + nthread - 1) / tidNum;
                int tidBeg = std::min(levelBegin + tidAvg * tid, levelEnd);
                int tidEnd = std::min(levelBegin + tidAvg * (tid + 1), levelEnd);
                std::fill(affinityLevelNode.begin() + (tidBeg - levelBegin) * nthread, affinityLevelNode.begin() + (tidEnd - levelBegin) * nthread, 0);
                for (int i = tidBeg; i < tidEnd; i++)
                {
                    int nodeOrig = permToOrig[i];
                    int nodeLevel = nodeToLevel[nodeOrig];
                    // avgLoad += cost[nodeOrig];
                    loadTotal[tid] += cost[nodeOrig];
                    for (int parentPtr = DAGrowptr[nodeOrig]; parentPtr < DAGrowptr[nodeOrig + 1] - 1; parentPtr++)
                    {
                        int parent = DAGcolidx[parentPtr];
                        int runThread = origNodeToThread[parent];
                        int levelDiff = nodeLevel - nodeToLevel[parent];
                        assert(levelDiff > 0);
                        if (levelDiff <= LEVEL_WINDOW_SIZE)
                        {
                            double vote = AMPLIFY_FACTOR * std::log2(1.0 * (group_ptr[parent + 1] - group_ptr[parent] + 1)) / std::log2(levelDiff + 1);
                            // threadSelectScore[runThread] += vote;
                            threadSelectScoreVec[tid * nthread + runThread] += vote;
                            nodeAffinityScore[i] += vote;
                            // assert(maxAvgThreadLen * tid + (i - tidBeg) * nthread + runThread < maxAvgThreadLen * (tid + 1));
                            // assert(maxAvgThreadLen * tid + (i - tidBeg) * nthread + runThread < maxAvgThreadLen * (tid + 1));
                            affinityLevelNode[(i - levelBegin) * nthread + runThread] += vote;
                        }
                    }
                    // printf("node %d affinity: %lf\n", i, nodeAffinityScore[i]);
                }

#pragma omp barrier
#pragma omp master
                {

                    double avgLoad = 0;
                    threadChoose.clear();

                    for (int i = 0; i < nthread; i++)
                    {
                        avgLoad += loadTotal[i];
                        threadNode[i].clear();
                        threadSelectScore[i] = threadSelectScoreVec[i];
                        for (int j = 1; j < nthread; j++)
                        {
                            threadSelectScore[i] += threadSelectScoreVec[i + j * nthread];
                        }
                    }

                    avgLoad /= exec_thread;

                    // sort threadIdx by threadSelectScore
                    std::nth_element(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread, threadIdxVec.end(),
                                     [&threadSelectScore](int a, int b)
                                     { return threadSelectScore[a] > threadSelectScore[b]; });

                    if (exec_thread == 1)
                    {
                        assert(threadIdxVec[0] >= 0.0);
                        levelThreadOne[lvl] = threadIdxVec[0];
                    }

                    threadChoose.insert(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread);

                    // computing threadScore, and assign node to thread
                    for (int i = levelBegin; i < levelEnd; i++)
                    {
                        // std::fill(affinity.begin(), affinity.end(), 0.0);
                        int nodeOrig = permToOrig[i];

                        double maxScore = std::numeric_limits<double>::lowest(); // get the minmum negative number of double type
                        int targetThread = -1;
                        double affinityTotal = nodeAffinityScore[i];
                        // compute thread score for each thread and find the maximum score of thread
                        for (int t = 0; t < nthread; t++)
                        {
                            if (threadChoose.find(t) == threadChoose.end())
                                continue;
                            // affinity[t] /= nodeAffinityScore[i];
                            double loadPenalty = threadPenalty[t];
                            double affinityS = affinityLevelNode[(i - levelBegin) * nthread + t] / affinityTotal;
                            // threadScore[t] = AWARD_ALPHA_PARMETER * affinity[t] + loadPenalty;
                            threadScore[t] = AWARD_ALPHA_PARMETER * affinityS + loadPenalty;
                            if (threadScore[t] > maxScore)
                            {
                                maxScore = threadScore[t];
                                targetThread = t;
                            }
                        }

                        threadNode[targetThread].push_back(nodeOrig);
                        threadLocalLoad[targetThread] += cost[nodeOrig];
                        double util = threadLocalLoad[targetThread] / avgLoad;
                        double loadPenalty = 0.0;
                        if (util < 1.0)
                        {
                            const double remainUtil = 1.0 - util; // [0.0, 1.0]
                            loadPenalty = LOAD_REGULARIZATION_BETA_PARMETER * std::log(1.0 + remainUtil);
                        }
                        else if (util > 1.05)
                        {
                            loadPenalty = std::numeric_limits<double>::lowest();
                        }
                        else
                        {
                            loadPenalty = -OVER_REGULARIZATION_GMMA_PARMETER * (std::exp(10 * (util - 1)) - 1);
                        }
                        threadPenalty[targetThread] = loadPenalty;
                    }

#ifndef NDEBUG
                    int total = 0;
                    for (auto &item : threadChoose)
                    {
                        total += threadNode[item].size();
                    }
                    assert(total == nodeNumLevel);
#endif

                    // maping threadNode to taskRows, origNodeToThread, and nodeByLevelThreadTask
                    int index = levelBegin;
                    for (int et = 0; et < nthread; et++)
                    {
                        int taskIdx = et * nlevels + lvl;
                        if (threadChoose.find(et) != threadChoose.end())
                        {
                            int nodeNum = threadNode[et].size();
                            taskRows[taskIdx].first = index;
                            taskRows[taskIdx].second = index + nodeNum;

                            // put node into nodeByLevelThreadTask
                            for (auto &item : threadNode[et])
                            {
                                nodeByLevelThreadTask[index++] = item;
                                origNodeToThread[item] = et;
                            }
                            // assert(nodeNum > 0);
                        }
                        else
                        {
                            taskRows[taskIdx].first = index;
                            taskRows[taskIdx].second = index;
                        }
                    }
                }

            } // for-loop: nlevels
        }
#ifndef NDEBUG
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            if (nthread_per_level[lvl] == 1)
            {
                assert(levelThreadOne[lvl] != -1);
                // printf(" %d ", levelThreadOne[lvl]);
            }
        }
        // printf("\n");
#endif

#ifndef NDEBUG
        assert(taskRows.back().second == nodes);
        // check taskRows
        std::pair<int, int> preTask = taskRows[0]; // thread 0 for level 0;
        // std::pair<int, int> &postTask taskRows[0];
        // preTask = ;
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            for (int t = 0; t < nthread; t++)
            {
                if (lvl != 0 || t != 0) // not fisrt Task
                {
                    // std::pair<int, int>
                    auto postTask = taskRows[t * nlevels + lvl];
                    assert(preTask.second == postTask.first);
                    preTask = postTask;
                }
            }
        }

#endif

        threadIdxVec.clear();
        threadIdxVec.shrink_to_fit();
        threadNode.clear();
        threadNode.shrink_to_fit();
        threadChoose.clear();

        threadSelectScore.clear();
        threadSelectScore.shrink_to_fit();
        threadScore.clear();
        threadScore.shrink_to_fit();
        affinity.clear();
        affinity.shrink_to_fit();
        threadLocalLoad.clear();
        threadLocalLoad.shrink_to_fit();

        /***************************** merge level task with continuous one execution thread *******************/

        int *merge_ptr = MALLOC(int, nlevels + 1);
        CHECK_POINTER(merge_ptr);
        std::fill_n(merge_ptr, nlevels + 1, 0);
#ifndef NDEBUG
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            if (nthread_per_level[lvl] == 1)
            {
                assert(levelThreadOne[lvl] != -1);
            }
        }
        // printf("\n");
#endif
        int *nthread_per_level_tmp = MALLOC(int, nlevels);
        CHECK_POINTER(nthread_per_level_tmp);
        std::copy(nthread_per_level, nthread_per_level + nlevels, nthread_per_level_tmp);

        int level_m = 0;
        for (int lvl = 1; lvl < nlevels;)
        {
            if (nthread_per_level_tmp[lvl] == 1 && nthread_per_level_tmp[level_m] == 1)
            {

                // 合并taskRows 到level_m的线程
                int preThread = levelThreadOne[level_m];
                int postThread = levelThreadOne[lvl];
                assert(preThread != -1);
                assert(postThread != -1);
                // 找到taskRows
                // 这里taskRow原本的位置似乎不是level_m，应该是merge_ptr[level_m]
                std::pair<int, int> &preTask = taskRows[preThread * nlevels + merge_ptr[level_m]];
                std::pair<int, int> &postTask = taskRows[postThread * nlevels + lvl];

                preTask.second = postTask.second;
                postTask.first = 0;
                postTask.second = 0;
                lvl++;
            }
            else
            {
                level_m++;
                nthread_per_level_tmp[level_m] = nthread_per_level_tmp[lvl];
                levelThreadOne[level_m] = levelThreadOne[lvl];
                // level_ptr_merged[level_m] = level_ptr[lvl];
                merge_ptr[level_m] = lvl;
                lvl++;
            }
        }

        level_m++;
        merge_ptr[level_m] = nlevels;
        FREE(merge_ptr);
        FREE(nthread_per_level_tmp);

        this->taskMapToBoundaries(taskRows, nlevels, nodes, nodeByLevelThreadTask.data(), nthread);
    }

    void Task::constructMappingByFourRuleParallelBack(DAG *&dag_group, DAG *&dag_csr, const LevelSet *levelset, const double *cost, const int nthread, int *const nthread_per_level, const int *group_ptr, const int *groupset)
    {

        constexpr double alpha = 1.0, beta = 0.8, gamma = 6.0;
        constexpr int affinityWindowSize = 5;

        int *level_ptr = levelset->level_ptr;
        int *permToOrig = levelset->permToOrig;
        int *origToPerm = levelset->permutation;
        int *nodeToLevel = levelset->nodeToLevel;
        int nlevels = levelset->getLevels();
        int nodes = levelset->getNodeNum();

        // DAG *dag_csr = DAG::inverseDAG(*dag_group, true);
        const int *DAGrowptr = dag_csr->DAG_ptr.data();
        const int *DAGcolidx = dag_csr->DAG_set.data();

        // 将node按照level内线程分配好的顺序进行排列，确保每个线程的task中node是紧密排列的，这样可以根据task找到对应的node
        std::vector<int> nodeByLevelThreadTask(nodes);
        std::vector<int> origNodeToThread(nodes, -1);

        std::vector<std::pair<int, int>> taskRows(nlevels * nthread);
        std::vector<int> levelThreadOne(nlevels, -1); // record the thread choosed for level with one execution thread

        // 分配level 1按照负载进行分配
        double *costBuffer = MALLOC(double, nodes + nthread);
        CHECK_POINTER(costBuffer);

        // process level one
        {
            int l = 0;
            int exec_thread = nthread_per_level[0];
            int levelBegin = level_ptr[0], levelEnd = level_ptr[1];
            // printf("level one node num: %d, exec_thread: %d\n", levelEnd - levelBegin, exec_thread);
            double load = 0.0;
            for (int i = levelBegin; i < levelEnd; i++)
            {
                costBuffer[i] = load;
                load += cost[permToOrig[i]];
            }
            double loadPerThread = (load + exec_thread - 1) / exec_thread;
            int preEnd = levelBegin;
            int r = levelBegin;
            load = 0;
            int t = 0;
            if (exec_thread == 1)
            {
                levelThreadOne[l] = 0;
            }

            // int avgThreadSkip = nthread / exec_thread;
            for (t = 0; t < exec_thread; t++)
            // for (t = 0; t < nthread; t+= avgThreadSkip)
            {
                int newr = std::lower_bound(&costBuffer[r], &costBuffer[levelEnd], (t + 1) * loadPerThread) - costBuffer;
                r = newr;
                int begin = preEnd;
                int end = std::min(r, levelEnd);
                // printf("thread %d begin: %d, end: %d\n",t, begin, end);
                preEnd = end;
                taskRows[t * nlevels + l] = make_pair(begin, end);
                for (int k = begin; k < end; k++)
                {
                    int nodeOrig = permToOrig[k];
                    nodeByLevelThreadTask[k] = nodeOrig;
                    origNodeToThread[nodeOrig] = t; // mapping reverse perm node to thread
                    // threadTime[t] += cost[nodeOrig];          // update threadTime by adding node cost time
                    // nodeFinishTime[nodeOrig] = threadTime[t]; // update nodeFinishTime using threadTime
                }
                ++r;
            }
            for (t = exec_thread; t < nthread; t++)
            {
                taskRows[t * nlevels + l] = make_pair(levelEnd, levelEnd);
            }
        }
        FREE(costBuffer);

        std::vector<int> threadIdxVec(nthread);          // 作为线程的候选idx，方便进行排序
        std::vector<std::list<int>> threadNode(nthread); // nthread lists to record nodes assigned to it
        std::unordered_set<int> threadChoose;
        std::vector<double> threadSelectScore(nthread);
        threadChoose.reserve(nthread);

        // the memmory for thread score computing
        std::vector<double> threadScore(nthread);
        std::vector<double> affinity(nthread);
        std::vector<double> threadLocalLoad(nthread);
        // std::vector<double> threadPenalty(nthread);
        std::vector<double> nodeAffinityScore(nodes, 0.0);

        std::vector<double> threadSelectScoreVec(nthread * nthread);
        std::vector<double> loadTotal(nthread);

        int maxLevel = 0;
        int maxDiffAvg = 0;
#pragma omp parallel for reduction(max : maxLevel) reduction(max : maxDiffAvg)
        for (int i = 0; i < nlevels; i++)
        {
            int diff = level_ptr[i + 1] - level_ptr[i];
            int diffAvg = (diff + nthread - 1) / nthread;
            maxLevel = std::max(maxLevel, diff);
            maxDiffAvg = std::max(maxDiffAvg, diffAvg);
        }
        // 计算每个线程的最大使用内存量, 每个线程都使用这个区间内的数据，不允许操作数据的越界
        int maxAvgThreadLen = maxDiffAvg * nthread;
        std::vector<double> affinityLevelNode(maxLevel * nthread);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int tidNum = omp_get_num_threads();

            for (int lvl = 1; lvl < nlevels; lvl++)
            {
#pragma omp barrier
                int levelBegin = level_ptr[lvl], levelEnd = level_ptr[lvl + 1];
                int exec_thread = nthread_per_level[lvl];
                int nodeNumLevel = levelEnd - levelBegin;

                threadIdxVec[tid] = tid;
                threadSelectScore[tid] = 0;
                // affinity[tid] = 0;
                threadLocalLoad[tid] = 0;

                std::fill(threadSelectScoreVec.begin() + tid * nthread, threadSelectScoreVec.begin() + (tid + 1) * nthread, 0);
                std::fill(affinityLevelNode.begin() + tid * maxLevel, affinityLevelNode.begin() + (tid + 1) * maxLevel, 0);
                loadTotal[tid] = 0;

                int tidAvg = (nodeNumLevel + nthread - 1) / tidNum;
                int tidBeg = std::min(levelBegin + tidAvg * tid, levelEnd);
                int tidEnd = std::min(levelBegin + tidAvg * (tid + 1), levelEnd);

                for (int i = tidBeg; i < tidEnd; i++)
                {
                    int nodeOrig = permToOrig[i];
                    int nodeLevel = nodeToLevel[nodeOrig];
                    // avgLoad += cost[nodeOrig];
                    loadTotal[tid] += cost[nodeOrig];
                    for (int parentPtr = DAGrowptr[nodeOrig]; parentPtr < DAGrowptr[nodeOrig + 1] - 1; parentPtr++)
                    {
                        int parent = DAGcolidx[parentPtr];
                        int runThread = origNodeToThread[parent];
                        int levelDiff = nodeLevel - nodeToLevel[parent];
                        assert(levelDiff > 0);
                        if (levelDiff <= LEVEL_WINDOW_SIZE)
                        {
                            double vote = AMPLIFY_FACTOR * std::log2(1.0 * (group_ptr[parent + 1] - group_ptr[parent] + 1)) / std::log2(levelDiff + 1);
                            // threadSelectScore[runThread] += vote;
                            threadSelectScoreVec[tid * nthread + runThread] += vote;
                            nodeAffinityScore[i] += vote;
                            // assert(maxAvgThreadLen * tid + (i - tidBeg) * nthread + runThread < maxAvgThreadLen * (tid + 1));
                            // assert(maxAvgThreadLen * tid + (i - tidBeg) * nthread + runThread < maxAvgThreadLen * (tid + 1));
                            affinityLevelNode[(i - levelBegin) * nthread + runThread] += vote;
                        }
                    }
                    // printf("node %d affinity: %lf\n", i, nodeAffinityScore[i]);
                }

#pragma omp barrier
#pragma omp master
                {

                    double avgLoad = 0;
                    threadChoose.clear();

                    for (int i = 0; i < nthread; i++)
                    {
                        avgLoad += loadTotal[i];
                        threadNode[i].clear();
                        threadSelectScore[i] = threadSelectScoreVec[i];
                        for (int j = 1; j < nthread; j++)
                        {
                            threadSelectScore[i] += threadSelectScoreVec[i + j * nthread];
                        }
                    }
                    fflush(stdout);
                    avgLoad /= exec_thread;

                    // sort threadIdx by threadSelectScore
                    std::nth_element(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread, threadIdxVec.end(),
                                     [&threadSelectScore](int a, int b)
                                     { return threadSelectScore[a] > threadSelectScore[b]; });

                    if (exec_thread == 1)
                    {
                        assert(threadIdxVec[0] >= 0.0);
                        levelThreadOne[lvl] = threadIdxVec[0];
                    }

                    threadChoose.insert(threadIdxVec.begin(), threadIdxVec.begin() + exec_thread);

                    // computing threadScore, and assign node to thread
                    for (int i = levelBegin; i < levelEnd; i++)
                    {
                        // std::fill(affinity.begin(), affinity.end(), 0.0);
                        int nodeOrig = permToOrig[i];
                        // int nodeLevel = nodeToLevel[nodeOrig];
                        // int parentNum = DAGrowptr[nodeOrig + 1] - DAGrowptr[nodeOrig] - 1;
                        double maxScore = std::numeric_limits<double>::lowest(); // get the minmum negative number of double type
                        int targetThread = -1;
                        // compute thread score for each thread and find the maximum score of thread
                        for (int t = 0; t < nthread; t++)
                        {
                            double util = threadLocalLoad[t] / avgLoad;

                            double loadPenalty = 0.0;
                            if (util < 1.0)
                            {
                                const double remainUtil = 1.0 - util; // [0.0, 1.0]
                                loadPenalty = LOAD_REGULARIZATION_BETA_PARMETER * std::log(1.0 + remainUtil);
                            }
                            else if (util > 1.05)
                            {
                                loadPenalty = std::numeric_limits<double>::lowest();
                            }
                            else
                            {
                                loadPenalty = -OVER_REGULARIZATION_GMMA_PARMETER * (std::exp(10 * (util - 1)) - 1);
                            }

                            // threadScore[t] = AWARD_ALPHA_PARMETER * affinity[t] + loadPenalty;
                            double affinitS = affinityLevelNode[(i - levelBegin) * nthread + t] / nodeAffinityScore[i];

                            threadScore[t] = AWARD_ALPHA_PARMETER * affinitS + loadPenalty;
                            if (threadScore[t] > maxScore && threadChoose.find(t) != threadChoose.end())
                            {
                                maxScore = threadScore[t];
                                targetThread = t;
                            }
                        }

                        threadNode[targetThread].push_back(nodeOrig);
                        threadLocalLoad[targetThread] += cost[nodeOrig];
                    }

#ifndef NDEBUG
                    int total = 0;
                    for (auto &item : threadChoose)
                    {
                        total += threadNode[item].size();
                    }
                    assert(total == nodeNumLevel);
#endif

                    // maping threadNode to taskRows, origNodeToThread, and nodeByLevelThreadTask
                    int index = levelBegin;
                    for (int et = 0; et < nthread; et++)
                    {
                        int taskIdx = et * nlevels + lvl;
                        if (threadChoose.find(et) != threadChoose.end())
                        {
                            int nodeNum = threadNode[et].size();
                            taskRows[taskIdx].first = index;
                            taskRows[taskIdx].second = index + nodeNum;

                            // put node into nodeByLevelThreadTask
                            for (auto &item : threadNode[et])
                            {
                                nodeByLevelThreadTask[index++] = item;
                                origNodeToThread[item] = et;
                            }
                            // assert(nodeNum > 0);
                        }
                        else
                        {
                            taskRows[taskIdx].first = index;
                            taskRows[taskIdx].second = index;
                        }
                    }
                } // loop omp master
            } // loop nlevels
        }

        // for (int lvl = 1; lvl < nlevels; lvl++)
        // {

        // } // for-loop: nlevels

#ifndef NDEBUG
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            if (nthread_per_level[lvl] == 1)
            {
                assert(levelThreadOne[lvl] != -1);
                // printf(" %d ", levelThreadOne[lvl]);
            }
        }
        // printf("\n");
#endif

#ifndef NDEBUG
        assert(taskRows.back().second == nodes);
        // check taskRows
        std::pair<int, int> preTask = taskRows[0]; // thread 0 for level 0;
        // std::pair<int, int> &postTask taskRows[0];
        // preTask = ;
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            for (int t = 0; t < nthread; t++)
            {
                if (lvl != 0 || t != 0) // not fisrt Task
                {
                    // std::pair<int, int>
                    auto postTask = taskRows[t * nlevels + lvl];
                    assert(preTask.second == postTask.first);
                    preTask = postTask;
                }
            }
        }

#endif

        threadIdxVec.clear();
        threadIdxVec.shrink_to_fit();
        threadNode.clear();
        threadNode.shrink_to_fit();
        threadChoose.clear();

        threadSelectScore.clear();
        threadSelectScore.shrink_to_fit();
        threadScore.clear();
        threadScore.shrink_to_fit();
        affinity.clear();
        affinity.shrink_to_fit();
        threadLocalLoad.clear();
        threadLocalLoad.shrink_to_fit();
        // delete dag_csr;


        /***************************** merge level task with continuous one execution thread *******************/
        int *level_ptr_merged = MALLOC(int, nlevels + 1);
        int *nodeToLevel_merged = MALLOC(int, nodes);
        int *merge_ptr = MALLOC(int, nlevels + 1);
        CHECK_POINTER(level_ptr_merged);
        CHECK_POINTER(nodeToLevel_merged);
        CHECK_POINTER(merge_ptr);
        std::fill_n(level_ptr_merged, nlevels + 1, 0);
        std::fill_n(merge_ptr, nlevels + 1, 0);
#ifndef NDEBUG
        for (int lvl = 0; lvl < nlevels; lvl++)
        {
            if (nthread_per_level[lvl] == 1)
            {
                assert(levelThreadOne[lvl] != -1);
                // printf(" %d ", levelThreadOne[lvl]);
            }
        }
        // printf("\n");
#endif
        int *nthread_per_level_tmp = MALLOC(int, nlevels);
        CHECK_POINTER(nthread_per_level_tmp);
        std::copy(nthread_per_level, nthread_per_level + nlevels, nthread_per_level_tmp);

        int level_m = 0;
        for (int lvl = 1; lvl < nlevels;)
        {
            if (nthread_per_level_tmp[lvl] == 1 && nthread_per_level_tmp[level_m] == 1)
            {

                // 合并taskRows 到level_m的线程
                int preThread = levelThreadOne[level_m];
                int postThread = levelThreadOne[lvl];
                assert(preThread != -1);
                assert(postThread != -1);
                // 找到taskRows
                // 这里taskRow原本的位置似乎不是level_m，应该是merge_ptr[level_m]
                std::pair<int, int> &preTask = taskRows[preThread * nlevels + merge_ptr[level_m]];

                std::pair<int, int> &postTask = taskRows[postThread * nlevels + lvl];
                preTask.second = postTask.second;
                postTask.first = 0;
                postTask.second = 0;
                lvl++;
            }
            else
            {
                level_m++;
                nthread_per_level_tmp[level_m] = nthread_per_level_tmp[lvl];
                levelThreadOne[level_m] = levelThreadOne[lvl];
                level_ptr_merged[level_m] = level_ptr[lvl];
                merge_ptr[level_m] = lvl;
                lvl++;
            }
        }

        level_m++;
        level_ptr_merged[level_m] = level_ptr[nlevels];
        merge_ptr[level_m] = nlevels;

        int *level_ptr_merged_alias = MALLOC(int, level_m + 1);
        CHECK_POINTER(level_ptr_merged_alias);
        memcpy(level_ptr_merged_alias, level_ptr_merged, (level_m + 1) * sizeof(int));
        FREE(level_ptr_merged);
        FREE(nodeToLevel_merged);
        FREE(merge_ptr);
        FREE(level_ptr_merged_alias);
        FREE(nthread_per_level_tmp);

        this->taskMapToBoundaries(taskRows, nlevels, nodes, nodeByLevelThreadTask.data(), nthread);
    }

    void deleteIntraThreadEdge()
    {
    }

    void twoHopTransitiveRedcution()
    {
    }

    TaskSchedule::TaskSchedule(int ntasks, int cparents)
    {
        this->ntasks = ntasks;
        this->cparents = cparents;

        parents = MALLOC(int *, ntasks);
        nparents = MALLOC(int, ntasks);
        parentsBuf = MALLOC(int, cparents);
        taskFinsished = MALLOC(int, ntasks);
    }

    TaskSchedule::~TaskSchedule()
    {
        {
            // for (int task = 0; task < ntasks; task++)
            // {
            //     FREE(parents[task]);
            // }
            FREE(parents);
            FREE(nparents);
            FREE(parentsBuf);
            int *taskFinishedTmp = (int *)taskFinsished;
            FREE(taskFinishedTmp);

            FREE(parentsBackward);
            FREE(nparentsBackward);
            FREE(parentsBufBackward)
        }
    }

    void TaskSchedule::constructTaskSchedule(const DAG *taskDAG)
    {
        const int *taskPtr = taskDAG->DAG_ptr.data();
        const int *taskColIdx = taskDAG->DAG_set.data();
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthread = omp_get_max_threads();
            int taskPerThread = (ntasks + nthread - 1) / nthread;
            int taskBeg = std::min(taskPerThread * tid, ntasks);
            int taskEnd = std::min(taskPerThread * (tid + 1), ntasks);
            std::fill_n(taskFinsished + taskBeg, taskEnd - taskBeg, 0);
            for (int task = taskBeg; task < taskEnd; task++)
            {

                nparents[task] = taskPtr[task + 1] - taskPtr[task];
                parents[task] = parentsBuf + taskPtr[task];
                //  int p = 0;
                for (int j = 0; j < nparents[task]; j++)
                {
                    // int parent = ;
                    parents[task][j] = taskColIdx[taskPtr[task] + j];
                }
            }
        }
    }

    void TaskSchedule::constructInverseSchedule(const DAG *taskDAG)
    {
        parentsBackward = MALLOC(int *, ntasks);
        nparentsBackward = MALLOC(int, ntasks);
        parentsBufBackward = MALLOC(int, cparents);
        DAG *inverseTaskDAG = DAG::inverseDAG(*taskDAG, true); // CSR

        const int *backwardTaskPtr = inverseTaskDAG->DAG_ptr.data();
        const int *backwardTaskColIdx = inverseTaskDAG->DAG_set.data();
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthread = omp_get_max_threads();
            int taskPerThread = (ntasks + nthread - 1) / nthread;
            int taskBeg = std::min(taskPerThread * tid, ntasks);
            int taskEnd = std::min(taskPerThread * (tid + 1), ntasks);
            std::fill_n(taskFinsished + taskBeg, taskEnd - taskBeg, 0);
            for (int task = taskBeg; task < taskEnd; task++)
            {

                nparentsBackward[task] = backwardTaskPtr[task + 1] - backwardTaskPtr[task];
                parentsBackward[task] = parentsBufBackward + backwardTaskPtr[task];
                //  int p = 0;
                for (int j = 0; j < nparentsBackward[task]; j++)
                {
                    // int parent = ;
                    parentsBackward[task][j] = backwardTaskColIdx[backwardTaskPtr[task] + j];
                }
            }
        }
        delete inverseTaskDAG;
    }

    int TaskSchedule::getP2PNum()
    {
        int total = 0;
        #pragma omp parallel for reduction(+ : total)
        for (int i = 0; i < ntasks; i++)
        {
            total += nparents[i];
        }
        return total;
    }

} // namespace Merge
