#pragma once
#include "mkl.h"
#include "SPM.hpp"

#include "Merge.hpp"
#include "TimeMeasure.hpp"
#include "ILUDemo.hpp"
#include "MarcoUtils.hpp"
#include "MathUtils.hpp"

using namespace Merge;
using namespace SPM;

namespace ILURT
{
    class ILU_Serial : public ILUDemo
    {
    private:
        /* data */
    protected:
        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();

            // copy data from csrA to lu
            std::copy(this->csrA->values, this->csrA->values + nnz, lu);

            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
        }
        TimeMeasure fused_code() override
        {
            // recopy data to lu
            std::copy(this->csrA->values, this->csrA->values + nnz, lu);
            TimeMeasure t1;
            t1.start_timer();
            ilu0csr_uplooking_ref(csrA, lu);
            t1.measure_elasped_time();
            return t1;
        }

    public:
        ILU_Serial(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName)
            : ILUDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_lu = correct_lu;
        }
        ~ILU_Serial() = default;
    };

    class ILU_MKL : public ILUDemo
    {
    protected:
        int *ipar;
        double *dpar;
        CSR *csr_mkl;
        bool opt;

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            dpar[30] = 1.0e-16, dpar[31] = 1.0e-10; // diagonal threshold and replace value if ipar[30] != 0
            csr_mkl->make1BasedIndexing();          // mkl only support 1-based;
            // copy data from csrA to lu
            std::copy(this->csrA->values, this->csrA->values + nnz, lu);

            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
        }

        TimeMeasure fused_code() override
        {
            // recopy data to lu
            std::copy(this->csrA->values, this->csrA->values + nnz, lu);
            TimeMeasure t1;
            t1.start_timer();
            MKL_INT ierr = 0;
            dcsrilu0(&n, csr_mkl->values, csr_mkl->rowptr, csr_mkl->colidx, lu, ipar, dpar, &ierr);
            t1.measure_elasped_time();

            if (ierr != 0)
            {
                std::cerr << "MKL ILU failure, the error code is " << ierr << endl;
                std::exit(1);
            }

            return t1;
        }

    public:
        ILU_MKL(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName, bool opt = false)
            : ILUDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_lu = correct_lu;
            this->opt = opt;
            ipar = MALLOC(int, 128);
            dpar = MALLOC(double, 128);
            std::fill_n(ipar, 128, 0);
            std::fill_n(dpar, 128, 0.0);
            csr_mkl = new CSR(*csrA);
        }

        ~ILU_MKL()
        {
            delete csr_mkl;
            FREE(ipar);
            FREE(dpar);
        }
    };

    class ILU_LevelSet : public ILUDemo
    {
    private:
    protected:
        DAG *ilu0DAG;
        LevelSet *levelset;

        void buildset() override
        {
            TimeMeasure t1;
            t1.start_timer();
            int lowerNnz = 0;
            const int *rowptr = csrA->rowptr;
            const int *diagptr = csrA->diagptr;
            const int *colidx = csrA->colidx;

            for (int i = 0; i < n; i++)
            {
                lowerNnz += diagptr[i] - rowptr[i] + 1;
            }
            // initial DAG in CSR format
            // ilu0DAG->format = DAG_MAT::DAG_CSR;
            // ilu0DAG->edges = lowerNnz;
            // ilu0DAG->n = n;
            // ilu0DAG->DAG_ptr.resize(n + 1);
            // ilu0DAG->DAG_set.resize(lowerNnz);
            // ilu0DAG->DAG_ptr[0] = 0;

            ilu0DAG = new DAG(n, lowerNnz, DAG_MAT::DAG_CSR);
            ilu0DAG->DAG_ptr[0] = 0;

            for (int i = 0; i < n; i++)
            {
                ilu0DAG->DAG_ptr[i + 1] = ilu0DAG->DAG_ptr[i] + diagptr[i] - rowptr[i] + 1;
                std::copy(colidx + rowptr[i], colidx + diagptr[i] + 1, ilu0DAG->DAG_set.begin() + ilu0DAG->DAG_ptr[i]);
            }
            assert(ilu0DAG->DAG_ptr.back() == lowerNnz);
            assert(ilu0DAG->DAG_set.back() == n - 1);

            ilu0DAG->findLevelsPostOrder(levelset);
            int nlevels = levelset->getLevels();
            // printf("nlevels:%d\n", nlevels);

#ifndef NDEBUG
            LevelSet *levelsetVf = new LevelSet();
            ilu0DAG->findLevels(levelsetVf);
            bool check = levelset->equal(*levelsetVf);
            fflush(stdout);
            assert(check);
            delete levelsetVf;
#endif
            // copy data from csrA to lu
            std::copy(this->csrA->values, this->csrA->values + nnz, lu);

            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
        }

        TimeMeasure fused_code() override
        {
            // recopy data to lu
            std::copy(this->csrA->values, this->csrA->values + nnz, lu);
            TimeMeasure t1;
            t1.start_timer();
            ilu0csr_uplooking_levelset_kernel(this->csrA, levelset, lu);
            t1.measure_elasped_time();
            return t1;
        }

    public:
        ILU_LevelSet(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName, int num_thread)
            : ILUDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_lu = correct_lu;
            this->num_thread = num_thread;

            levelset = new LevelSet();
            ilu0DAG = new DAG();
        }
        ~ILU_LevelSet()
        {
            if (ilu0DAG)
                delete ilu0DAG;
            delete levelset;
        }

        int getWavefront() const { return levelset->getLevels(); }
    };


    /********************************* Reverse and Forward Tree with P2P and node Maping in Four rule  ****************************** */
    class ILU_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule : public ILUDemo
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
            // construct lower triangular matrix
            int lowerNnz = 0;
            const int *rowptr = csrA->rowptr;
            const int *diagptr = csrA->diagptr;
            const int *colidx = csrA->colidx;

            for (int i = 0; i < n; i++)
            {
                lowerNnz += diagptr[i] - rowptr[i] + 1;
            }

            CSR *csrALower = new CSR(n, n, lowerNnz);
            csrALower->rowptr[0];
            for (int i = 0; i < n; i++) // contain diagonal element
            {
                csrALower->rowptr[i + 1] = csrALower->rowptr[i] + diagptr[i] - rowptr[i] + 1;
                std::copy(colidx + rowptr[i], colidx + diagptr[i] + 1, csrALower->colidx + csrALower->rowptr[i]);
                // std::copy(val + rowptr[i], colidx + diagptr[i] + 1, csrALower->colidx + csrALower->rowptr[i]);
            }
            CSC *cscALower = new CSC();
            CSR::transpositionToCSC(csrALower, cscALower);

            DAG *dag = new DAG(n, lowerNnz);
            dag->DAG_ptr.clear();
            dag->DAG_set.clear();
            // two-hop transitive reduction
            DAG::partialSparsification_CSC(n, lowerNnz, cscALower->colptr, cscALower->rowidx, dag->DAG_ptr, dag->DAG_set, false);

            // ******************************** reverse coarsen **************************
            Coarsen::reverseTreeCoarseningBFS_all(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);

            // printf("reverse ngroups: %d\n", ngroups);

            // build new grouped DAG
            Coarsen::buildGroupDAGParallel(n, ngroups, group_ptr.data(), group_set.data(),
                                           dag->DAG_ptr.data(), dag->DAG_set.data(), dag_group->DAG_ptr, dag_group->DAG_set);
            dag_group->setNodeNum(ngroups);
            delete dag;

            /********************************** forward coarsen ************************ */
            std::vector<int> group_ptr_f, group_set_f;
            DAG *dag_group_f = new DAG();
            int ngroups_f = 0;
            Coarsen::forwardTreeCoarseningBFS_all(ngroups, dag_group->DAG_ptr, dag_group->DAG_set, ngroups_f, group_ptr_f, group_set_f, false);
            // mapping group_f into group_ptr
            Coarsen::groupRemapping(group_ptr, group_set, group_ptr_f, group_set_f, ngroups_f);

            // build new DAG based on grouped_DAG, with group_f
            Coarsen::buildGroupDAGParallel(ngroups, ngroups_f, group_ptr_f.data(), group_set_f.data(),
                                           dag_group->DAG_ptr.data(), dag_group->DAG_set.data(), dag_group_f->DAG_ptr, dag_group_f->DAG_set);

            ngroups = ngroups_f;
            // printf("after two coarsening ngroups: %d\n", ngroups);
            dag_group_f->setNodeNum(ngroups_f);
            dag_group_f->updateEdges();
            LevelSet *levelset = new LevelSet();
            dag_group_f->findLevelsPostOrder(levelset);
            levelset_merge = levelset;
            delete dag_group;
            dag_group = dag_group_f;

            Coarsen::costComputation(ngroups, cscALower->colptr, cscALower->rowidx, csrALower->rowptr, csrALower->colidx,
                                     Coarsen::Kernel::SpTRSV_LL, group_ptr.data(), group_set.data(), true, cost);

            // level_merge_group(levelset, csrA, cost.data(), nthread_per_level, levelset_merge, this->num_thread, group_ptr.data(), group_set.data(), ngroups);
            // level_merge_group(levelset, csrA, cost.data(), nthread_per_level, levelset_merge, this->num_thread, group_ptr.data(), group_set.data(), ngroups);
            // level_merge(levelset, csrA, nthread_per_level, levelset_merge, this->num_thread);
            level_nthreads_group_ILU0(levelset_merge, csrA, group_ptr.data(), group_set.data(), this->num_thread, this->nthread_per_level);
            // delete levelset;

            orderingGroupSet(ngroups, group_ptr, group_set);

            // 转p2p
            task = new Merge::Task(ngroups, false);
            DAG *dag_inv = DAG::inverseDAG(*dag_group);
            // task alloc
            // task->constructTask(levelset_merge, cost.data(), this->num_thread, nthread_per_level);
            // task->constructMappingSerial(dag_group, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            // task->constructMappingByFinishTime(dag_group, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            task->constructMappingByFourRuleParallel(dag_group, dag_inv, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());

            // constrcut task DAG
            task->constructTaskDAGParallel(dag_group, dag_inv, taskDAG, this->num_thread, false);
            delete dag_inv;

            // // printf("taskNUm: %d\n", taskDAG->getNodeNum());
            // // printf("cparents: %d\n", taskDAG->DAG_ptr.back());
            // // fflush(stdout);

            // construct task schedule
            schedule = new TaskSchedule(taskDAG->getNodeNum(), taskDAG->DAG_ptr.back());
            schedule->constructTaskSchedule(taskDAG);

            // if (this->hasPerm)
            // {
            //     // 对稀疏矩阵进行重排
            //     std::vector<int> group_set_p, group_ptr_p; // 记录经过perm后的ptr和set集合，随后将类中的group_ptr和set进行std::move
            //     groupPerm(ngroups, task->origToThreadContPerm, group_set, group_ptr, group_set_p, group_ptr_p);
            //     std::vector<int> origToPerm(group_set_p.size(), 0);
            //     getInversePerm(origToPerm.data(), group_set_p.data(), group_set_p.size());

            //     assert(isPerm(group_set.data(), group_set.size()));
            //     assert(isPerm(group_set_p.data(), group_set_p.size()));
            //     assert(isPerm(origToPerm.data(), origToPerm.size()));
            //     // 对CSR进行深度拷贝
            //     // this->csrA_perm = new CSR(*csrA);
            //     // csrA_perm.
            //     csrA_perm = csrA->permute(origToPerm.data(), group_set_p.data()); // not sort to keep the diag element
            //     this->permToOrig = group_set_p;
            //     this->origToPerm = std::move(origToPerm);

            //     group_ptr = std::move(group_ptr_p);
            //     group_set = std::move(group_set_p); // inverse permuation : perm -> orig
            // }
            t1.measure_elasped_time();
            this->analysis_time = t1.getTime();
        }

        TimeMeasure fused_code() override
        {
            // recopy data to lu
            std::copy(this->csrA->values, this->csrA->values + nnz, lu);
            TimeMeasure t1;
            if (this->getWavefront() == 0)
            {
                t1.start_timer();
                ilu0csr_uplooking_ref(csrA, lu);
                t1.measure_elasped_time();
            }
            else
            {
                t1.start_timer();
                ilu0_p2p_csr_group_merge_no_perm_alloc(csrA, lu, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                t1.measure_elasped_time();
            }

            return t1;
        }

    public:
        ILU_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName, int num_thread)
            : ILUDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_lu = correct_lu;
            this->num_thread = num_thread;

            levelset_merge = new LevelSet();
            this->dag_group = new DAG();
            this->taskDAG = new DAG();
        }

        ~ILU_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule()
        {
            delete levelset_merge;
            FREE(nthread_per_level);
            delete dag_group;
            if (task)
                delete task;
            delete taskDAG;
            if (schedule)
            {
                delete schedule;
            }
        }

        int getWavefront() const { return schedule->getP2PNum(); }
    };


} // namespace ILURT
