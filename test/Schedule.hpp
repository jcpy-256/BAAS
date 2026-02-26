/**
 * @note: this file implements the schedule of all loop-carried algorithm (containing forward schedule and backward schedule),
 * including LTRSV, UTRSV, ILU0, IC0
 */
#include <iostream>
#include "SPM.hpp"
#include "Merge.hpp"
#include "Profile.hpp"
#include "MarcoUtils.hpp"
#include "MathUtils.hpp"

#ifndef DELETE_SAFE
#define DELETE_SAFE(x)   \
    {                    \
        if (x)           \
        {                \
            delete x;    \
            x = nullptr; \
        }                \
    }
#endif

using SPM::CSR, SPM::CSC, SPM::DAG, SPM::LevelSet, SPM::DAG_MAT;

using Merge::Task, Merge::TaskSchedule, Merge::level_merge_group, Merge::level_nthreads_group, Merge::level_nthreads_group_TRSV;
using namespace Coarsen;

namespace Schedule
{

    enum ALG_SCHEDULE
    {
        P2P,
        P2P_MAPPING,
        P2P_TIME,
        P2P_RULE,
    };

    class Scheduler
    {
        // private:
    protected:
        // general data
        const CSR *csrA; // the original matrix
        bool isSymmetric;
        bool isSerial = false;
        LevelSet *levelset_merge;
        int *nthread_per_level;
        int ngroups;
        CSC *cscA; // the dependency DAG
        DAG *dag_group;
        std::vector<int> group_set, group_ptr;
        std::vector<double> cost;
        Kernel kerenl;
        ALG_SCHEDULE alg;
        int num_thread;

        // trsv perm
        bool isTRSVPerm;

        CSR *csrA_perm;
        double *x_perm;

        // scheduling information
        Task *task;
        TaskSchedule *taskSchedule;
        DAG *taskDAG;

        CSR *csrALower;
        CSC *cscALower;

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

    public:
        std::vector<int> permToOrigByThread, origToPermByThread;

        Scheduler(const CSR *csrA, bool isSymmetric, bool isTRSVPerm, ALG_SCHEDULE alg, int num_thread)
        {
            this->csrA = csrA;
            this->isSymmetric = isSymmetric;
            this->isTRSVPerm = isTRSVPerm;
            this->alg = alg;
            this->num_thread = num_thread;
            dag_group = new DAG();
            this->taskDAG = new DAG();

            if (isTRSVPerm)
            {
                x_perm = MALLOC(double, csrA->m);
                CHECK_POINTER(x_perm);
            }

            // construct
            int lowerNnz = 0;
            const int *rowptr = csrA->rowptr;
            const int *diagptr = csrA->diagptr;
            const int *colidx = csrA->colidx;
            int n = csrA->n;

            for (int i = 0; i < n; i++)
            {
                lowerNnz += diagptr[i] - rowptr[i] + 1;
            }

            this->csrALower = new CSR(n, n, lowerNnz);
            csrALower->rowptr[0];
            for (int i = 0; i < n; i++) // contain diagonal element
            {
                csrALower->rowptr[i + 1] = csrALower->rowptr[i] + diagptr[i] - rowptr[i] + 1;
                std::copy(colidx + rowptr[i], colidx + diagptr[i] + 1, csrALower->colidx + csrALower->rowptr[i]);

            }
            this->cscALower = new CSC();
            CSR::transpositionToCSC(csrALower, cscALower);
        }
        ~Scheduler()
        {
            FREE(nthread_per_level);
            DELETE_SAFE(levelset_merge);
            DELETE_SAFE(dag_group);
            DELETE_SAFE(task);
            DELETE_SAFE(taskSchedule);
            DELETE_SAFE(taskDAG);
            DELETE_SAFE(csrA_perm);
            FREE(x_perm);
        }

        void preprocessing();
        void permPreprocessing();
        void runIC0(const CSR *mat, double *lu);
        void runIC0Perm(const CSR *matPerm, double *lu);
        void runForwardTRSV(const CSR *L, double *x, const double *b, bool isPerm);
        void runForwardTRSVNoPerm(const CSR *L, double *x, const double *b);
        void runForwardTRSVPerm(const CSR *LPerm, double *x, const double *b);
        void runBackwardTRSV(const CSR *L, double *x, const double *b, bool isPerm);
        void runBackwardTRSVNoPerm(const CSR *L, double *x, const double *b);
        void runBackwardTRSVPerm(const CSR *L, double *x, const double *b);
    };

    // scheduling simultaneously for different kernel
    void Scheduler::preprocessing()
    {
        // constrcut CSR lower and CSC lower matrix
        int lowerNnz = 0;
        const int *rowptr = csrA->rowptr;
        const int *diagptr = csrA->diagptr;
        const int *colidx = csrA->colidx;
        int n = csrA->n;

        // two-hop transitive redcution
        // 1. create an empty DAG
        lowerNnz = csrALower->getNnz();
        DAG *dag = new DAG(n, lowerNnz);
        dag->DAG_ptr.clear();
        dag->DAG_set.clear();
        // 2. execute reduction
        DAG::partialSparsification_CSC(n, lowerNnz, cscALower->colptr, cscALower->rowidx, dag->DAG_ptr, dag->DAG_set, false);

        // ************************************ reverse coarsen *****************************************
        Coarsen::reverseTreeCoarseningBFS_all(n, dag->DAG_ptr, dag->DAG_set, ngroups, group_ptr, group_set);

        // build new grouped DAG
        Coarsen::buildGroupDAGParallel(n, ngroups, group_ptr.data(), group_set.data(),
                                       dag->DAG_ptr.data(), dag->DAG_set.data(), dag_group->DAG_ptr, dag_group->DAG_set);
        dag_group->setNodeNum(ngroups);
        delete dag;

        /************************************ forward coarsen ******************************************* */
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
        delete dag_group;
        dag_group = dag_group_f;
        dag_group_f = nullptr;

        // cost computing
        Coarsen::costComputation(ngroups, cscALower->colptr, cscALower->rowidx, csrALower->rowptr, csrALower->colidx,
                                 Coarsen::Kernel::SpTRSV_LL, group_ptr.data(), group_set.data(), true, cost);

        /******************************** Level Merge ****************************** */
        if (this->alg == P2P)
        {
            LevelSet *levelset = new LevelSet();
            dag_group->findLevelsPostOrder(levelset);
            level_merge_group(levelset, csrA, cost.data(), nthread_per_level, levelset_merge, this->num_thread, group_ptr.data(), group_set.data(), ngroups);
            delete levelset;
            levelset = nullptr;
        }
        else if (this->alg == P2P_MAPPING || this->alg == P2P_TIME || this->alg == P2P_RULE)
        {
            LevelSet *levelset = new LevelSet();
            dag_group->findLevelsPostOrder(levelset);
            levelset_merge = levelset;
            levelset = nullptr;
            // level_nthreads_group(levelset_merge, csrA, group_ptr.data(), group_set.data(), this->num_thread, this->nthread_per_level);
            level_nthreads_group_TRSV(levelset_merge, csrA, group_ptr.data(), group_set.data(), this->num_thread, this->nthread_per_level);
        }
        else
        {
            throw std::runtime_error("This scheduler algorithm is not supported currently. ");
            exit(1);
        }

        // ordering groupset to ensure correct execution flow
        orderingGroupSet(ngroups, group_ptr, group_set);
        DAG *dag_csr = DAG::inverseDAG(*dag_group);

        // ****************************** constrcut p2p DAG and Scheduler ****************************
        task = new Merge::Task(ngroups, false);

        switch (alg)
        {
        case P2P:
            task->constructTask(levelset_merge, cost.data(), num_thread, nthread_per_level);
            break;
        case P2P_MAPPING:
            task->constructMappingMergeSerial(dag_group, dag_csr, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            break;
        case P2P_TIME:
            task->constructMappingByFinishTime(dag_group, dag_csr, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            break;
        case P2P_RULE:
            task->constructMappingByFourRuleParallel(dag_group, dag_csr, levelset_merge, cost.data(), this->num_thread, this->nthread_per_level, group_ptr.data(), group_set.data());
            break;
        default:
            throw std::runtime_error("This scheduler algorithm is not supported currently. ");
            exit(1);
            break;
        }
        task->constructTaskDAGParallel(dag_group, dag_csr, taskDAG, this->num_thread, false);
        taskSchedule = new TaskSchedule(taskDAG->getNodeNum(), taskDAG->DAG_ptr.back());
        taskSchedule->constructTaskSchedule(taskDAG);
        delete dag_csr;

        if (isSymmetric) // uppper triangular matrix
        {
            taskSchedule->constructInverseSchedule(taskDAG);
        }

        if (taskSchedule->getP2PNum() == 0)
        {
            this->isSerial = true;
        }
    }

    void Scheduler::permPreprocessing()
    {
        std::vector<int> group_set_p, group_ptr_p; // 记录经过perm后的ptr和set集合，随后将类中的group_ptr和set进行std::move
        groupPerm(ngroups, task->origToThreadContPerm, group_set, group_ptr, group_set_p, group_ptr_p);
        std::vector<int> origToPerm(group_set_p.size(), 0);
        getInversePerm(origToPerm.data(), group_set_p.data(), group_set_p.size());

        assert(isPerm(group_set.data(), group_set.size()));
        assert(isPerm(group_set_p.data(), group_set_p.size()));
        assert(isPerm(origToPerm.data(), origToPerm.size()));

        this->permToOrigByThread = group_set_p; // not move but copy
        this->origToPermByThread = std::move(origToPerm);

        group_ptr = std::move(group_ptr_p);
        group_set = std::move(group_set_p); // inverse permuation : perm -> orig
    }

    void Scheduler::runIC0(const CSR *mat, double *lu)
    {

        spic0_csr_uL_p2p_group_merge_no_perm(mat, lu, levelset_merge,
                                             nthread_per_level, group_ptr.data(),
                                             group_set.data(), task, taskSchedule);
    }

    void Scheduler::runIC0Perm(const CSR *matPerm, double *lu)
    {

        spic0_csr_uL_p2p_group_merge_perm(matPerm, lu, levelset_merge,
                                          nthread_per_level, group_ptr.data(),
                                          group_set.data(), task, taskSchedule);
    }

    void Scheduler::runForwardTRSVNoPerm(const CSR *L, double *x, const double *b)
    {

        sptrsv_p2p_csr_group_merge_no_perm_alloc(L, x, b, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, taskSchedule);
    }

    void Scheduler::runForwardTRSVPerm(const CSR *LPerm, double *x, const double *b)
    {
        sptrsv_p2p_csr_group_merge_perm_alloc(LPerm, x, b, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, taskSchedule);
    }

    void Scheduler::runBackwardTRSVNoPerm(const CSR *L, double *x, const double *b)
    {

        sptrsv_backward_p2p_csr_group_merge_no_perm_alloc(L, x, b, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, taskSchedule);
    }

    void Scheduler::runBackwardTRSVPerm(const CSR *LPerm, double *x, const double *b)
    {

        sptrsv_backward_p2p_csr_group_merge_perm_alloc(LPerm, x, b, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, taskSchedule);
    }

} // namespace Schedule
