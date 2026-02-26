#include <iostream>
#include <string>

#include "SPM.hpp"
#include "TimeMeasure.hpp"
#include "MarcoUtils.hpp"
#include "Merge.hpp"
#include "LevelMerge.hpp"
#include "MathUtils.hpp"

using Merge::level_merge_group, Merge::level_nthreads_group_IC0, Merge::level_nthreads_group;
using Merge::TaskSchedule, Merge::Task;
using SPM::CSR, SPM::LevelSet, SPM::DAG;
// using namespace Merge;

namespace ICRT
{
    class ICDemo
    {
    private:
        /* data */
    protected:
        TimeMeasure timeMeasure;
        // bool hasPerm = false;

        int n;
        int lowerNnz;
        int nnz;
        std::vector<int> permToOrig, origToPerm; // corresponding to inverse permutation and permutation
        double *lu;
        double *correct_lu;
        std::string algName;
        int num_test;
        int num_thread;
        double elasped_time = 0;
        double analysis_time = 0;

        CSR *csrA;
        // CSR *csrLower;

        virtual void setup();
        virtual void buildset() {};
        virtual TimeMeasure fused_code() = 0;
        virtual void testing();

    public:
        ICDemo(/* args */);
        ICDemo(int n, int nnz, std::string algName);
        ICDemo(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName);

        ~ICDemo();

        TimeMeasure evaluate();
        double *getSolution() { return lu; }
        void set_num_test(int num_test) { this->num_test = num_test; }
        void set_num_thread(int num_thread) { this->num_thread = num_thread; }
        double getAnalysisTime() const { return analysis_time; }
        double getElaspedTime() const { return elasped_time; }
        std::string getAlgName() { return algName; }
        void getLU(CSR *&L, CSR *U);
    };

    ICDemo::ICDemo() : lu(nullptr), correct_lu(nullptr), csrA(nullptr)
    {
        this->num_test = 10;
        this->num_thread = 1;
        this->n = 0;
        this->lowerNnz = 0;
        this->nnz = 0;
    };

    ICDemo::~ICDemo()
    {
        FREE(lu);
    }

    ICDemo::ICDemo(int n, int nnz, std::string algName)
    {
        this->n = n;
        this->nnz = nnz;
        this->algName = algName;
        lu = MALLOC(double, nnz);
        CHECK_POINTER(lu);
    }
    ICDemo::ICDemo(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName) : ICDemo(n, nnz, algName)
    {
        this->csrA = csrA;
        this->correct_lu = correct_lu;
    }

    void ICDemo::setup()
    {
    }

    void ICDemo::testing()
    {
        if (correct_lu != nullptr)
        {
            std::cout << "=============== result check ===================\n";
            bool flag = true;
            for (int i = 0; i < n; i++)
            {
                if (flag == false)
                    break;
                for (int k = csrA->rowptr[i]; k <= csrA->diagptr[i]; k++)
                {
                    if (fabs(correct_lu[k] - lu[k]) > 1.0e-10)
                    {
                        flag = false;
                        break;
                    }
                }
            }
            if (flag == false)
            {
                cout << algName + " code IC0 != reference factorization.\n";
            }
            else
            {
                cout << algName + " code IC0 check pass.\n";
            }
        }
        fflush(stdout);
    }

    TimeMeasure ICDemo::evaluate()
    {
        // printf("ILu)
        std::cout << "################ ILU0 runing in " << this->algName + " method #################" << endl;
        TimeMeasure avg;
        std::vector<TimeMeasure> time_array;
        TimeMeasure analsis;
        analsis.start_timer();
        buildset();
        analsis.measure_elasped_time();
        analysis_time = analsis.getTime();

        setup();

        // if (hasPerm)
        // {
        //     // permutation matrix
        // }

        for (int i = 0; i < num_test; i++)
        {
            TimeMeasure t1;
            t1 = fused_code();
            time_array.push_back(t1);
        }

        // if (hasPerm)
        // {
        //     // inverse permutation matrix
        // }

        testing();
        for (int i = 0; i < num_test; i++)
        {
            avg.elasped_time += time_array[i].elasped_time;
        }
        avg.elasped_time /= num_test;
        this->elasped_time = avg.elasped_time;

        std::cout << "################ ILU0 ending in " << this->algName + " method #################" << endl;
        return avg;
    }

    void ICDemo::getLU(CSR *&L, CSR *U)
    {
        // A->make0BasedIndexing();
        int nnzL, nnzU;
        // SPD matrix
        nnzL = nnzU = (nnz + n) / 2;

        L = new CSR(n, n, nnzL);
        U = new CSR(n, n, nnzU);
#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i < n; i++)
            {
                L->rowptr[i + 1] = csrA->diagptr[i] - csrA->rowptr[i] + 1; // contain diagonal elements
                U->rowptr[i + 1] = csrA->rowptr[i + 1] - csrA->diagptr[i];
            }

            // prefix sum
#pragma omp single
            for (int i = 0; i < n; i++)
            {
                L->rowptr[i + 1] += L->rowptr[i];
                U->rowptr[i + 1] += U->rowptr[i];
            }

#pragma omp for
            for (int i = 0; i < n; i++)
            {
                std::copy(csrA->colidx + csrA->rowptr[i], csrA->colidx + csrA->diagptr[i] + 1, L->colidx + L->rowptr[i]);
                std::copy(csrA->colidx + csrA->diagptr[i], csrA->colidx + csrA->rowptr[i + 1], U->colidx + U->rowptr[i]);
            }
        } // for: omp for
        // 填充LU
        int *U_rowptr_alias = MALLOC(int, n + 1);
        std::copy(U->rowptr, U->rowptr + n + 1, U_rowptr_alias);
        for (int i = 0; i < n; i++)
        {
            assert(lu[csrA->diagptr[i]] > 0);
            for (int j = csrA->rowptr[i]; j <= csrA->diagptr[i]; j++)
            {
                double val = lu[j];

                int L_offset = L->rowptr[i] + j - csrA->rowptr[i];
                int U_offset = U_rowptr_alias[csrA->colidx[j]];
                L->values[L_offset] = val;
                U->values[U_offset] = val;
                U_rowptr_alias[csrA->colidx[j]]++;
            }
        }

#ifndef NDEBUG
        CSR *UT = U->transpose();
        bool isEqual = UT->equals(*L);
        assert(isEqual);
        printf("%s\n", isEqual ? "the transposed U matrix is equal to L matrix!" : "the transposed U matrix is not equal to L matrix!");
        delete UT;
#endif
        FREE(U_rowptr_alias);
    }

    class IC_Serial : public ICDemo
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
            spic0_csr_uL_serial(csrA, lu);
            t1.measure_elasped_time();
            return t1;
        }

    public:
        IC_Serial(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName)
            : ICDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_lu = correct_lu;
        }
        ~IC_Serial() = default;
    };

    class IC_LevelSet : public ICDemo
    {
    private:
    protected:
        DAG *ic0DAG;
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

            ic0DAG = new DAG(n, lowerNnz, DAG_MAT::DAG_CSR);
            ic0DAG->DAG_ptr[0] = 0;

            for (int i = 0; i < n; i++)
            {
                ic0DAG->DAG_ptr[i + 1] = ic0DAG->DAG_ptr[i] + diagptr[i] - rowptr[i] + 1;
                std::copy(colidx + rowptr[i], colidx + diagptr[i] + 1, ic0DAG->DAG_set.begin() + ic0DAG->DAG_ptr[i]);
            }
            assert(ic0DAG->DAG_ptr.back() == lowerNnz);
            assert(ic0DAG->DAG_set.back() == n - 1);

            ic0DAG->findLevelsPostOrder(levelset);
            int nlevels = levelset->getLevels();
            // printf("nlevels:%d\n", nlevels);

#ifndef NDEBUG
            LevelSet *levelsetVf = new LevelSet();
            ic0DAG->findLevels(levelsetVf);
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
            // ilu0csr_uplooking_levelset_kernel(this->csrA, levelset, lu);
            spic0_csr_uL_levelset_kernel(this->csrA, lu, levelset);
            t1.measure_elasped_time();
            return t1;
        }

    public:
        IC_LevelSet(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName, int num_thread)
            : ICDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_lu = correct_lu;
            this->num_thread = num_thread;

            levelset = new LevelSet();
            ic0DAG = nullptr;
        }
        ~IC_LevelSet()
        {
            if (ic0DAG)
                delete ic0DAG;
            delete levelset;
        }

        int getWavefront() const { return levelset->getLevels(); }
    };


    /********************************* Reverse and Forward Tree with P2P and node Maping in Four rule  ****************************** */
    class IC_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule : public ICDemo
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
#pragma omp barrier

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
            level_nthreads_group_IC0(levelset, csrA, group_ptr.data(), group_set.data(), this->num_thread, this->nthread_per_level);
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
                spic0_csr_uL_serial(csrA, lu);
                t1.measure_elasped_time();
            }
            else
            {
                t1.start_timer();
                spic0_csr_uL_p2p_group_merge_no_perm(csrA, lu, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                // ilu0_p2p_csr_group_merge_no_perm_alloc(csrA, lu, levelset_merge, nthread_per_level, group_ptr.data(), group_set.data(), task, schedule);
                t1.measure_elasped_time();
            }

            return t1;
        }

    public:
        IC_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName, int num_thread)
            : ICDemo(n, nnz, algName)
        {
            this->csrA = csrA;
            this->correct_lu = correct_lu;
            this->num_thread = num_thread;

            levelset_merge = new LevelSet();
            this->dag_group = new DAG();
            this->taskDAG = new DAG();
        }

        ~IC_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule()
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