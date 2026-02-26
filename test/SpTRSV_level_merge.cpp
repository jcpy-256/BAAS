#include <iostream>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <sstream>
#include <iomanip>

#include "SPM.hpp"
#include "Merge.hpp"
#include "papiProfiling.hpp"

// #include "SpTRSV.hpp"
#include "SpTRSVRuntime.hpp"
#include "MarcoUtils.hpp"
#include "FusionDemo.hpp"
#include "csv_utils.hpp"

using namespace std;

using namespace SPM;
using namespace Merge;
using namespace SpTRSVRT;
using namespace IO_Utils;

#define REPEAT_BENCH 100

int main(int argc, char *argv[])
{
    for (int i = 0; i < argc; i++)
    {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

    char *matrix_file = argv[1]; // matrix file
    int nthread = atoi(argv[2]); // the max threads

    int maxThread = omp_get_max_threads();
    printf("max num thread: %d\n", maxThread);
    printf("running num thread: %d\n", nthread);
    omp_set_num_threads(nthread);
    // omp_set_proc_bind(omp_proc_bind_close);
    char *res_csv = argv[3]; // the result csv file
    double alpha = 1.0;
    // int num_test = 100;
    int benchmark = atoi(argv[4]);

    int num_test = benchmark;

    bool isLower = true;
    int base = 0;
    CSR *csrA = new CSR(matrix_file, isLower, base);

    int m = csrA->m, n = csrA->n;
    int nnz = csrA->getNnz();
    double *correct_x = MALLOC(double, n);
    std::fill_n(correct_x, m, alpha);
    assert(m == n);

    CSC *cscA = new CSC(m, n, nnz);
    CSR::transpositionToCSC(csrA, cscA);

    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("Matrix_Name");
    Runtime_headers.emplace_back("row");
    Runtime_headers.emplace_back("nnz");
    Runtime_headers.emplace_back("test_turns");
    Runtime_headers.emplace_back("Algorithm");
    // Runtime_headers.emplace_back("Kernel");
    Runtime_headers.emplace_back("Core");
    Runtime_headers.emplace_back("Scheduling_Time");
    Runtime_headers.emplace_back("Executor_Runtime");
    Runtime_headers.emplace_back("nlevel");
    // Runtime_headers.emplace_back("Profitable");

    std::ostringstream parStr;
    parStr << std::fixed << std::setprecision(2) << "_lws_" << LEVEL_WINDOW_SIZE << "_amf_" << AMPLIFY_FACTOR << "_aha_" << (AWARD_ALPHA_PARMETER)
           << "_gma_" << (OVER_REGULARIZATION_GMMA_PARMETER) << "_bt_" << (LOAD_REGULARIZATION_BETA_PARMETER);

    // std::string Data_name = "./output/csv/SpTRSV_Merge_O3_" + std::string("thread_") + std::to_string(nthread) + std::string("_") + std::string(res_csv) + parStr.str();
    std::string Data_name = "../output/csv/TRSV/SpTRSV_BAAS_O3_" + std::string("thread_") + std::to_string(nthread) + std::string("_") + std::string(res_csv);
    CSVManager runtime_csv(Data_name, "some address", Runtime_headers, false);

    // *************************** SpTRSV LL Serial **********************************
    SpRTSV_Serial *trsv_serial = new SpRTSV_Serial(csrA, correct_x, n, nnz, alpha, "Serial");
    trsv_serial->set_num_test(num_test);
    TimeMeasure serial_time = trsv_serial->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Serial", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_serial->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(serial_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    // double *x_serial = trsv_serial->getSolution();
    // std::copy(x_serial, x_serial + m, correct_x);
    // runtime_csv.addElementToRecord()
    delete trsv_serial;

    // ***************************** SpTRSV LL MKL ********************
    bool isMKLOpt = true;
    std::string mklAlg = isMKLOpt ? "MKLOpt" : "MKL";
    SpTRSV_MKL *trsv_mkl = new SpTRSV_MKL(csrA, correct_x, n, nnz, alpha, mklAlg, nthread, isMKLOpt);
    trsv_mkl->set_num_test(num_test);
    TimeMeasure trsv_mkl_time = trsv_mkl->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord(mklAlg, "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_mkl->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_mkl_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    //   double *x_mkl= trsv_mkl->getSolution();
    // std::copy(x_mkl, x_mkl + m, correct_x);
    delete trsv_mkl;

    // ************************** SpTRSV LL Wavefront ********************************
    SpTRSV_Wavefront *trsv_wavefront = new SpTRSV_Wavefront(csrA, correct_x, n, nnz, alpha, "Wavefront", nthread);
    trsv_wavefront->set_num_test(num_test);
    TimeMeasure wf_time = trsv_wavefront->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Wavrfront", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_wavefront->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(wf_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_wavefront->getWavefront(), "nlevel");
    runtime_csv.addRecord();
    delete trsv_wavefront;

    /******************************* SpTRSV LL Wavefront dual group all merge ******************** */
    SpTRSV_DUAL_TREE_BFS_WF_Merge *trsv_dual_group_merge = new SpTRSV_DUAL_TREE_BFS_WF_Merge(csrA, cscA, correct_x, n, nnz, alpha, "Dual Group Merge", nthread);
    trsv_dual_group_merge->set_num_test(num_test);
    TimeMeasure trsv_dual_group_merge_time = trsv_dual_group_merge->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Dual Group Merge", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_dual_group_merge->getAnalysisTime(), "Scheduling_Time");
    // runtime_csv.addElementToRecord(1, "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_dual_group_merge->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(trsv_dual_group_merge->getWavefront(), "nlevel");
    runtime_csv.addRecord();
    delete trsv_dual_group_merge;

    /******************************* SpTRSV LL Wavefront dual group merge with p2p (BAASWOS)******************** */
    SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P *trsv_dual_group_merge_p2p = new SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P(csrA, cscA, correct_x, n, nnz, alpha, "Dual Group Merge P2P", nthread, false);
    trsv_dual_group_merge_p2p->set_num_test(num_test);
    TimeMeasure trsv_dual_group_merge_p2p_time = trsv_dual_group_merge_p2p->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Dual Group Merge P2P", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p->getAnalysisTime(), "Scheduling_Time");
    // runtime_csv.addElementToRecord(1, "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete trsv_dual_group_merge_p2p;

    /******************************* SpTRSV LL Wavefront dual group merge with p2p  and Perm (BAASWOS Permutation)********************* */
    SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P *trsv_dual_group_merge_p2p_perm = new SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P(csrA, cscA, correct_x, n, nnz, alpha, "Dual Group Merge P2P Perm", nthread, true);
    trsv_dual_group_merge_p2p_perm->set_num_test(num_test);
    TimeMeasure trsv_dual_group_merge_p2p_perm_time = trsv_dual_group_merge_p2p_perm->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Dual Group Merge P2P Perm", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_perm->getAnalysisTime(), "Scheduling_Time");
    // runtime_csv.addElementToRecord(1, "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_perm_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_perm->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete trsv_dual_group_merge_p2p_perm;


    /******************************* SpTRSV LL Wavefront dual group merge with p2p and mapping of FourRule (BAAS)******************** */
    SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule *trsv_dual_group_merge_p2p_mapfourrule = new SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule(csrA, cscA, correct_x, n, nnz, alpha, "Dual Group Merge P2P MappingFourRule", nthread, false);
    trsv_dual_group_merge_p2p_mapfourrule->set_num_test(num_test);

    TimeMeasure trsv_dual_group_merge_p2p_mapfourrule_time = trsv_dual_group_merge_p2p_mapfourrule->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Dual Group Merge P2P MappingFourRule", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_mapfourrule->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_mapfourrule_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_mapfourrule->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete trsv_dual_group_merge_p2p_mapfourrule;

    // // /******************************* SpTRSV LL Wavefront dual group merge with p2p and mapping of four rule, and Maitrx Perm (BAAS Permutation)******************** */
    SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule *trsv_dual_group_merge_p2p_mapfourrule_perm = new SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule(csrA, cscA, nullptr, n, nnz, alpha, "Dual Group Merge P2P MappingFourRule Perm", nthread, true);
    trsv_dual_group_merge_p2p_mapfourrule_perm->set_num_test(num_test);

    TimeMeasure trsv_dual_group_merge_p2p_mapfourrule_perm_time = trsv_dual_group_merge_p2p_mapfourrule_perm->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Dual Group Merge P2P MappingFourRule Perm", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_mapfourrule_perm->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_mapfourrule_perm_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_dual_group_merge_p2p_mapfourrule_perm->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete trsv_dual_group_merge_p2p_mapfourrule_perm;

    // ###################################################################################################
    /************************************* Ablation Study Test **************************************** */
    // ###################################################################################################
    
    /********************************* BAASWOAS ******************************/
    SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationBase *trsv_mapfourrule_ablationbase = new SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationBase(csrA, cscA, correct_x, n, nnz, alpha, "P2P MappingFourRule AblationBase", nthread, false);
    trsv_mapfourrule_ablationbase->set_num_test(num_test);
    TimeMeasure trsv_mapfourrule_ablationbase_time = trsv_mapfourrule_ablationbase->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("P2P MappingFourRule AblationBase", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_mapfourrule_ablationbase->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_mapfourrule_ablationbase_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_mapfourrule_ablationbase->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    // delete trsv_mapfourrule_ablationbase;

    /********************************* BAASWOAS Permutation******************************/
    SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationBase *trsv_mapfourrule_perm_ablationbase = new SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationBase(csrA, cscA, correct_x, n, nnz, alpha, "P2P MappingFourRule AblationBase", nthread, true);
    trsv_mapfourrule_perm_ablationbase->set_num_test(num_test);
    TimeMeasure trsv_mapfourrule_perm_ablationbase_time = trsv_mapfourrule_perm_ablationbase->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("P2P MappingFourRule AblationBase Perm", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_mapfourrule_perm_ablationbase->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_mapfourrule_perm_ablationbase_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_mapfourrule_perm_ablationbase->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete trsv_mapfourrule_perm_ablationbase;

    /***************************** BAASWOA  *************************************/
    SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationWOAggregation *trsv_mapfourrule_ablationbase_woaggregation = new SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationWOAggregation(csrA, cscA, correct_x, n, nnz, alpha, "P2P MappingFourRuleWOA", nthread, false);
    trsv_mapfourrule_ablationbase_woaggregation->set_num_test(num_test);
    TimeMeasure trsv_mapfourrule_ablationbase_woaggregation_time = trsv_mapfourrule_ablationbase_woaggregation->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("P2P MappingFourRuleWOA", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_mapfourrule_ablationbase_woaggregation->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_mapfourrule_ablationbase_woaggregation_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_mapfourrule_ablationbase_woaggregation->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete trsv_mapfourrule_ablationbase_woaggregation;

    /***************************** BAASWOA  Permutation *************************************/
    SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationWOAggregation *trsv_mapfourrule_perm_ablationbase_woa = new SpTRSV_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRuleAblationWOAggregation(csrA, cscA, correct_x, n, nnz, alpha, " P2P MappingFourRuleWOA Perm", nthread, true);
    trsv_mapfourrule_perm_ablationbase_woa->set_num_test(num_test);
    TimeMeasure trsv_mapfourrule_perm_ablationbase_woa_time = trsv_mapfourrule_perm_ablationbase_woa->evaluate();
    runtime_csv.addElementToRecord(matrix_file, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("P2P MappingFourRuleWOA Perm", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(trsv_mapfourrule_perm_ablationbase_woa->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(trsv_mapfourrule_perm_ablationbase_woa_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(trsv_mapfourrule_perm_ablationbase_woa->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete trsv_mapfourrule_perm_ablationbase_woa;

    FREE(correct_x);
    // printf("end\n");

    // printf("matrix n:%d\n", csrA->n);

    fflush(stdout);
    delete csrA;

    return 0;
}