#include <iostream>
#include <cstdio>
#include <iomanip>

#include "SPM.hpp"
#include "Merge.hpp"
#include "papiProfiling.hpp"

// #include "SpTRSV.hpp"
#include "ICRuntime.hpp"
#include "MarcoUtils.hpp"
#include "csv_utils.hpp"

using namespace std;

using namespace SPM;
using namespace Merge;
using namespace ICRT;
using namespace IO_Utils;

bool check_ic0(const int n, const int *rowptr, const int *diagptr, const double *first, const double *second, const double tol)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = rowptr[i]; k <= diagptr[i]; k++)
        {
            if (fabs(first[k] - second[k]) > tol)
            {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    for (int i = 0; i < argc; i++)
    {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

    if (argc < 4)
    {
        fprintf(stderr, "this program is lack of parameters!\n");
    }
    const char *filename = argv[1];
    int nthread = atoi(argv[2]);
    // int nthread = 16;

    omp_set_num_threads(nthread);
    char *res_csv = argv[3];

    int benchmark = atoi(argv[4]);

    int num_test = benchmark;

    printf("-------------------------matrix %s--------------------\n", filename);
    const bool isLower = true;
    const int BASE = 0;
    CSR *csrA = new CSR(filename); // load spd matrix and construct diagptr

    const int m = csrA->m;
    const int n = csrA->n;
    const int nnz = csrA->getNnz();
    assert(m == n);

    double *correct_lu = MALLOC(double, nnz);

    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("Matrix_Name");
    Runtime_headers.emplace_back("row");
    Runtime_headers.emplace_back("nnz");
    Runtime_headers.emplace_back("test_turns");
    Runtime_headers.emplace_back("Algorithm");
    Runtime_headers.emplace_back("Core");
    Runtime_headers.emplace_back("Scheduling_Time");
    Runtime_headers.emplace_back("Executor_Runtime");
    Runtime_headers.emplace_back("nlevel");

    std::string Data_name = "../output/csv/IC0/IC0_BAAS_O3_" + std::string("thread_") + std::to_string(nthread) + std::string("_") + std::string(res_csv);
    CSVManager runtime_csv(Data_name, "some address", Runtime_headers, false);

    // ***************************************** IC0 Serial *****************************************
    double *serial_lu = MALLOC(double, nnz);
    std::copy(csrA->values, csrA->values + nnz, serial_lu);
    spic0_csr_uL_serial(csrA, serial_lu);
    std::copy(serial_lu, serial_lu + nnz, correct_lu);
    FREE(serial_lu);

    IC_Serial *ic0_serial = new IC_Serial(csrA, correct_lu, n, nnz, "Serial");
    ic0_serial->set_num_test(num_test);
    TimeMeasure ic0_serial_time = ic0_serial->evaluate();
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Serial", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(ic0_serial->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(ic0_serial_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete ic0_serial;


    // ***************************************** IC0 LevelSet *****************************************
    IC_LevelSet *ic0_levelset = new IC_LevelSet(csrA, correct_lu, n, nnz, "LevelSet", nthread);
    ic0_levelset->set_num_test(num_test);
    TimeMeasure ic0_levelset_time = ic0_levelset->evaluate();
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("LevelSet", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(ic0_levelset->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(ic0_levelset_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(ic0_levelset->getWavefront(), "nlevel");
    runtime_csv.addRecord();
    delete ic0_levelset;

    IC_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule *ic0_dual_group_p2p_mapfourrule = new IC_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule(csrA, correct_lu, n, nnz, "Dual Group Merge P2P MappingFourRule", nthread);
    ic0_dual_group_p2p_mapfourrule->set_num_test(num_test);
    TimeMeasure ic0_dual_group_p2p_mapfourrule_time = ic0_dual_group_p2p_mapfourrule->evaluate();
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Dual Group Merge P2P MappingFourRule", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(ic0_dual_group_p2p_mapfourrule->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(ic0_dual_group_p2p_mapfourrule_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(ic0_dual_group_p2p_mapfourrule->getWavefront(), "nlevel");
    runtime_csv.addRecord();
    delete ic0_dual_group_p2p_mapfourrule;

    return 0;
}
