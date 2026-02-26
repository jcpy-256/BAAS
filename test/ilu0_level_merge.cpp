#include <iostream>
#include <cstdio>
#include <iomanip>

#include "SPM.hpp"
#include "Merge.hpp"
#include "papiProfiling.hpp"

// #include "SpTRSV.hpp"
#include "ILURuntime.hpp"
#include "MarcoUtils.hpp"
#include "csv_utils.hpp"

using namespace std;

using namespace SPM;
using namespace Merge;
using namespace ILURT;
using namespace IO_Utils;

CSR *forceLowerSymmmetric(const CSR *csrA);

bool compareCSRMatrix(int m, int n, const int *rowptr, const int *colidx, const double *refVal, const double *actVal, const double tolerance);

void ilu0csr_merge(CSR *A);

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

    CSR *csrA = new CSR(filename);

    const int m = csrA->m;
    const int n = csrA->n;
    const int nnz = csrA->getNnz();
    assert(m == n);
    double *correct_lu = MALLOC(double, nnz);
    CHECK_POINTER(correct_lu);
    std::copy(csrA->values, csrA->values + nnz, correct_lu);
    ilu0csr_uplooking_ref(csrA, correct_lu);

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

    std::string Data_name = "../output/csv/ILU0/ILU0_BAAS_O3_" + std::string("thread_") + std::to_string(nthread) + std::string("_") + std::string(res_csv);
    CSVManager runtime_csv(Data_name, "some address", Runtime_headers, false);

    // ******************************* ILU0 Serial ********************************
    // ILU_Serial *ilu0_serial = new ILU_Serial(csrA, nullptr, n, nnz, "Serial");
    // ilu0_serial->set_num_test(num_test);
    // TimeMeasure serial_time = ilu0_serial->evaluate();
    // runtime_csv.addElementToRecord(filename, "Matrix_Name");
    // runtime_csv.addElementToRecord(n, "row");
    // runtime_csv.addElementToRecord(nnz, "nnz");
    // runtime_csv.addElementToRecord(num_test, "test_turns");
    // runtime_csv.addElementToRecord("Serial", "Algorithm");
    // runtime_csv.addElementToRecord(nthread, "Core");
    // runtime_csv.addElementToRecord(ilu0_serial->getAnalysisTime(), "Scheduling_Time");
    // runtime_csv.addElementToRecord(serial_time.getTime(), "Executor_Runtime");
    // runtime_csv.addElementToRecord(1, "nlevel");
    // runtime_csv.addRecord();
    // double *lu_serial = ilu0_serial->getSolution();
    // std::copy(lu_serial, lu_serial + nnz, correct_lu);
    // delete ilu0_serial;

    // ******************************* ILU0 MKL ****************************************
    bool opt = true;
    std::string mklAlg = opt == true ? "MKLOpt" : "MKL";
    ILU_MKL *ilu0_mkl = new ILU_MKL(csrA, correct_lu, n, nnz, mklAlg);
    ilu0_mkl->set_num_test(num_test);
    TimeMeasure mkl_time = ilu0_mkl->evaluate();
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord(mklAlg, "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(ilu0_mkl->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(mkl_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete ilu0_mkl;

    /********************************** ILU0 LevelSet ********************************* */
    // ILU_LevelSet *ilu0_levelset = new ILU_LevelSet(csrA, correct_lu, n, nnz, "LevelSet", nthread);
    // ilu0_levelset->set_num_test(num_test);
    // TimeMeasure levelset_time = ilu0_levelset->evaluate();
    // runtime_csv.addElementToRecord(filename, "Matrix_Name");
    // runtime_csv.addElementToRecord(n, "row");
    // runtime_csv.addElementToRecord(nnz, "nnz");
    // runtime_csv.addElementToRecord(num_test, "test_turns");
    // runtime_csv.addElementToRecord("LevelSet", "Algorithm");
    // runtime_csv.addElementToRecord(nthread, "Core");
    // runtime_csv.addElementToRecord(ilu0_levelset->getAnalysisTime(), "Scheduling_Time");
    // runtime_csv.addElementToRecord(levelset_time.getTime(), "Executor_Runtime");
    // runtime_csv.addElementToRecord(ilu0_levelset->getWavefront(), "nlevel");
    // runtime_csv.addRecord();
    // delete ilu0_levelset;
    // fflush(stdout);


    /************************************ ILU0 dual group merge p2p using mapping with FourRule (BAAS) *********************************  */
    ILU_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule *ilu0_dual_group_p2p_mapfourrule = new ILU_DUAL_TREE_BFS_WF_Merge_P2P_MapFourRule(csrA, correct_lu, n, nnz, "Dual Group Merge P2P MappingFourRule", nthread);
    ilu0_dual_group_p2p_mapfourrule->set_num_test(num_test);
    TimeMeasure ilu0_dual_group_p2p_mapfourrule_time = ilu0_dual_group_p2p_mapfourrule->evaluate();
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(n, "row");
    runtime_csv.addElementToRecord(nnz, "nnz");
    runtime_csv.addElementToRecord(num_test, "test_turns");
    runtime_csv.addElementToRecord("Dual Group Merge P2P MappingFourRule", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(ilu0_dual_group_p2p_mapfourrule->getAnalysisTime(), "Scheduling_Time");
    runtime_csv.addElementToRecord(ilu0_dual_group_p2p_mapfourrule_time.getTime(), "Executor_Runtime");
    runtime_csv.addElementToRecord(ilu0_dual_group_p2p_mapfourrule->getWavefront(), "nlevel");
    // runtime_csv.addElementToRecord(1, "nlevel");
    runtime_csv.addRecord();
    delete ilu0_dual_group_p2p_mapfourrule;

    FREE(correct_lu);
    delete csrA;
}

bool compareCSRMatrix(int m, int n, const int *rowptr, const int *colidx, const double *refVal, const double *actVal, const double tolerance)
{
    // int ret = true;
    for (int i = 0; i < m; i++)
    {
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
        {
            // ret = diff > tolerance ? false : true;
            if (fabs(refVal[j] - actVal[j]) > tolerance)
            {
                printf("error is %lf\n", fabs(refVal[j] - actVal[j]));
                return false;
            }
        }
    }
    return true;
}

CSR *forceLowerSymmmetric(const CSR *csrA)
{

    const int m = csrA->m;
    const int n = csrA->n;
    const int *csrColIdx = csrA->colidx;
    const int *csrRowPtr = csrA->rowptr;
    const double *csrVal = csrA->values;
    if (m != n)
    {
        std::cerr << "the matrix is no a square matrix, and the force symmetric couldn't proceed!\n";
        exit(1);
    }
    int nnz = csrA->getNnz();

    int *csrRowCounter = MALLOC(int, m + 1);
    // std::copy(csrA->rowptr, csrA->rowptr + m + 1, csrRowCounter);
    memcpy(csrRowCounter, csrA->rowptr, (m + 1) * sizeof(int));

    for (int i = 0; i < m; i++)
    {
        csrRowCounter[i] = csrRowCounter[i + 1] - csrRowCounter[i];
    }

    double maxAbsoluteValue = __DBL_MIN__;

    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
        {
            if (i != csrColIdx[j])
            {
                csrRowCounter[csrColIdx[j]]++;
            }
        }
    }

    exclusive_scan(csrRowCounter, m + 1);
    // assert(csrRowCounter[m] == )

    int nnz_tmp = csrRowCounter[m]; // 0-based
    assert(nnz_tmp == 2 * nnz - m);
    int *csrRowPtr_alias = (int *)malloc((m + 1) * sizeof(int));
    CHECK_POINTER(csrRowPtr_alias);
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    CHECK_POINTER(csrColIdx_alias);
    double *csrVal_alias = (double *)malloc(nnz_tmp * sizeof(double));
    CHECK_POINTER(csrVal_alias);

    memcpy(csrRowPtr_alias, csrRowCounter, (m + 1) * sizeof(int));
    memset(csrRowCounter, 0, (m + 1) * sizeof(int));

    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
        {
            if (i != csrColIdx[j]) // not a diagonal element
            {
                // 原始值
                int row = i;
                int col = csrColIdx[j];
                assert(row > col);
                // val =
                int offset = csrRowPtr_alias[row] + csrRowCounter[row];
                csrColIdx_alias[offset] = col;
                csrVal_alias[offset] = csrVal[j];
                csrRowCounter[row]++;
                // 对称值

                row = csrColIdx[j];
                col = i;
                assert(col > row);
                offset = csrRowPtr_alias[row] + csrRowCounter[row];
                csrColIdx_alias[offset] = col;
                csrVal_alias[offset] = csrVal[j];
                csrRowCounter[row]++;
            }
            else // diagonal element
            {
                // assert(row == col);
                int offset = csrRowPtr_alias[i] + csrRowCounter[i];
                csrColIdx_alias[offset] = csrColIdx[j];
                csrVal_alias[offset] = csrVal[j];
                // csrVal_alias[offset] = maxAbsoluteValue;
                csrRowCounter[i]++;
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        quicksort(csrColIdx_alias, csrVal_alias, csrRowPtr_alias[i], csrRowPtr_alias[i + 1] - 1);
    }

    CSR *csrTmp = new CSR();
    csrTmp->m = m;
    csrTmp->n = n;
    csrTmp->colidx = csrColIdx_alias;
    csrTmp->rowptr = csrRowPtr_alias;
    csrTmp->values = csrVal_alias;

    csrTmp->diag = MALLOC(double, m);
    csrTmp->idiag = MALLOC(double, m);
    printf("orig m:%d, n:%d, nnz:%d\n", m, n, nnz);

    fflush(stdout);
    csrTmp->constructDiagPtr();
    // csrTmp->computeInverseDiag()

    CSR *csrSym = new CSR(*csrTmp);
    delete csrTmp;

    return csrSym;
}

void ilu0csr_merge(CSR *A)
{
    assert(A->diagptr);
    const int nnz = A->getNnz();
    const int m = A->m;
    const int n = A->n;
    const int *rowptr = A->rowptr;
    const int *colidx = A->colidx;
    const int *diagptr = A->diagptr;
    double *val = A->values;
    using SPM::DAG, SPM::LevelSet;
    using SPM::DAG_MAT;

    if (m != n)
    {
        throw std::runtime_error("this matrix is not a square matrix");
    }
    std::vector<int> colIdxMapping(n, -1);

    // construct DAG by lower triangular part of matrix A
    // traverse CSR A to get the non-zero number in lower triangular
    int lower_nnz = 0;
    for (int i = 0; i < m; i++)
    {
        lower_nnz += diagptr[i] - rowptr[i] + 1;
    }
    // printf("lower triangular part has %d nnz\n", lower_nnz);

    DAG *ilu0DAG = new DAG(m, lower_nnz, DAG_MAT::DAG_CSR);
    ilu0DAG->DAG_ptr[0] = 0;
    for (int i = 0; i < m; i++)
    {
        ilu0DAG->DAG_ptr[i + 1] = ilu0DAG->DAG_ptr[i] + diagptr[i] - rowptr[i] + 1;
        std::copy(colidx + rowptr[i], colidx + diagptr[i] + 1, ilu0DAG->DAG_set.begin() + ilu0DAG->DAG_ptr[i]);
    }
    printf("DAG ptr back is %d\n", ilu0DAG->DAG_ptr.back());

    assert(ilu0DAG->DAG_ptr.back() == lower_nnz);
    assert(ilu0DAG->DAG_set.back() == m - 1);
}
