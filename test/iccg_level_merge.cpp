/***
 * this file implements the preconditioner-CG linear solver for Ax = b, A is a SPD matrix
 */
#include "SPM.hpp"
#include "Merge.hpp"
#include "Schedule.hpp"
#include "csv_utils.hpp"
#include "MarcoUtils.hpp"
#include "MathUtils.hpp"
#include "BlasDemo.hpp"
#include "time_utils.hpp"
#include "mkl.h"

using namespace SPM;
using namespace Merge;
using namespace IO_Utils;
using Schedule::Scheduler;

// 定义检查 sparse_status_t 的宏
#define CHECK_MKL_SPARSE(status)                                                \
    do                                                                          \
    {                                                                           \
        sparse_status_t _status = (status);                                     \
        if (_status != SPARSE_STATUS_SUCCESS)                                   \
        {                                                                       \
            fprintf(stderr, "MKL sparse error at %s:%d: ", __FILE__, __LINE__); \
            switch (_status)                                                    \
            {                                                                   \
            case SPARSE_STATUS_NOT_INITIALIZED:                                 \
                fprintf(stderr, "SPARSE_STATUS_NOT_INITIALIZED\n");             \
                break;                                                          \
            case SPARSE_STATUS_ALLOC_FAILED:                                    \
                fprintf(stderr, "SPARSE_STATUS_ALLOC_FAILED\n");                \
                break;                                                          \
            case SPARSE_STATUS_INVALID_VALUE:                                   \
                fprintf(stderr, "SPARSE_STATUS_INVALID_VALUE\n");               \
                break;                                                          \
            case SPARSE_STATUS_EXECUTION_FAILED:                                \
                fprintf(stderr, "SPARSE_STATUS_EXECUTION_FAILED\n");            \
                break;                                                          \
            case SPARSE_STATUS_INTERNAL_ERROR:                                  \
                fprintf(stderr, "SPARSE_STATUS_INTERNAL_ERROR\n");              \
                break;                                                          \
            case SPARSE_STATUS_NOT_SUPPORTED:                                   \
                fprintf(stderr, "SPARSE_STATUS_NOT_SUPPORTED\n");               \
                break;                                                          \
            default:                                                            \
                fprintf(stderr, "Unknown error (%d)\n", _status);               \
                break;                                                          \
            }                                                                   \
            /* 可以选择在此处退出程序或执行其他错误处理 */                      \
            /* exit(EXIT_FAILURE); */                                           \
        }                                                                       \
    } while (0)

enum PCG_TYPE
{
    PCG_REF,
    PCG_MKL,
    PCG_OPT,
    PCG_PERM,
    PCG_OMP,
};

void pcg_mkl(const CSR *A, const double *b, double *x, const int maxiter, const double tol, std::map<std::string, double> &testMap);

void pcg_ref(const CSR *A, const double *, double *x, const int maxiter, const double tol, std::map<std::string, double> &testMap);

void pcg_opt(const CSR *A, const double *b, double *x, const int maxiter, const double tol, const int num_thread, std::map<std::string, double> &testMap);
void pcg_opt_perm(const CSR *A, const double *b, double *x, const int maxiter, const double tol, const int num_thread, std::map<std::string, double> &testMap);

void IC0_preconditioner(const CSR *A, CSR *&L, CSR *&U, PCG_TYPE type, Scheduler *scheduler);

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        fprintf(stderr, "The program need more than three parameters. \n");
        exit(1);
    }

    printf("parameters number: %d\n", argc);
    
    for (int i = 0; i < argc; i++) {
        printf("argv[%d] = %s\n", i, argv[i]);
    }
    const char *filename = argv[1];
    const int nthread = atoi(argv[2]);
    const char *res_csv = argv[3];
    omp_set_num_threads(nthread);
    printf("=======================Preconditioner-CG for Matrix: %s======================\n", filename);
    CSR *csrA = new CSR(filename);
    printf("m: %d, n: %d, nnz: %d\n", csrA->m, csrA->n, csrA->getNnz());
    fflush(stdout);

    double *b = MALLOC(double, csrA->m);
    double *x = MALLOC(double, csrA->m);
    CHECK_POINTER(b);
    CHECK_POINTER(x);
#pragma omp parallel for
    for (int i = 0; i < csrA->m; i++)
    {
        b[i] = 1;
        x[i] = 0.0;
    }

    int maxiter = atoi(argv[4]);
    double tol = 1.0e-8;

    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("Matrix_Name");
    Runtime_headers.emplace_back("row");
    Runtime_headers.emplace_back("nnz");
    Runtime_headers.emplace_back("iterations");
    Runtime_headers.emplace_back("Algorithm");
    // Runtime_headers.emplace_back("Kernel");
    Runtime_headers.emplace_back("Core");
    Runtime_headers.emplace_back("pre_time");
    Runtime_headers.emplace_back("ic_time");
    Runtime_headers.emplace_back("trsv_time");
    Runtime_headers.emplace_back("spmv_time");
    Runtime_headers.emplace_back("iter_time");
    Runtime_headers.emplace_back("total_time");
    std::string Data_name = "../output/csv/ICCG/PCG_BAAS_O3_" + std::string("thread_") + std::to_string(nthread) + std::string("_iterate") + "_" + std::to_string(maxiter) + "_" + std::string(res_csv);
    CSVManager runtime_csv(Data_name, "some address", Runtime_headers, false);

    std::map<std::string, double> testMap;
    testMap["iterations"] = 0.0;
    testMap["pre_time"] = 0.0;
    testMap["ic_time"] = 0.0;
    testMap["trsv_time"] = 0.0;
    testMap["spmv_time"] = 0.0;
    testMap["iter_time"] = 0.0;
    testMap["total_time"] = 0.0;

    // printf("m: %d\n", csrA->m);
    pcg_ref(csrA, b, x, maxiter, tol, testMap);
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(csrA->m, "row");
    runtime_csv.addElementToRecord(csrA->getNnz(), "nnz");
    runtime_csv.addElementToRecord((int)testMap["iterations"], "iterations");
    runtime_csv.addElementToRecord("Serial", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(testMap["pre_time"], "pre_time");
    runtime_csv.addElementToRecord(testMap["ic_time"], "ic_time");
    runtime_csv.addElementToRecord(testMap["trsv_time"], "trsv_time");
    runtime_csv.addElementToRecord(testMap["spmv_time"], "spmv_time");
    runtime_csv.addElementToRecord(testMap["iter_time"], "iter_time");
    runtime_csv.addElementToRecord(testMap["total_time"], "total_time");
    runtime_csv.addRecord();

    // 时间重置
    for (auto it = testMap.begin(); it != testMap.end(); it++)
    {
        it->second = 0.0;
    }
    pcg_opt(csrA, b, x, maxiter, tol, nthread, testMap);
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(csrA->m, "row");
    runtime_csv.addElementToRecord(csrA->getNnz(), "nnz");
    runtime_csv.addElementToRecord((int)testMap["iterations"], "iterations");
    runtime_csv.addElementToRecord("P2P_RULE", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(testMap["pre_time"], "pre_time");
    runtime_csv.addElementToRecord(testMap["ic_time"], "ic_time");
    runtime_csv.addElementToRecord(testMap["trsv_time"], "trsv_time");
    runtime_csv.addElementToRecord(testMap["spmv_time"], "spmv_time");
    runtime_csv.addElementToRecord(testMap["iter_time"], "iter_time");
    runtime_csv.addElementToRecord(testMap["total_time"], "total_time");
    runtime_csv.addRecord();

    for (auto it = testMap.begin(); it != testMap.end(); it++)
    {
        it->second = 0.0;
    }
    // pcg_opt_perm(csrA, b, x, maxiter, tol, nthread, testMap);
    pcg_opt_perm(csrA, b, x, maxiter, tol, nthread, testMap);
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(csrA->m, "row");
    runtime_csv.addElementToRecord(csrA->getNnz(), "nnz");
    runtime_csv.addElementToRecord((int)testMap["iterations"], "iterations");
    runtime_csv.addElementToRecord("P2P_RULE_Perm", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(testMap["pre_time"], "pre_time");
    runtime_csv.addElementToRecord(testMap["ic_time"], "ic_time");
    runtime_csv.addElementToRecord(testMap["trsv_time"], "trsv_time");
    runtime_csv.addElementToRecord(testMap["spmv_time"], "spmv_time");
    runtime_csv.addElementToRecord(testMap["iter_time"], "iter_time");
    runtime_csv.addElementToRecord(testMap["total_time"], "total_time");
    runtime_csv.addRecord();

    // 时间重置
    for (auto it = testMap.begin(); it != testMap.end(); it++)
    {
        it->second = 0.0;
    }

    pcg_mkl(csrA, b, x, maxiter, tol, testMap);
    runtime_csv.addElementToRecord(filename, "Matrix_Name");
    runtime_csv.addElementToRecord(csrA->m, "row");
    runtime_csv.addElementToRecord(csrA->getNnz(), "nnz");
    runtime_csv.addElementToRecord((int)testMap["iterations"], "iterations");
    runtime_csv.addElementToRecord("MKL_IMPL", "Algorithm");
    runtime_csv.addElementToRecord(nthread, "Core");
    runtime_csv.addElementToRecord(testMap["pre_time"], "pre_time");
    runtime_csv.addElementToRecord(testMap["ic_time"], "ic_time");
    runtime_csv.addElementToRecord(testMap["trsv_time"], "trsv_time");
    runtime_csv.addElementToRecord(testMap["spmv_time"], "spmv_time");
    runtime_csv.addElementToRecord(testMap["iter_time"], "iter_time");
    runtime_csv.addElementToRecord(testMap["total_time"], "total_time");
    runtime_csv.addRecord();
    return 0;
}

void pcg_opt_perm(const CSR *A, const double *b, double *x, const int maxiter, const double tol, const int num_thread, std::map<std::string, double> &testMap)
{
    printf("------------the optimized perm PCG !-----------------\n");

    double pre_time = 0, ic_time = 0, trsv_time = 0, spmv_time = 0, iter_time = 0, total_time = 0;
    double *r = MALLOC(double, A->m);
    CHECK_POINTER(r);
    copyVector(r, b, A->m);
    CSR *L, *U;

    total_time -= getCurrentTimeMilli();

    pre_time -= getCurrentTimeMilli();

    // step1. construct schedule
    Scheduler *scheduler = new Scheduler(A, true, true, Schedule::ALG_SCHEDULE::P2P_RULE, num_thread);
    scheduler->preprocessing();     // 进行预处理操作
    scheduler->permPreprocessing(); // 进行perm的预处理
    const int *inversePerm = scheduler->permToOrigByThread.data();
    const int *forwardPerm = scheduler->origToPermByThread.data();
    CSR *APerm = A->permute(forwardPerm, inversePerm, true);

    // step2: MKL SpMV preprocessing
    sparse_matrix_t matrixA;
    struct matrix_descr descA;
    sparse_status_t retA = mkl_sparse_d_create_csr(&matrixA, SPARSE_INDEX_BASE_ZERO, APerm->m, APerm->n,
                                                   APerm->rowptr, APerm->rowptr + 1, APerm->colidx, APerm->values);
    descA.diag = SPARSE_DIAG_NON_UNIT;
    descA.mode = SPARSE_FILL_MODE_FULL; // FULL
    descA.type = SPARSE_MATRIX_TYPE_GENERAL;
    retA = mkl_sparse_set_mv_hint(matrixA, SPARSE_OPERATION_NON_TRANSPOSE, descA, 200);
    CHECK_MKL_SPARSE(retA);
    retA = mkl_sparse_set_memory_hint(matrixA, SPARSE_MEMORY_AGGRESSIVE);
    CHECK_MKL_SPARSE(retA);
    retA = mkl_sparse_optimize(matrixA);
    CHECK_MKL_SPARSE(retA);

    pre_time += getCurrentTimeMilli();

    ic_time -= getCurrentTimeMilli();
    IC0_preconditioner(APerm, L, U, PCG_PERM, scheduler);
    ic_time += getCurrentTimeMilli();

    // printf("After IC0\n");

    // r: r - A * x
    spmv_time -= getCurrentTimeMilli();
    // dcsrmv(A->m, A->values, A->rowptr, A->colidx, x, r, -1, 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, matrixA, descA, x, 1.0, r); // r = -Ax + r
    spmv_time += getCurrentTimeMilli();
    double norm0 = norm(r, A->m);
    double rel_err = 1.0;

    double *z = MALLOC(double, A->m);
    double *y = MALLOC(double, A->m);
    CHECK_POINTER(z);
    CHECK_POINTER(y);

    // printf("update residual !\n");
    // fflush(stdout);
    // z = M\r, where M is preconditioner
    // lower_csr_trsv_serial(*L, y, r);
    // upper_csr_trsv_serial(*U, z, y);
    trsv_time -= getCurrentTimeMilli();
    scheduler->runForwardTRSVPerm(L, y, r);
    scheduler->runBackwardTRSVPerm(U, z, y);
    trsv_time += getCurrentTimeMilli();

    // bool check1 = fvectorEqual(y, yb, A->m, 1.0e-10);
    // check1 = fvectorEqual(z, zb, A->m, 1.0e-10);

    double *p = MALLOC(double, A->m);
    copyVector(p, z, A->m); // 搜索方向
    // double rz = dot(r, z, A->m);
    double rz = cblas_ddot(A->m, r, 1, z, 1);
    int k = 1;

    double *Ap = MALLOC(double, A->m);

    iter_time -= getCurrentTimeMilli();
    int stop_iteration = k;
    // iterate
    while (k <= maxiter)
    {
        double old_rz = rz;
        // step 1: spmv
        spmv_time -= getCurrentTimeMilli();
        // spmv_omp(*A, p, Ap); // Ap = A * p
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrixA, descA, p, 0.0, Ap); // Ap = 1.0 * A*p + 0 * Ap
        spmv_time += getCurrentTimeMilli();

        // step 2: 计算步长 alpha
        // double alpha = old_rz / dot(p, Ap, A->m); // 更新步长
        double alpha = old_rz / cblas_ddot(A->m, p, 1, Ap, 1); // 更新步长

        // step 3: 更新解
        // waxpby(A->m, x, 1, x, alpha, p);
        cblas_daxpby(A->m, alpha, p, 1, 1.0, x, 1);

        // step 4: 更新残差
        // waxpby(A->m, r, 1, r, -alpha, Ap); // r = r - alpha * A * Ap
        cblas_daxpby(A->m, -alpha, Ap, 1, 1.0, r, 1);
        // rel_err = norm(r, A->m) / norm0;
        rel_err = cblas_dnrm2(A->m, r, 1) / norm0;
        if (rel_err < tol)
        {
            stop_iteration = k;
        }
        // break;

        // step 5: 更新预条件残差
        // lower_csr_trsv_serial(*L, y, r);
        // upper_csr_trsv_serial(*U, z, y);
        trsv_time -= getCurrentTimeMilli();
        scheduler->runForwardTRSVPerm(L, y, r);
        scheduler->runBackwardTRSVPerm(U, z, y);
        trsv_time += getCurrentTimeMilli();

        // step 6: 更新搜索方向系数
        // rz = dot(r, z, A->m);
        rz = cblas_ddot(A->m, r, 1, z, 1);
        double beta = rz / old_rz; // 确保新的搜索方向和之间的搜索方向关于A共轭

        // step 7: 更新搜索方向
        // waxpby(A->m, p, 1, z, beta, p); // p = z + beta * p
        cblas_daxpby(A->m, 1.0, z, 1, beta, p, 1);
        ++k;
    }
    iter_time += getCurrentTimeMilli();
    total_time += getCurrentTimeMilli();

    // printf("iter = %d rel_err = %g\n", k, rel_err);

    // double *x_perm = MALLOC(double, A->m);
    // for(int i=0; i < A->m; i++)
    // {

    // }

    double spmv_bytes = (k + 1) * (12. * A->getNnz() + (4. + 2 * 8) * A->m);
    double trsv_bytes = k * (12. * L->getNnz() + 12. * U->getNnz() + (8. + 2 * 4 + 2 * 2 * 8) * L->m);
    // printf("spmv_perf = %g gbps trsv_perf = %g gbps\n", spmv_bytes / spmv_time / 1e9, trsv_bytes / trsv_time / 1e9);
    // delete LU;

    testMap["iterations"] = maxiter;
    testMap["pre_time"] = pre_time;
    testMap["ic_time"] = ic_time;
    testMap["trsv_time"] = trsv_time;
    testMap["spmv_time"] = spmv_time;
    testMap["iter_time"] = iter_time;
    testMap["total_time"] = total_time;

    delete L;
    delete U;
    delete APerm;

    FREE(r);
    FREE(z);
    FREE(y);
    FREE(p);
    FREE(Ap);
    mkl_sparse_destroy(matrixA);
    printf("------------the optimized perm PCG end !-----------------\n");
}

void pcg_opt(const CSR *A, const double *b, double *x, const int maxiter, const double tol, const int num_thread, std::map<std::string, double> &testMap)
{
    printf("------------the optimized PCG !-----------------\n");

    double pre_time = 0, ic_time = 0, trsv_time = 0, spmv_time = 0, iter_time = 0, total_time = 0;
    double *r = MALLOC(double, A->m);
    CHECK_POINTER(r);
    copyVector(r, b, A->m);
    CSR *L, *U;

    total_time -= getCurrentTimeMilli();

    pre_time -= getCurrentTimeMilli();

    // step1. construct schedule
    Scheduler *scheduler = new Scheduler(A, true, false, Schedule::ALG_SCHEDULE::P2P_RULE, num_thread);
    scheduler->preprocessing(); // 进行预处理操作

    // step2: MKL SpMV preprocessing
    sparse_matrix_t matrixA;
    struct matrix_descr descA;
    sparse_status_t retA = mkl_sparse_d_create_csr(&matrixA, SPARSE_INDEX_BASE_ZERO, A->m, A->n,
                                                   A->rowptr, A->rowptr + 1, A->colidx, A->values);
    descA.diag = SPARSE_DIAG_NON_UNIT;
    descA.mode = SPARSE_FILL_MODE_FULL; // FULL
    descA.type = SPARSE_MATRIX_TYPE_GENERAL;
    retA = mkl_sparse_set_mv_hint(matrixA, SPARSE_OPERATION_NON_TRANSPOSE, descA, 200);
    CHECK_MKL_SPARSE(retA);
    retA = mkl_sparse_set_memory_hint(matrixA, SPARSE_MEMORY_AGGRESSIVE);
    CHECK_MKL_SPARSE(retA);
    retA = mkl_sparse_optimize(matrixA);
    CHECK_MKL_SPARSE(retA);

    pre_time += getCurrentTimeMilli();

    ic_time -= getCurrentTimeMilli();
    IC0_preconditioner(A, L, U, PCG_OPT, scheduler);
    ic_time += getCurrentTimeMilli();

    // printf("After IC0\n");

    // r: r - A * x
    spmv_time -= getCurrentTimeMilli();
    // dcsrmv(A->m, A->values, A->rowptr, A->colidx, x, r, -1, 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, matrixA, descA, x, 1.0, r); // r = -Ax + r
    spmv_time += getCurrentTimeMilli();
    double norm0 = norm(r, A->m);
    double rel_err = 1.0;

    double *z = MALLOC(double, A->m);
    double *y = MALLOC(double, A->m);
    CHECK_POINTER(z);
    CHECK_POINTER(y);

    // printf("update residual !\n");
    // fflush(stdout);
    // z = M\r, where M is preconditioner
    // lower_csr_trsv_serial(*L, y, r);
    // upper_csr_trsv_serial(*U, z, y);
    trsv_time -= getCurrentTimeMilli();
    scheduler->runForwardTRSVNoPerm(L, y, r);
    scheduler->runBackwardTRSVNoPerm(U, z, y);
    trsv_time += getCurrentTimeMilli();

    // bool check1 = fvectorEqual(y, yb, A->m, 1.0e-10);
    // check1 = fvectorEqual(z, zb, A->m, 1.0e-10);

    double *p = MALLOC(double, A->m);
    copyVector(p, z, A->m); // 搜索方向
    // double rz = dot(r, z, A->m);
    double rz = cblas_ddot(A->m, r, 1, z, 1);
    int k = 1;

    double *Ap = MALLOC(double, A->m);

    iter_time -= getCurrentTimeMilli();
    int stop_iteration = k;
    // iterate
    while (k <= maxiter)
    {
        double old_rz = rz;
        // step 1: spmv
        spmv_time -= getCurrentTimeMilli();
        // spmv_omp(*A, p, Ap); // Ap = A * p
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrixA, descA, p, 0.0, Ap); // Ap = 1.0 * A*p + 0 * Ap
        spmv_time += getCurrentTimeMilli();

        // step 2: 计算步长 alpha
        // double alpha = old_rz / dot(p, Ap, A->m); // 更新步长
        double alpha = old_rz / cblas_ddot(A->m, p, 1, Ap, 1); // 更新步长

        // step 3: 更新解
        // waxpby(A->m, x, 1, x, alpha, p);
        cblas_daxpby(A->m, alpha, p, 1, 1.0, x, 1);

        // step 4: 更新残差
        // waxpby(A->m, r, 1, r, -alpha, Ap); // r = r - alpha * A * Ap
        cblas_daxpby(A->m, -alpha, Ap, 1, 1.0, r, 1);
        // rel_err = norm(r, A->m) / norm0;
        rel_err = cblas_dnrm2(A->m, r, 1) / norm0;
        if (rel_err < tol)
        // break;
        {
            stop_iteration = k;
        }

        // step 5: 更新预条件残差
        // lower_csr_trsv_serial(*L, y, r);
        // upper_csr_trsv_serial(*U, z, y);
        trsv_time -= getCurrentTimeMilli();
        scheduler->runForwardTRSVNoPerm(L, y, r);
        scheduler->runBackwardTRSVNoPerm(U, z, y);
        trsv_time += getCurrentTimeMilli();

        // step 6: 更新搜索方向系数
        // rz = dot(r, z, A->m);
        rz = cblas_ddot(A->m, r, 1, z, 1);
        double beta = rz / old_rz; // 确保新的搜索方向和之间的搜索方向关于A共轭

        // step 7: 更新搜索方向
        // waxpby(A->m, p, 1, z, beta, p); // p = z + beta * p
        cblas_daxpby(A->m, 1.0, z, 1, beta, p, 1);
        ++k;
    }
    iter_time += getCurrentTimeMilli();
    total_time += getCurrentTimeMilli();

    // printf("iter = %d rel_err = %g\n", k, rel_err);

    double spmv_bytes = (k + 1) * (12. * A->getNnz() + (4. + 2 * 8) * A->m);
    double trsv_bytes = k * (12. * L->getNnz() + 12. * U->getNnz() + (8. + 2 * 4 + 2 * 2 * 8) * L->m);
    // printf("spmv_perf = %g gbps trsv_perf = %g gbps\n", spmv_bytes / spmv_time / 1e9, trsv_bytes / trsv_time / 1e9);
    // delete LU;

    // testMap["iterations"] = k;
    testMap["iterations"] = maxiter;
    testMap["pre_time"] = pre_time;
    testMap["ic_time"] = ic_time;
    testMap["trsv_time"] = trsv_time;
    testMap["spmv_time"] = spmv_time;
    testMap["iter_time"] = iter_time;
    testMap["total_time"] = total_time;

    delete L;
    delete U;

    FREE(r);
    FREE(z);
    FREE(y);
    FREE(p);
    FREE(Ap);
    mkl_sparse_destroy(matrixA);
    printf("------------the optimized PCG end !-----------------\n");
}

void pcg_mkl(const CSR *A, const double *b, double *x, const int maxiter, const double tol, std::map<std::string, double> &testMap)
{
    // mkl_d_ic

    printf("------------the MKL PCG !-----------------\n");

    double pre_time = 0, ic_time = 0, trsv_time = 0, spmv_time = 0, iter_time = 0, total_time = 0;

    double *r = MALLOC(double, A->m);
    copyVector(r, b, A->m);
    CSR *L, *U;

    total_time -= getCurrentTimeMilli();

    ic_time -= getCurrentTimeMilli();
    IC0_preconditioner(A, L, U, PCG_REF, nullptr);
    ic_time += getCurrentTimeMilli();

    pre_time -= getCurrentTimeMilli();
    // preprocessing for trsv and spmv
    sparse_matrix_t matrixL, matrixU, matrixA;
    struct matrix_descr descL, descU, descA;
    sparse_status_t retL = mkl_sparse_d_create_csr(&matrixL, SPARSE_INDEX_BASE_ZERO, L->m, L->n,
                                                   L->rowptr, L->rowptr + 1, L->colidx, L->values);
    sparse_status_t retU = mkl_sparse_d_create_csr(&matrixU, SPARSE_INDEX_BASE_ZERO, U->m, U->n,
                                                   U->rowptr, U->rowptr + 1, U->colidx, U->values);
    sparse_status_t retA = mkl_sparse_d_create_csr(&matrixA, SPARSE_INDEX_BASE_ZERO, A->m, A->n,
                                                   A->rowptr, A->rowptr + 1, A->colidx, A->values);
    descL.diag = SPARSE_DIAG_NON_UNIT;
    descL.mode = SPARSE_FILL_MODE_LOWER; // 下三角
    descL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descU.diag = SPARSE_DIAG_NON_UNIT;
    descU.mode = SPARSE_FILL_MODE_UPPER; // 上三角
    descU.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descA.diag = SPARSE_DIAG_NON_UNIT;
    descA.mode = SPARSE_FILL_MODE_FULL; // FULL
    descA.type = SPARSE_MATRIX_TYPE_GENERAL;

    retL = mkl_sparse_set_sv_hint(matrixL, SPARSE_OPERATION_NON_TRANSPOSE, descL, 200);
    retU = mkl_sparse_set_sv_hint(matrixU, SPARSE_OPERATION_NON_TRANSPOSE, descU, 200);
    retL = mkl_sparse_set_memory_hint(matrixL, SPARSE_MEMORY_AGGRESSIVE);
    retU = mkl_sparse_set_memory_hint(matrixU, SPARSE_MEMORY_AGGRESSIVE);
    retL = mkl_sparse_optimize(matrixL);
    retU = mkl_sparse_optimize(matrixU);

    retA = mkl_sparse_set_mv_hint(matrixA, SPARSE_OPERATION_NON_TRANSPOSE, descA, 200);
    retA = mkl_sparse_set_memory_hint(matrixA, SPARSE_MEMORY_AGGRESSIVE);
    retA = mkl_sparse_optimize(matrixA);
    pre_time += getCurrentTimeMilli();

    // r: r - A * x
    spmv_time -= getCurrentTimeMilli();
    // dcsrmv(A->m, A->values, A->rowptr, A->colidx, x, r, -1, 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, matrixA, descA, x, 1.0, r); // r = -Ax + r
    spmv_time += getCurrentTimeMilli();

    // double norm0 = norm(r, A->m);
    double norm0 = cblas_dnrm2(A->m, r, 1); // norm2
    double rel_err = 1.0;

    double *z = MALLOC(double, A->m);
    double *y = MALLOC(double, A->m);

    // z = M\r, where M is preconditioner
    trsv_time -= getCurrentTimeMilli();

    // lower_csr_trsv_serial(*L, y, r);
    // upper_csr_trsv_serial(*U, z, y);
    retL = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrixL, descL, r, y); // Ly = r
    retU = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrixU, descU, y, z); // Uz = y
    trsv_time += getCurrentTimeMilli();

    double *p = MALLOC(double, A->m);
    copyVector(p, z, A->m); // 搜索方向
    // double rz = dot(r, z, A->m);
    double rz = cblas_ddot(A->m, r, 1, z, 1);
    int k = 1;

    double *Ap = MALLOC(double, A->m);
    int stop_iteration = k;

    // iterate
    iter_time -= getCurrentTimeMilli();
    while (k <= maxiter)
    {
        double old_rz = rz;
        // step 1: spmv
        spmv_time -= getCurrentTimeMilli();
        // spmv_omp(*A, p, Ap); // Ap = A * p
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrixA, descA, p, 0.0, Ap); // Ap = 1.0 * A*p + 0 * Ap
        spmv_time += getCurrentTimeMilli();

        // step 2: 计算步长 alpha
        // double alpha = old_rz / dot(p, Ap, A->m); // 更新步长
        double alpha = old_rz / cblas_ddot(A->m, p, 1, Ap, 1); // 更新步长

        // step 3: 更新解
        // waxpby(A->m, x, 1, x, alpha, p); // x = x + alpha * p
        cblas_daxpby(A->m, alpha, p, 1, 1.0, x, 1);

        // step 4: 更新残差
        // waxpby(A->m, r, 1, r, -alpha, Ap); // r = r - alpha * Ap
        cblas_daxpby(A->m, -alpha, Ap, 1, 1.0, r, 1);
        // rel_err = norm(r, A->m) / norm0;
        rel_err = cblas_dnrm2(A->m, r, 1) / norm0;
        if (rel_err < tol)
        // break;
        {
            stop_iteration = k;
        }

        // step 5: 更新预条件残差
        trsv_time -= getCurrentTimeMilli();
        // lower_csr_trsv_serial(*L, y, r);
        // upper_csr_trsv_serial(*U, z, y);
        retL = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrixL, descL, r, y); // Ly = r
        retU = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrixU, descU, y, z); // Uz = y
        trsv_time += getCurrentTimeMilli();

        // step 6: 更新搜索方向系数
        // rz = dot(r, z, A->m);
        rz = cblas_ddot(A->m, r, 1, z, 1);
        double beta = rz / old_rz; // 确保新的搜索方向和之间的搜索方向关于A共轭

        // step 7: 更新搜索方向
        // waxpby(A->m, p, 1, z, beta, p); // p = z + beta * p
        cblas_daxpby(A->m, 1.0, z, 1, beta, p, 1);
        ++k;
    }
    iter_time += getCurrentTimeMilli();

    total_time += getCurrentTimeMilli();

    // printf("iter = %d rel_err = %g\n", k, rel_err);

    double spmv_bytes = (k + 1) * (12. * A->getNnz() + (4. + 2 * 8) * A->m);
    double trsv_bytes = k * (12. * L->getNnz() + 12. * U->getNnz() + (8. + 2 * 4 + 2 * 2 * 8) * L->m);
    // printf("spmv_perf = %g gbps trsv_perf = %g gbps\n", spmv_bytes / spmv_time / 1e9, trsv_bytes / trsv_time / 1e9);
    // delete LU;

    // testMap["iterations"] = k;
    testMap["iterations"] = maxiter;
    testMap["pre_time"] = pre_time;
    testMap["ic_time"] = ic_time;
    testMap["trsv_time"] = trsv_time;
    testMap["spmv_time"] = spmv_time;
    testMap["iter_time"] = iter_time;
    testMap["total_time"] = total_time;

    delete L;
    delete U;

    FREE(r);
    FREE(z);
    FREE(y);
    FREE(p);
    FREE(Ap);
    mkl_sparse_destroy(matrixL);
    mkl_sparse_destroy(matrixU);
    mkl_sparse_destroy(matrixA);
    printf("------------the MKL PCG  end !-----------------\n");
}

void pcg_ref(const CSR *A, const double *b, double *x, const int maxiter, const double tol, std::map<std::string, double> &testMap)
{
    printf("------------the OMP PCG !-----------------\n");

    double pre_time = 0, ic_time = 0, trsv_time = 0, spmv_time = 0, iter_time = 0, total_time = 0;

    total_time -= getCurrentTimeMilli();

    double *r = MALLOC(double, A->m);
    copyVector(r, b, A->m);
    CSR *L, *U;

    ic_time -= getCurrentTimeMilli();
    IC0_preconditioner(A, L, U, PCG_REF, nullptr);
    ic_time += getCurrentTimeMilli();

    // r: r - A * x
    spmv_time -= getCurrentTimeMilli();
    dcsrmv(A->m, A->values, A->rowptr, A->colidx, x, r, -1, 1);
    spmv_time += getCurrentTimeMilli();

    double norm0 = norm(r, A->m);
    double rel_err = 1.0;

    double *z = MALLOC(double, A->m);
    double *y = MALLOC(double, A->m);

    // z = M\r, where M is preconditioner
    trsv_time -= getCurrentTimeMilli();
    lower_csr_trsv_serial(*L, y, r);
    upper_csr_trsv_serial(*U, z, y);
    trsv_time += getCurrentTimeMilli();

    double *p = MALLOC(double, A->m);
    copyVector(p, z, A->m); // 搜索方向
    double rz = dot(r, z, A->m);
    int k = 1;

    double *Ap = MALLOC(double, A->m);
    int stop_iteration = k;

    // iterate
    iter_time -= getCurrentTimeMilli();
    while (k <= maxiter)
    {
        double old_rz = rz;
        // step 1: spmv
        spmv_time -= getCurrentTimeMilli();
        spmv_omp(*A, p, Ap); // Ap = A * p
        spmv_time += getCurrentTimeMilli();

        // step 2: 计算步长 alpha
        double alpha = old_rz / dot(p, Ap, A->m); // 更新步长

        // step 3: 更新解
        waxpby(A->m, x, 1, x, alpha, p);

        // step 4: 更新残差
        waxpby(A->m, r, 1, r, -alpha, Ap); // r = r - alpha * A * Ap
        rel_err = norm(r, A->m) / norm0;
        if (rel_err < tol)
        // break;
        {
            stop_iteration = k;
        }

        // step 5: 更新预条件残差
        trsv_time -= getCurrentTimeMilli();
        lower_csr_trsv_serial(*L, y, r);
        upper_csr_trsv_serial(*U, z, y);
        trsv_time += getCurrentTimeMilli();

        // step 6: 更新搜索方向系数
        rz = dot(r, z, A->m);
        double beta = rz / old_rz; // 确保新的搜索方向和之间的搜索方向关于A共轭

        // step 7: 更新搜索方向
        waxpby(A->m, p, 1, z, beta, p); // p = z + beta * p
        ++k;
    }
    iter_time += getCurrentTimeMilli();

    total_time += getCurrentTimeMilli();

    // printf("iter = %d rel_err = %g\n", k, rel_err);

    double spmv_bytes = (k + 1) * (12. * A->getNnz() + (4. + 2 * 8) * A->m);
    double trsv_bytes = k * (12. * L->getNnz() + 12. * U->getNnz() + (8. + 2 * 4 + 2 * 2 * 8) * L->m);
    // printf("spmv_perf = %g gbps trsv_perf = %g gbps\n", spmv_bytes / spmv_time / 1e9, trsv_bytes / trsv_time / 1e9);
    // delete LU;

    // testMap["iterations"] = k;
    testMap["iterations"] = maxiter;
    testMap["pre_time"] = pre_time;
    testMap["ic_time"] = ic_time;
    testMap["trsv_time"] = trsv_time;
    testMap["spmv_time"] = spmv_time;
    testMap["iter_time"] = iter_time;
    testMap["total_time"] = total_time;

    delete L;
    delete U;

    FREE(r);
    FREE(z);
    FREE(y);
    FREE(p);
    FREE(Ap);
    printf("------------the OMP PCG  end !-----------------\n");
}

void IC0_preconditioner(const CSR *A, CSR *&L, CSR *&U, PCG_TYPE type, Scheduler *scheduler)
{
    // A->make0BasedIndexing();
    int m = A->m;
    int n = A->n;
    int nnz = A->getNnz();
    int nnzL, nnzU;
    // SPD matrix
    nnzL = nnzU = (nnz + m) / 2;

    L = new CSR(m, n, nnzL);
    U = new CSR(m, n, nnzU);
#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        L->rowptr[i + 1] = A->diagptr[i] - A->rowptr[i] + 1; // contain diagonal elements
        U->rowptr[i + 1] = A->rowptr[i + 1] - A->diagptr[i];
    }

    for (int i = 0; i < m; i++)
    {
        L->rowptr[i + 1] += L->rowptr[i];
        U->rowptr[i + 1] += U->rowptr[i];
    }

#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        std::copy(A->colidx + A->rowptr[i], A->colidx + A->diagptr[i] + 1, L->colidx + L->rowptr[i]);
        std::copy(A->colidx + A->diagptr[i], A->colidx + A->rowptr[i + 1], U->colidx + U->rowptr[i]);
    }

    // 进行分解程序
    double *lu = MALLOC(double, nnz);
    // copyVector(lu, A->values, nnz);
    std::copy(A->values, A->values + nnz, lu);
    for (int i = 0; i < nnz; i++)
    {
        assert(std::isnan(lu[i]) == false);
        assert(std::isinf(lu[i]) == false);
    }

    switch (type)
    {
    case PCG_REF:
        // printf("run IC0 Serial!\n");
        spic0_csr_uL_serial(A, lu);
        // printf("run IC0 Serial!\n");
        fflush(stdout);
        break;
    case PCG_OPT:
        scheduler->runIC0(A, lu);
        break;
    case PCG_PERM:
        scheduler->runIC0Perm(A, lu);
    case PCG_OMP:
        spic0_csr_uL_levelset(A, lu);
        break;;

    default:
        break;
    }

    // spic0_csr_uL_levelset(A, lu);

    // 填充LU
    int *U_rowptr_alias = MALLOC(int, m + 1);
    std::copy(U->rowptr, U->rowptr + n + 1, U_rowptr_alias);

    for (int i = 0; i < m; i++)
    {
        assert(lu[A->diagptr[i]] > 0);
        for (int j = A->rowptr[i]; j <= A->diagptr[i]; j++)
        {
            double val = lu[j];

            int L_offset = L->rowptr[i] + j - A->rowptr[i];
            int U_offset = U_rowptr_alias[A->colidx[j]];
            L->values[L_offset] = val;
            U->values[U_offset] = val;
            U_rowptr_alias[A->colidx[j]]++;
        }
    }

#ifndef NDEBUG
    CSR *UT = U->transpose();
    bool isEqual = UT->equals(*L, true);

    printf("%s\n", isEqual ? "the transposed U matrix is equal to L matrix!" : "the transposed U matrix is not equal to L matrix!");
    fflush(stdout);
    assert(isEqual);

    delete UT;
#endif

    FREE(lu);
    FREE(U_rowptr_alias);
}
