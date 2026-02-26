
#include "FusionDemo.hpp"
#include "MarcoUtils.hpp"
#include "SPM.hpp"
#include "papiProfiling.hpp"

using namespace SPM;
// #include "MathUtils.hpp"

namespace SpTRSVRT
{
    FusionDemo::FusionDemo() : x(nullptr), y(nullptr), correct_x(nullptr)
    {
        num_test = 100;
    }

    FusionDemo::~FusionDemo()
    {
        FREE(x);
        FREE(y);
        // FREE(co)
    }

    FusionDemo::FusionDemo(int n, int nnz, std::string algName) : FusionDemo()
    {
        this->n = n;
        this->nnz = nnz;
        this->algName = algName;
        this->csrA_perm = nullptr;
        x = MALLOC(double, n);
        y = MALLOC(double, n);
        CHECK_POINTER(x);
        CHECK_POINTER(y);
    }

    FusionDemo::FusionDemo(CSR *csrA, double *correct_x, int n, int nnz, double alpha, std::string algName) : FusionDemo(n, nnz, algName)
    {
        this->csrA = csrA;
        this->correct_x = correct_x;
        this->alpha = alpha;
    }

    // void FusionDemo::setup()
    // {

    // }
    void FusionDemo::setup()
    {

        // std::fill_n(y, n, 1.0);
        int *rowptr = csrA->rowptr;
        // int *colidx = csrA->colidx;
        double *values = csrA->values;
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
            {
                sum += values[j] * this->alpha; // values[j]
            }
            y[i] = sum;
        }
        std::fill_n(x, n, 0);
    }
    void FusionDemo::testing()
    {
        if (correct_x != nullptr)
            if (fvectorEqual(x, correct_x, n, 1e-10) == false)
            {
                cout << algName + " code solutions != reference solutions.\n";
            }
            else
            {
                cout << algName + " code solutions check pass.\n";
            }
    }

    TimeMeasure FusionDemo::evaluate()
    {
        TimeMeasure avg;
        std::vector<TimeMeasure> time_array;
        TimeMeasure analsis;
        analsis.start_timer();
        buildset();
        analsis.measure_elasped_time();
        analysis_time = analsis.getTime();

        setup();
        std::vector<Profiler::CacheLevel> levels{Profiler::CacheLevel::CACHE_L1, Profiler::CacheLevel::CACHE_L2};

        double *y_perm;
        double *y_back = y; // 存储y指针，以便验证
        // CHECK_POINTER(y_back);

        if (hasPerm)
        {

            y_perm = MALLOC(double, n);
            CHECK_POINTER(y_perm);
            assert(permToOrig.size() == n);
            assert(isPerm(permToOrig.data(), n));
            #pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                y_perm[i] = y[permToOrig[i]];
            }
            y = y_perm;
        }

        

        // if (permToOrig.size() != 0 && permToOrig.size() == n) // perm b
        // {
        // }

        // Profiler::PAPI_global_init_omp(levels);
        for (int i = 0; i < num_test; i++)
        {
            TimeMeasure t1;
            t1 = fused_code();
            time_array.push_back(t1);
        }

        // printf("permToOrig size: %d\n", permToOrig.size());

        if(hasPerm)
        {
            
            // printf(" perm to original solutions x vector!\n");
            double *x_perm = MALLOC(double, n);
            CHECK_POINTER(x_perm);
#pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                x_perm[permToOrig[i]] = x[i];
            }
            FREE(x);
            x = x_perm;

            FREE(y_perm);
            y = y_back;
        }
//         if (permToOrig.size() != 0 && permToOrig.size() == n)
//         {
//             printf(" perm to original solutions x vector!\n");
//             double *x_perm = MALLOC(double, n);
//             CHECK_POINTER(x_perm);
// #pragma omp parallel for
//             for (int i = 0; i < n; i++)
//             {
//                 x_perm[permToOrig[i]] = x[i];
//             }
//             FREE(x);
//             x = x_perm;

//             FREE(y_perm);
//             y = y_back;
//         }
        // if(permToOrig.size() ==1)
        // {
        //     for(int i=0; i < n; i++)
        //     {
        //         printf("x[%d]: %lf\n",x[i]);
        //     }
        // }
        // Profiler::PAPIOMPCacheProfiler::destory();
        // Profiler::PAPIOMPCacheProfiler::report();
        // Profiler::PAPI_global_destory_omp();
        testing();

        // TimeMeasure avg
        for (int i = 0; i < num_test; i++)
        {
            avg.elasped_time += time_array[i].elasped_time;
        }
        avg.elasped_time /= num_test;
        this->elasped_time = avg.elasped_time;
        return avg;
    }

} // namespace SpTRSVRT
