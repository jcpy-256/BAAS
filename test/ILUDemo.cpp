#include "ILUDemo.hpp"
#include "MarcoUtils.hpp"

using namespace SPM;

namespace ILURT
{
    ILUDemo::ILUDemo() : lu(nullptr), correct_lu(nullptr), csrA(nullptr)
    {
        this->num_test = 10;
        this->num_thread = 1;
        this->n = 0;
        this->lowerNnz = 0;
        this->nnz = 0;
    };

    ILUDemo::~ILUDemo()
    {
        FREE(lu);
    }

    ILUDemo::ILUDemo(int n, int nnz, std::string algName)
    {
        this->n = n;
        this->nnz = nnz;
        this->algName = algName;
        lu = MALLOC(double, nnz);
        CHECK_POINTER(lu);
    }
    ILUDemo::ILUDemo(CSR *csrA, double *correct_lu, int n, int nnz, std::string algName) : ILUDemo(n, nnz, algName)
    {
        this->csrA = csrA;
        this->correct_lu = correct_lu;
    }

    void ILUDemo::setup()
    {
    }

    void ILUDemo::testing()
    {
        if (correct_lu != nullptr)
        {
            std::cout<<"=============== result check ===================\n";
            if (fvectorEqual(correct_lu, lu, nnz, 1.0e-10) == false)
            {
                cout << algName + " code ILU0 != reference factorization.\n";
            }
            else
            {
                cout << algName + " code ILU0 check pass.\n";
            }
        }
        fflush(stdout);
    }

    TimeMeasure ILUDemo::evaluate()
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

} // namespace ILURT