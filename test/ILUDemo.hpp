#pragma once 
#include <iostream>
#include <string>

#include "SPM.hpp"
#include "TimeMeasure.hpp"
#include "MathUtils.hpp"
using SPM::CSR;

namespace ILURT
{
    class ILUDemo
    {
    private:
        /* data */
    protected:
        TimeMeasure timeMeasure;
        // bool hasPerm = false;

        int n;
        int lowerNnz;
        int nnz;
        std::vector<int> permToOrig, origToPerm;    // corresponding to inverse permutation and permutation
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
        virtual void buildset(){};
        virtual TimeMeasure fused_code()=0;
        virtual void testing();

    public:
        ILUDemo(/* args */);
        ILUDemo(int n, int nnz, std::string algName);
        ILUDemo(CSR *csrA,  double *correct_lu, int n, int nnz, std::string algName);

        ~ILUDemo();

        TimeMeasure evaluate();
        double *getSolution(){return lu; }
        void set_num_test(int num_test) { this->num_test = num_test;}
        void set_num_thread(int num_thread) {this->num_thread = num_thread;}
        double getAnalysisTime() const {return analysis_time  ;}
        double getElaspedTime() const {return elasped_time;}
        std::string getAlgName() {return algName; }


    };

} // namespace ILURT

