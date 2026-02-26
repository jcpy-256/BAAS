
#pragma once

#include<string>
#include<iostream>

using namespace std;
#include "TimeMeasure.hpp"
#include "SPM.hpp"
using namespace SPM;

namespace SpTRSVRT
{
    class FusionDemo
    {
    protected:
        TimeMeasure timeMeasure;
        bool hasPerm = false;
        
        int n;
        int nnz;
        std::vector<int> permToOrig, origToPerm;    // corrsponding to inverse permutation and permutation
        double *y;
        double *x;
        double *correct_x;  // remote memory only read
        double alpha = 1.0;
        std::string algName;
        int num_test;
        int num_thread;
        double elasped_time = 0;
        double analysis_time = 0;

        CSR *csrA;  // remote memory only read
        CSR *csrA_perm; 

        virtual void setup();
        virtual void buildset(){};    // preprocessing
        // virtual void evaluate();    // running 
        virtual TimeMeasure fused_code() = 0;
        virtual void testing();


    public:
        FusionDemo(/* args */);
        FusionDemo(int n, int nnz, std::string algName);
        FusionDemo(CSR *csrA, double *correct_x, int n, int nnz, double alpha, std::string algName);
        // FusionDemo(/* args */);
        ~FusionDemo();

        TimeMeasure evaluate();
        double *getSolution() {return x;};
        void set_num_test(int num_test) {this->num_test = num_test;}
        void set_num_thread(int num_thread) {this->num_thread = num_thread;}
        double getAnalysisTime() const {return analysis_time;}
        double getElaspedTime() const {return elasped_time;}
        std::string getName() {return algName; }

    };



} // namespace SpTRSVRT
