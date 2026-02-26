#pragma once

#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <omp.h>
#include <mutex>

#include "papi.h"

namespace Profiler
{

    enum CacheLevel
    {
        CACHE_L1,
        CACHE_L2,
        CACHE_L3
    };

    void PAPI_global_init_omp(const std::vector<CacheLevel> &levels);
    void PAPI_global_destory_omp();

    class PAPIOMPCacheProfiler
    {
    private:
        int tid;
        static inline int evenSet = PAPI_NULL;
        static inline int evenNum = 0;
        static inline std::vector<long long> values;
        static inline std::vector<CacheLevel> levels;
        static inline std::map<CacheLevel, std::pair<int, int>> cacheEvents{
            {CACHE_L1, {PAPI_L1_DCA, PAPI_L1_DCM}},
            {CACHE_L2, {PAPI_L2_DCA, PAPI_L2_DCM}},
            {CACHE_L3, {PAPI_L3_DCA, PAPI_L3_DCM}},
        };

        static std::string cacheLevelName(CacheLevel l);
        static inline int isSetEvent = false;
        static inline bool isEnding = false;

    public:
        PAPIOMPCacheProfiler(/* args */);
        ~PAPIOMPCacheProfiler();

        static void setIsSetEvent(bool isSetEvent);
        static void resizeValue(const int size);
        // static void resizeLevels(const int size);
        static void setEventNum(const int num);
        static void setIsEnding(const bool flag);

        // 设置需要检测的cache level列表
        static void set(const std::vector<CacheLevel> &levels);
        static void create_eventset();
        static void destory();
        void start();
        void stop();
        static void report();
    };

    // class PAPICacheProfiling
    // {
    // private:
    //     /* data */
    // public:
    //     PAPICacheProfiling(/* args */);
    //     ~PAPICacheProfiling();
    //     void PAPIPrint();
    // };

} // namespace Profiler
