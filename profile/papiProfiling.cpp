#include "papiProfiling.hpp"

#include <iostream>
#include <vector>
#include <papi.h>

namespace Profiler
{
    void PAPI_global_init_omp(const std::vector<CacheLevel> &levels)
    {
        
        if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        {
            std::cerr << "PAPI initialization error!" << std::endl;
            exit(1);
        }
        // PAPIOMPCacheProfiler::setIsSetEvent(false);
        PAPIOMPCacheProfiler::create_eventset(); // create eventset
        PAPIOMPCacheProfiler::set(levels);
        PAPIOMPCacheProfiler::setIsSetEvent(true);
        int threadNum = omp_get_max_threads();
        PAPIOMPCacheProfiler::setEventNum(levels.size());
        PAPIOMPCacheProfiler::resizeValue(threadNum * levels.size() * 2);
        // PAPIOMPCacheProfiler::resizeValue(levels.size());
        // PAPIOMPCacheProfiler
    }

    void PAPI_global_destory_omp()
    {
        PAPIOMPCacheProfiler::destory(); // clean eventset
        PAPI_shutdown();

        PAPIOMPCacheProfiler::setIsSetEvent(false);
    }

    void PAPIOMPCacheProfiler::report()
    {
        if (!isEnding)
        {
            // perror()
            fprintf(stderr, "The Profiling is not end. Please ensure that the profiling has been completed!\n");
            exit(EXIT_FAILURE);
        }
        else
        {
            int threadNum = omp_get_max_threads();
            int nEvent = levels.size();
            for (int tid = 0; tid < threadNum; tid++)
            {
                for (auto &l : levels)
                {
                    long long access = values[tid * nEvent * 2];
                    if (access == 0)
                        continue;
                    long long miss = values[tid * nEvent * 2 + 1];
                    double hitRatio = 1.0 - ((double)miss / access);
                    double missRatio = ((double)miss / access);
                    std::string LevelName = cacheLevelName(l);
                    printf("%s Access: %lld\n", LevelName.c_str(), access);
                    printf("%s Missed: %lld\n", LevelName.c_str(), miss);
                    printf("%s hitRatio: %lf\n", LevelName.c_str(), hitRatio);
                    printf("%s missRatio: %lf\n", LevelName.c_str(), missRatio);
                }
            }
        }
    }

    void PAPIOMPCacheProfiler::setIsSetEvent(bool flag)
    {
        isSetEvent = flag;
    }

    void PAPIOMPCacheProfiler::setEventNum(const int num)
    {
        evenNum = num;
    }

    void PAPIOMPCacheProfiler::setIsEnding(const bool flag)
    {
        isEnding = flag;
    }

    // void PAPIOMPCacheProfiler::resizeLevels(const int size)
    // {
    //     levels.resize(size);
    // }

    void PAPIOMPCacheProfiler::resizeValue(const int size)
    {
        values.resize(size, 0);
    }
    PAPIOMPCacheProfiler::PAPIOMPCacheProfiler(/* args */)
    {
        // 每个线程注册
        // PAPI_register_thread(); // PAPI <= 5.0
        // create_eventset();
        this->tid = omp_get_thread_num();
    }

    PAPIOMPCacheProfiler::~PAPIOMPCacheProfiler()
    {
        // if (evenSet != PAPI_NULL)
        // {
        //     PAPI_cleanup_eventset(evenSet);
        //     PAPI_destroy_eventset(&evenSet);
        // }
        // PAPI_unregister_thread(); // PAPI <= 5.0
    }

    void PAPIOMPCacheProfiler::destory()
    {
        if (evenSet != PAPI_NULL)
        {
            PAPI_cleanup_eventset(evenSet);
            PAPI_destroy_eventset(&evenSet);
        }
        PAPIOMPCacheProfiler::values.clear();
        PAPIOMPCacheProfiler::values.shrink_to_fit();
        PAPIOMPCacheProfiler::levels.clear();
        PAPIOMPCacheProfiler::levels.shrink_to_fit();
    }

    void PAPIOMPCacheProfiler::create_eventset()
    {
        if (PAPI_create_eventset(&evenSet) != PAPI_OK)
        {
            std::cerr << "PAPI_create_eventset failed in \n"
                      << __FILE__ << ":" << __LINE__ << std::endl;
            exit(1);
        }
    }
    std::string PAPIOMPCacheProfiler::cacheLevelName(CacheLevel l)
    {
        switch (l)
        {
        case CACHE_L1:
            return "L1";
        case CACHE_L2:
            return "L2";
        case CACHE_L3:
            return "L3";
        }
        return "";
    }

    void PAPIOMPCacheProfiler::set(const std::vector<CacheLevel> &cacheLevels)
    {
        for (auto l : cacheLevels)
        {
            if (cacheEvents.count(l))
            {
                if (PAPI_add_event(evenSet, cacheEvents[l].first) != PAPI_OK)
                {
                    
                    std::cerr << "PAPI add events error! This Event may be not supported." << std::endl;
                    exit(1);
                }

                if (PAPI_add_event(evenSet, cacheEvents[l].second) != PAPI_OK)
                {
                    std::cerr << "PAPI add events error! This Event may be not supported." << std::endl;
                    exit(1);
                }
                // PAPI_add_event(evenSet, cacheEvents[l].first);
                // PAPI_add_event(evenSet, cacheEvents[l].second);
            }
            levels.push_back(l);
        }
    }
    void PAPIOMPCacheProfiler::start()
    {
        // 开始测量
        if (PAPI_start(evenSet) != PAPI_OK)
        {
            fprintf(stderr, "PAPI_start failed in thread %d\n", tid);
            exit(EXIT_FAILURE);
        }
    }

    void PAPIOMPCacheProfiler::stop()
    {
        long long ret[2] = {0, 0};
        // 开始测量
        if (PAPI_stop(evenSet, ret) != PAPI_OK)
        {
            fprintf(stderr, "PAPI_stop failed in thread %d\n", tid);
            exit(EXIT_FAILURE);
        }
        values[tid * evenNum] += ret[0];
        values[tid * evenNum + 1] += ret[1];
    }

    enum CacheLevelBack
    {
        L1,
        L2,
        L3
    };

    class PAPICacheProfiler
    {
    private:
        int EventSet;
        std::vector<int> events;
        std::vector<long long> values;
        std::vector<CacheLevelBack> levels;
        bool running;

    public:
        PAPICacheProfiler() : EventSet(PAPI_NULL), running(false)
        {
            // 初始化 PAPI（只需一次）
            static bool papi_initialized = false;
            if (!papi_initialized)
            {
                if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
                {
                    std::cerr << "PAPI initialization error!" << std::endl;
                    exit(1);
                }
                papi_initialized = true;
            }

            if (PAPI_create_eventset(&EventSet) != PAPI_OK)
            {
                std::cerr << "PAPI create eventset error!" << std::endl;
                exit(1);
            }
        }

        // 设置要测试的缓存层级
        void set(const std::vector<CacheLevelBack> &chosen_levels)
        {
            levels = chosen_levels;
            events.clear();

            for (auto lvl : levels)
            {
                switch (lvl)
                {
                case L1:
                    events.push_back(PAPI_L1_DCM); // L1 misses
                    events.push_back(PAPI_L1_DCA); // L1 accesses
                    break;
                case L2:
                    events.push_back(PAPI_L2_DCM);
                    events.push_back(PAPI_L2_DCA);
                    break;
                case L3:
                    events.push_back(PAPI_L3_TCM);
                    events.push_back(PAPI_L3_TCA);
                    break;
                }
            }

            values.resize(events.size(), 0);

            if (PAPI_add_events(EventSet, events.data(), events.size()) != PAPI_OK)
            {
                std::cerr << "PAPI add events error! (可能是 CPU 不支持某些事件)" << std::endl;
                exit(1);
            }
        }

        void start()
        {
            if (running)
                return;
            if (PAPI_start(EventSet) != PAPI_OK)
            {
                std::cerr << "PAPI start error!" << std::endl;
                exit(1);
            }
            running = true;
        }

        void stop()
        {
            if (!running)
                return;
            if (PAPI_stop(EventSet, values.data()) != PAPI_OK)
            {
                std::cerr << "PAPI stop error!" << std::endl;
                exit(1);
            }
            running = false;
        }

        void report() const
        {
            std::cout << "=== Cache Profiling Report ===" << std::endl;

            int idx = 0;
            for (auto lvl : levels)
            {
                long long misses = values[idx];
                long long accesses = values[idx + 1];
                idx += 2;

                std::string name = (lvl == L1 ? "L1" : (lvl == L2 ? "L2" : "L3"));

                if (accesses > 0)
                {
                    double ratio = (double)misses / accesses;
                    std::cout << name << " Cache: "
                              << "Misses=" << misses
                              << ", Accesses=" << accesses
                              << ", Miss Ratio=" << ratio << std::endl;
                }
                else
                {
                    std::cout << name << " Cache: Not supported or no data" << std::endl;
                }
            }
        }

        ~PAPICacheProfiler()
        {
            if (running)
                stop();
            PAPI_cleanup_eventset(EventSet);
            PAPI_destroy_eventset(&EventSet);
        }
    };

    // PAPICacheProfiling::PAPICacheProfiling(/* args */)
    // {
    // }

    // PAPICacheProfiling::~PAPICacheProfiling()
    // {
    // }

    // void PAPICacheProfiling::PAPIPrint()
    // {
    //     int retval = PAPI_library_init(PAPI_VER_CURRENT);
    //     if (retval != PAPI_VER_CURRENT)
    //     {
    //         printf("PAPI library init error!\n");
    //     }

    //     printf("PAPI version: %d.%d.%d\n",
    //            PAPI_VERSION_MAJOR(retval),
    //            PAPI_VERSION_MINOR(retval),
    //            PAPI_VERSION_REVISION(retval));

    //     if (PAPI_is_initialized() == PAPI_NOT_INITED)
    //     {
    //         printf("PAPI not initialized\n");
    //     }

    //     printf("PAPI is working correctly!\n");
    //     PAPI_shutdown();
    // }

} // namespace Profiler
