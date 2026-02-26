#pragma once



#include <chrono>
#include<string>
#include <stdexcept>


class TimeMeasure
{
private:
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> end_time;
    std::chrono::duration<double, std::milli> duration;
    
    bool is_measuring = false;
    bool measure_done = false;
public:
    enum class TimeMetric
    {
        SECONDS,
        MILLISECONDS,
        MICROSECONDS
    };
    double elasped_time;
    TimeMeasure(/* args */);
    ~TimeMeasure();

    void start_timer();
    void measure_elasped_time();
    double getTime(TimeMetric tm = TimeMetric::MILLISECONDS) const;
    std::string getTimeStr(TimeMetric tm = TimeMetric::MILLISECONDS) const;
};
