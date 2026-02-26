#include <iostream>
#include<iomanip> 
#include <sstream>

#include "TimeMeasure.hpp"

TimeMeasure::TimeMeasure(/* args */)
{
    elasped_time = 0.0;
    measure_done = true;
    is_measuring = false;
}

TimeMeasure::~TimeMeasure()
{
}

void TimeMeasure::start_timer()
{
    start_time = std::chrono::steady_clock::now();
    is_measuring = true;
    measure_done = false;
}
void TimeMeasure::measure_elasped_time()
{
    if (!is_measuring)
    {
        throw std::runtime_error("Time measurement is not started. Call start_timer() first.");
    }
    end_time = std::chrono::steady_clock::now();
    duration = end_time - start_time;
    elasped_time = duration.count();
    is_measuring = false;
    measure_done = true;
}
double TimeMeasure::getTime(TimeMetric tm) const
{
    if (!measure_done)
    {
        throw std::runtime_error("No measurement available. Call measure() first.");
    }

    switch (tm)
    {
    case TimeMetric::SECONDS:
        // this->elasped_time = duration.count() ;
        return elasped_time * 1.0 / 1000;
    case TimeMetric::MILLISECONDS:
        return elasped_time;
    case TimeMetric::MICROSECONDS:
        return elasped_time * 1000.0;
    default:
        throw std::invalid_argument("Invalid time unit");
    }
}
std::string TimeMeasure::getTimeStr(TimeMetric tm) const
{
    double time = getTime(tm);
    std::string unitStr;
     std::ostringstream oss;
    
    // 控制输出格式：保留3位小数
    oss << std::fixed << std::setprecision(4) << time;
    std::string timestr = oss.str();

    switch (tm)
    {
    case TimeMetric::SECONDS:
        unitStr = "s";
        break;
    case TimeMetric::MILLISECONDS:
        unitStr = "ms";
        break;
    case TimeMetric::MICROSECONDS:
        unitStr = "μs";
        break;
    }

    return timestr + " " + unitStr;
}
