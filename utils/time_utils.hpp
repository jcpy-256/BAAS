#pragma once

#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>


static inline uint64_t getCurrentTimeMicro()
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return (uint64_t)(time.tv_sec * INT64_C(1000000) + time.tv_usec);
}

static inline double getCurrentTimeMilli()
{
    struct timeval time;

    gettimeofday(&time, NULL);
    return (double)(time.tv_sec * INT64_C(1000) + 1.0 * time.tv_usec / 1000);
}