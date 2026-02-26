#pragma once
#include <atomic>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h> // __mm_pause()
#define CPU_RELAX() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define CPU_RELAX() asm volatile("yield" ::: "memory")
#else
#define CPU_RELAX() // fallback: do nothing
#endif