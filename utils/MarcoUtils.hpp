#pragma once

#include <cstdlib>
#include <cstdio>

// // 定义检测指针的宏
// #define CHECK_POINTER(ptr) \
//     do { \
//         if ((ptr) == nullptr) { \
//             std::cerr << "Error: Null pointer detected in " << __FILE__ \
//                       << " at line " << __LINE__ << std::endl; \
//             std::exit(EXIT_FAILURE); \
//         } \
//     } while (0)

// 定义检测指针的宏
#ifndef CHECK_POINTER
#ifdef NDEBUG
#define CHECK_POINTER(ptr) ((void)0)
#else
#define CHECK_POINTER(ptr)                                                     \
    do                                                                         \
    {                                                                          \
        if ((ptr) == NULL)                                                     \
        {                                                                      \
            fprintf(stderr, "Error: Null pointer detected in %s at line %d\n", \
                    __FILE__, __LINE__);                                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0);
#endif
#endif

#ifndef MALLOC
#define MALLOC(type, count) ((type *)malloc((count) * sizeof(type)))
#endif

#ifndef FREE
#define FREE(x)      \
    {                \
        if (x)       \
            free(x); \
        x = NULL;    \
    }
#endif
