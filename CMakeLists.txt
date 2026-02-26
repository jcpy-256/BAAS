
cmake_minimum_required(VERSION 3.16.0)
project(SpTRSV_Merge VERSION 0.0.1  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 
set(CMAKE_CXX_EXTENSIONS OFF)        # 

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
add_compile_options(-w -mavx2)

# Figure out SSE level support
message(STATUS "Seaching for SSE...")
find_package(SSE)

# Figure out AVX level support
message(STATUS "Searching for AVX...")
find_package(AVX)

# Figure out FMA level support
message(STATUS "Searching for FMA...")
find_package(FMA)

# Figure out FMA level support
message(STATUS "Searching for FMA...")
find_package(FMA)



add_subdirectory("./spm/")
add_subdirectory("./merge/")
add_subdirectory("./blas")
add_subdirectory("./profile")
add_subdirectory("./test/")

