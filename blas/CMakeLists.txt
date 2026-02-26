file(GLOB SRC_LIB_FILES 
        "./*.cpp")
message(STATUS "${SRC_LIB_FILES}")
set(SRC_LIST ${SRC_LIB_FILES})
add_library(myblas SHARED ${SRC_LIB_FILES})
target_link_libraries(myblas PUBLIC spm)
target_link_libraries(myblas PUBLIC pthread m dl)
find_package(OpenMP REQUIRED)
target_link_libraries(myblas PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(myblas PUBLIC "./include/")
target_include_directories(myblas PUBLIC "../utils/")

string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
target_compile_options(myblas PUBLIC ${SIMD_FLAGS})

target_compile_features(myblas PUBLIC cxx_std_17)