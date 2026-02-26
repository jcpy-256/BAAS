

file(GLOB SRC_LIB_FILES 
        "./*.cpp")
message(STATUS "${SRC_LIB_FILES}")
set(SRC_LIST ${SRC_LIB_FILES})
add_library(spm SHARED ${SRC_LIB_FILES})
target_link_libraries(spm PUBLIC pthread m dl)
find_package(OpenMP REQUIRED)
target_link_libraries(spm PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(spm PUBLIC "./include/")
target_include_directories(spm PUBLIC "../utils/")

# Add SSE, AVX, and FMA flags to compiler flags
string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
target_compile_options(spm PUBLIC ${SIMD_FLAGS})

target_compile_features(spm PUBLIC cxx_std_17)