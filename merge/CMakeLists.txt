

set(MERGE_LIB merge)
file(GLOB SRC_LIB_FILES 
        "./*.cpp")
# add_library(${MERGE_LIB} SHARED ${SRC_LIB_FILES})
add_library(${MERGE_LIB} STATIC ${SRC_LIB_FILES})
target_link_libraries(${MERGE_LIB} PRIVATE pthread m dl)
find_package(OpenMP REQUIRED)
target_link_libraries(${MERGE_LIB} PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(${MERGE_LIB} PRIVATE spm profileLib)



target_include_directories(${MERGE_LIB} PUBLIC "./include/")
target_include_directories(${MERGE_LIB} PUBLIC "../spm/include/")
target_include_directories(${MERGE_LIB} PUBLIC "../profile/include/")
target_include_directories(${MERGE_LIB} PUBLIC "../utils/")


string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
target_compile_options(${MERGE_LIB} PUBLIC ${SIMD_FLAGS})

target_compile_features(${MERGE_LIB} PUBLIC cxx_std_17)
# target_include_directories()