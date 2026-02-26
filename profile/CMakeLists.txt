file(GLOB SRC_LIB_FILES 
        "./*.cpp")

message(STATUS "${SRC_LIB_FILES}")

set(tar profileLib)
add_library(${tar} SHARED ${SRC_LIB_FILES})

target_link_libraries(${tar} PUBLIC pthread m dl)
find_package(OpenMP REQUIRED)
target_link_libraries(${tar} PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(${tar} PUBLIC "./include/")
target_include_directories(${tar} PUBLIC "../utils/")

set(PAPI_INCLUDE "$ENV{PAPI_HOME}/include/")
set(PAPI_LIB "$ENV{PAPI_HOME}/lib/")
target_link_directories(${tar} PUBLIC ${PAPI_LIB})
target_include_directories(${tar} PUBLIC ${PAPI_INCLUDE})
target_link_libraries(${tar} PUBLIC papi)

string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
target_compile_options(${tar} PUBLIC ${SIMD_FLAGS})

target_compile_features(${tar} PUBLIC cxx_std_17)