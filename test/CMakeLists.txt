
set(tar SpTRSV_Level_Merge)

file(GLOB SOURCES "*.cpp") 
message(STATUS "${SOURCES}")
add_executable(${tar} "FusionDemo.cpp" "SpTRSV_level_merge.cpp")

target_link_libraries(${tar} PUBLIC merge spm)
target_link_libraries(${tar} PUBLIC profileLib)

set(MKL_INCLUDE "$ENV{METIS_HOME}/include/")
set(MKL_LIB "$ENV{METIS_HOME}/lib/")

target_link_directories(${tar} PUBLIC ${MKL_LIB})
target_include_directories(${tar} PUBLIC ${MKL_INCLUDE})
target_link_libraries(${tar} PUBLIC mkl_rt)

find_package(OpenMP REQUIRED)
target_link_libraries(${tar} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${tar} PUBLIC pthread m dl)
# target_link_libraries(${tar} PUBLIC m dl)

target_include_directories(${tar} PUBLIC "../test/")
target_include_directories(${tar} PUBLIC "../spm/include/")
target_include_directories(${tar} PUBLIC "../merge/include/")
target_include_directories(${tar} PUBLIC "../profile/include/")

string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
target_compile_options(${tar} PUBLIC ${SIMD_FLAGS})

target_compile_features(${tar} PUBLIC cxx_std_17)


###################### ILU0 #########################
set(ilu ILU_Level_Merge)

add_executable(${ilu} "ILUDemo.cpp" "ilu0_level_merge.cpp")

target_link_libraries(${ilu} PUBLIC merge spm)
target_link_libraries(${ilu} PUBLIC profileLib)

set(MKL_INCLUDE "$ENV{MKL_HOME}/include/")
set(MKL_LIB "$ENV{MKL_HOME}/lib/intel64/")

target_link_directories(${ilu} PUBLIC ${MKL_LIB})
target_include_directories(${ilu} PUBLIC ${MKL_INCLUDE})
target_link_libraries(${ilu} PUBLIC mkl_rt)

find_package(OpenMP REQUIRED)
target_link_libraries(${ilu} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${ilu} PUBLIC pthread m dl)
# target_link_libraries(${tar} PUBLIC m dl)

target_include_directories(${ilu} PUBLIC "../test/")
target_include_directories(${ilu} PUBLIC "../spm/include/")
target_include_directories(${ilu} PUBLIC "../merge/include/")
target_include_directories(${ilu} PUBLIC "../profile/include/")

target_compile_options(${ilu} PUBLIC ${SIMD_FLAGS})

target_compile_features(${ilu} PUBLIC cxx_std_17)


###################### IC0 #########################
set(ic IC_Level_Merge)

add_executable(${ic} "ic0_level_merge.cpp")

target_link_libraries(${ic} PUBLIC merge spm)
target_link_libraries(${ic} PUBLIC profileLib)

set(MKL_INCLUDE "$ENV{MKL_HOME}/include/")
set(MKL_LIB "$ENV{MKL_HOME}/lib/intel64/")

target_link_directories(${ic} PUBLIC ${MKL_LIB})
target_include_directories(${ic} PUBLIC ${MKL_INCLUDE})
target_link_libraries(${ic} PUBLIC mkl_rt)

find_package(OpenMP REQUIRED)
target_link_libraries(${ic} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${ic} PUBLIC pthread m dl)
# target_link_libraries(${tar} PUBLIC m dl)

target_include_directories(${ic} PUBLIC "../test/")
target_include_directories(${ic} PUBLIC "../spm/include/")
target_include_directories(${ic} PUBLIC "../merge/include/")
target_include_directories(${ic} PUBLIC "../profile/include/")

target_compile_options(${ic} PUBLIC ${SIMD_FLAGS})

target_compile_features(${ic} PUBLIC cxx_std_17)

###################### ICCG #########################
set(pcg ICCG_Level_Merge)

add_executable(${pcg} "iccg_level_merge.cpp")

target_link_libraries(${pcg} PUBLIC myblas merge spm)
target_link_libraries(${pcg} PUBLIC profileLib)

set(MKL_INCLUDE "$ENV{MKL_HOME}/include/")
set(MKL_LIB "$ENV{MKL_HOME}/lib/intel64/")

target_link_directories(${pcg} PUBLIC ${MKL_LIB})
target_include_directories(${pcg} PUBLIC ${MKL_INCLUDE})
target_link_libraries(${pcg} PUBLIC mkl_rt)

find_package(OpenMP REQUIRED)
target_link_libraries(${pcg} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${pcg} PUBLIC pthread m dl)
# target_link_libraries(${tar} PUBLIC m dl)

target_include_directories(${pcg} PUBLIC "../test/")
target_include_directories(${pcg} PUBLIC "../spm/include/")
target_include_directories(${pcg} PUBLIC "../merge/include/")
target_include_directories(${pcg} PUBLIC "../profile/include/")

target_compile_options(${pcg} PUBLIC ${SIMD_FLAGS})

target_compile_features(${pcg} PUBLIC cxx_std_17)