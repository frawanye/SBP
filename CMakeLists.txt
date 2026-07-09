cmake_minimum_required(VERSION 3.10)

option(DEBUG "If ON, executable will try to diagnose where failures occur." OFF)
option(SCOREP "If ON, will add score-p instrumentation to the executable." OFF)
option(VALGRIND "If ON, will not link against -fsanitize=address." OFF)
option(RPROFILE "If ON, will compile with -Xopenmp-target=amdgcn-amd-amdhsa -Rpass-analysis=kernel-resource-usage for register profiling." OFF)

# set compilers
if (SCOREP)
    set(CMAKE_C_COMPILER scorep-mpicc)
    set(CMAKE_CXX_COMPILER scorep-mpicxx)
else (SCOREP)
    set(CMAKE_C_COMPILER mpicc)
    set(CMAKE_CXX_COMPILER mpic++)
endif (SCOREP)

# set the project name
project(DistributedSBP VERSION 2022.02.04)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(extern/hungarian)
add_subdirectory(extern/abseil-cpp)
add_subdirectory(extern/robin-map)

set(INCLUDE_DIRS
        include
        include/blockmodel
        include/matrix
        include/distributed
        extern/tclap-1.2.2/include
        extern/json/include
        extern/pcg-cpp/include
        patches
        extern/hungarian
        extern/abseil-cpp
        extern/robin-map/include
        )

# add the executable
add_library(SBPLIB STATIC
        src/graph.cpp
        src/utils.cpp
        src/sbp.cpp
        src/entropy.cpp
        src/entropy_gpu.cpp
        src/evaluate.cpp
        src/common.cpp
        src/finetune.cpp
        src/finetune_gpu.cpp
        src/globals.cpp
        src/block_merge.cpp
        src/partition.cpp
        src/sample.cpp
        src/top_down_sbp.cpp
        src/blockmodel/blockmodel.cpp
        src/blockmodel/blockmodel_triplet.cpp
        src/matrix/dense_matrix.cpp
        src/matrix/dict_matrix.cpp
        src/matrix/dict_transpose_matrix.cpp
        src/distributed/dist_block_merge.cpp
        src/distributed/dist_blockmodel_triplet.cpp
        src/distributed/dist_common.cpp
        src/distributed/dist_top_down_sbp.cpp
        src/distributed/dist_finetune.cpp
        src/distributed/dist_sbp.cpp
        src/distributed/divide_and_conquer.cpp
        src/distributed/two_hop_blockmodel.cpp
        #    blockmodel/sparse/dist_dict_matrix.cpp
        src/fastlog.cpp
        src/rng.cpp
        src/fastlgamma.cpp
        src/spence.cpp)
target_include_directories(SBPLIB PUBLIC ${INCLUDE_DIRS})

add_executable(DistributedSBP src/main.cpp)
target_include_directories(DistributedSBP PUBLIC ${INCLUDE_DIRS})

add_executable(DivideAndConquerSBP src/DivideAndConquerSBP.cpp)
target_include_directories(DivideAndConquerSBP PUBLIC ${INCLUDE_DIRS})

add_executable(ladis src/LayerDistributedSBP.cpp)
target_include_directories(ladis PUBLIC ${INCLUDE_DIRS})

add_executable(TopDownSBP src/TopDownSBPMain.cpp)
target_include_directories(TopDownSBP PUBLIC ${INCLUDE_DIRS})

if (DEBUG AND VALGRIND)
    set(COMPILE_OPTS -g3 -static-libasan)
    message("Compiling with -g3 and -static-libasan. Valgrind cannot run with -fsanitize=address")
elseif (DEBUG)
    set(COMPILE_OPTS -g -fsanitize=address -fsanitize=undefined)
    message("Compiling with -g3 and -fsanitize=address. Valgrind will not work with these executables")
elseif (RPROFILE)
    set(COMPILE_OPTS -O3 -Xopenmp-target=amdgcn-amd-amdhsa -Rpass-analysis=kernel-resource-usage)
    message("Compiling with -O3 and -Xopenmp-target=amdgcn-amd-amdhsa -Rpass-analysis=kernel-resource-usage for register profiling.")
else()
    set(COMPILE_OPTS -O3)
    message("Compiling without debug options")
endif()

target_compile_options(SBPLIB PUBLIC -Wall -no-pie -fopenmp --offload-arch=gfx942:xnack+ "${COMPILE_OPTS}")
target_link_libraries(SBPLIB PUBLIC hungarian absl::flat_hash_map -fopenmp stdc++fs -fopenmp --offload-arch=gfx942:xnack+ "${COMPILE_OPTS}")
target_link_libraries(DistributedSBP PUBLIC SBPLIB absl::flat_hash_map "${COMPILE_OPTS}")
target_link_libraries(DivideAndConquerSBP PUBLIC SBPLIB absl::flat_hash_map "${COMPILE_OPTS}")
target_link_libraries(TopDownSBP PUBLIC SBPLIB absl::flat_hash_map "${COMPILE_OPTS}")
target_link_libraries(ladis PUBLIC SBPLIB absl::flat_hash_map "${COMPILE_OPTS}")

# Google Test Suite
enable_testing()
add_subdirectory(extern/googletest)
add_executable(Test
        test/blockmodel_test.cpp
        test/block_merge_test.cpp
        test/common_test.cpp
        test/dense_matrix_test.cpp
        test/dict_matrix_test.cpp
        test/dict_transpose_matrix_test.cpp
        test/entropy_test.cpp
        test/evaluate_test.cpp
        test/finetune_test.cpp
        test/test_main.cpp
        src/rng.cpp
        test/sample_test.cpp
        test/toy_example.hpp
        test/utils_test.cpp
        test/nonparametric_entropy_test.cpp)
target_include_directories(Test PUBLIC ${INCLUDE_DIRS})
include(GoogleTest)
target_link_libraries(Test SBPLIB gtest gmock hungarian absl::hash absl::flat_hash_map -fopenmp stdc++fs)
gtest_discover_tests(Test)
