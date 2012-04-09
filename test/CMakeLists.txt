#setup testframework
MESSAGE("------setting up testframework-------")

set(GTEST_DIR "../../../gtest-1.6.0")
MESSAGE("GTEST_DIR " ${GTEST_DIR})

include_directories("${GTEST_DIR}/include")
link_directories("${GTEST_DIR}/mybuild")
#cuda_include_directories("${GTEST_DIR}/include")

#find_library(GTEST_LIB libgtest "${GTEST_DIR}/mybuild")
#find_library(GTESTMAIN_LIB libgtest_main "${GTEST_DIR}/mybuild")
#find_library(GRIDDING3D_LIB gridding3D "${CMAKE_SOURCE_DIR}/build/src/Debug")

#cpu test sources
set(CPU_SOURCES gridding3D_tests.cpp ../src/griddingFunctions.cpp ../src/gridding3D_cpu.cpp)

#gpu test sources
set(GPU_SOURCES gridding3DGPU_tests.cpp)

#add source dir
add_executable(runUnitTests ${CPU_SOURCES} ../inc/gridding_cpu.hpp)
target_link_libraries(runUnitTests gtest gtest_main)

cuda_add_executable(runGPUUnitTests ${GPU_SOURCES} ../inc/gridding_gpu.hpp)

target_link_libraries(runGPUUnitTests gridding3D gtest gtest_main)
