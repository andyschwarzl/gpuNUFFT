#setup testframework
MESSAGE("------setting up testframework-------")

set(GTEST_DIR "../../../gtest-1.6.0")
MESSAGE("GTEST_DIR " ${GTEST_DIR})

include_directories("${GTEST_DIR}/include")
#cuda_include_directories("${GTEST_DIR}/include")

find_library(GTEST_LIB gtest "${GTEST_DIR}/mybuild/Debug")
find_library(GTESTMAIN_LIB gtest_main "${GTEST_DIR}/mybuild/Debug")
#find_library(GRIDDING3D_LIB gridding3D "${CMAKE_SOURCE_DIR}/build/src/Debug")

#cpu test sources
set(CPU_SOURCES gridding3D_tests.cpp ../src/griddingFunctions.cpp ../src/gridding3D_cpu.cpp)

#gpu test sources
set(GPU_SOURCES gridding3DGPU_tests.cpp)

#add source dir
add_executable(runUnitTests ${CPU_SOURCES} ../inc/gridding_cpu.hpp)
target_link_libraries(runUnitTests ${GTEST_LIB} ${GTESTMAIN_LIB})

cuda_add_executable(runGPUUnitTests ${GPU_SOURCES} ../inc/gridding_gpu.hpp)

target_link_libraries(runGPUUnitTests gridding3D ${GTEST_LIB} ${GTESTMAIN_LIB})
