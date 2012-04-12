#setup testframework
MESSAGE("------setting up testframework-------")

set(GTEST_DIR ${PROJECT_SOURCE_DIR}/../gtest-1.6.0)
MESSAGE("GTEST_DIR " ${GTEST_DIR})

include_directories("${GTEST_DIR}/include")
find_library(GTEST_LIB NAMES gtest libgtest HINTS "${GTEST_DIR}/mybuild" "${GTEST_DIR}/mybuild/Debug")
find_library(GTESTMAIN_LIB NAMES gtest_main libgtest_main HINTS "${GTEST_DIR}/mybuild" "${GTEST_DIR}/mybuild/Debug")
MESSAGE("GTEST_LIB = ${GTEST_LIB}")
MESSAGE("GTESTMAIN_LIB = ${GTESTMAIN_LIB}")

#cpu test sources
set(CPU_SOURCES gridding3D_tests.cpp ../src/griddingFunctions.cpp ../src/gridding3D_cpu.cpp)

#gpu test sources
set(GPU_SOURCES gridding3DGPU_tests.cpp)

#add source dir
add_executable(runUnitTests ${CPU_SOURCES} ../inc/gridding_cpu.hpp)
target_link_libraries(runUnitTests ${GTEST_LIB} ${GTESTMAIN_LIB})
set_target_properties(runUnitTests PROPERTIES LINK_FLAGS -lpthread)
#target_link_libraries(runUnitTests ${GTEST_LIB} gtest gtest_main)

cuda_add_executable(runGPUUnitTests ${GPU_SOURCES} ../inc/gridding_gpu.hpp)

target_link_libraries(runGPUUnitTests gridding3D ${GTEST_LIB} ${GTESTMAIN_LIB})
