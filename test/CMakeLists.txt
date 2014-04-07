#setup testframework
MESSAGE("------setting up testframework-------")

set(GTEST_DIR "${CMAKE_SOURCE_DIR}/../cuda-test/gridding3D/gtest-1.6.0")
MESSAGE("GTEST_DIR " ${GTEST_DIR})

include_directories("${GTEST_DIR}/include")

find_library(GTEST_LIB NAMES gtest libgtest HINTS "${GTEST_DIR}/mybuild" "${GTEST_DIR}/mybuild/Debug")
find_library(GTESTMAIN_LIB NAMES gtest_main libgtest_main HINTS "${GTEST_DIR}/mybuild" "${GTEST_DIR}/mybuild/Debug")

MESSAGE("GTEST_LIB = ${GTEST_LIB}")
MESSAGE("GTESTMAIN_LIB = ${GTESTMAIN_LIB}")

#gpu test sources
set(GPU_SOURCES gpu/gridding3DGPU_conv_tests.cpp
				gpu/gridding2DGPU_conv_tests.cpp
				gpu/gridding3DGPU_fft_tests.cpp
				gpu/gridding3DGPU_deapo_tests.cpp
				gpu/forward/gridding3DGPU_forward_tests.cpp)

#add CPU Tests
add_subdirectory(cpu)

cuda_add_executable(runGPUUnitTests ${GPU_SOURCES} ${GRIDDING_SOURCES}  ../inc/griddingFunctions.hpp ../inc/gridding_operator_factory.hpp ../inc/gridding_operator.hpp ../inc/gridding_kernels.hpp)
target_link_libraries(runGPUUnitTests ${GRID_LIB_NAME} ${GTEST_LIB} ${GTESTMAIN_LIB})

if(GEN_ATOMIC)
    cuda_add_executable(runGPUATMUnitTests ${GPU_SOURCES} ${GRIDDING_SOURCES} ../inc/griddingFunctions.hpp ../inc/gridding_operator_factory.hpp ../inc/gridding_operator.hpp ../inc/gridding_kernels.hpp)
	target_link_libraries(runGPUATMUnitTests ${GRID_LIB_ATM_NAME} ${GTEST_LIB} ${GTESTMAIN_LIB})
endif()
