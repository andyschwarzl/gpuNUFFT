#setup testframework
MESSAGE("------setting up testframework-------")

set(GTEST_DIR "${CMAKE_SOURCE_DIR}/../../gtest/googletest")
MESSAGE("GTEST_DIR " ${GTEST_DIR})

include_directories("${GTEST_DIR}/include")

find_library(GTEST_LIB NAMES gtest libgtest HINTS "${GTEST_DIR}/build" "${GTEST_DIR}/build/Debug")
find_library(GTESTMAIN_LIB NAMES gtest_main libgtest_main HINTS "${GTEST_DIR}/build" "${GTEST_DIR}/build/Debug")

MESSAGE("GTEST_LIB = ${GTEST_LIB}")
MESSAGE("GTESTMAIN_LIB = ${GTESTMAIN_LIB}")

#gpu test sources
set(GPU_SOURCES gpu/gpuNUFFT_conv_tests.cpp
				gpu/gpuNUFFT_2D_conv_tests.cpp
				gpu/gpuNUFFT_fft_tests.cpp
				gpu/gpuNUFFT_deapo_tests.cpp
				gpu/forward/gpuNUFFT_forward_tests.cpp)

#add CPU Tests
add_subdirectory(cpu)

cuda_add_executable(runGPUATMUnitTests ${GPU_SOURCES} ${GPUNUFFT_SOURCES} ../inc/gpuNUFFT_utils.hpp ../inc/gpuNUFFT_operator_factory.hpp ../inc/gpuNUFFT_operator.hpp ../inc/gpuNUFFT_kernels.hpp)
if(WIN32)
	target_link_libraries(runGPUATMUnitTests ${GRID_LIB_ATM_NAME} ${GTEST_LIB} ${GTESTMAIN_LIB})
else(WIN32)
	target_link_libraries(runGPUATMUnitTests pthread ${GRID_LIB_ATM_NAME} ${GTEST_LIB} ${GTESTMAIN_LIB})
endif(WIN32)
