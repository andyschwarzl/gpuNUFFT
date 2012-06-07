#setup testframework
MESSAGE("------setting up testframework-------")

set(GTEST_DIR "${CMAKE_SOURCE_DIR}/../gtest-1.6.0")
MESSAGE("GTEST_DIR " ${GTEST_DIR})

include_directories("${GTEST_DIR}/include")

find_library(GTEST_LIB NAMES gtest libgtest HINTS "${GTEST_DIR}/mybuild" "${GTEST_DIR}/mybuild/Debug")
find_library(GTESTMAIN_LIB NAMES gtest_main libgtest_main HINTS "${GTEST_DIR}/mybuild" "${GTEST_DIR}/mybuild/Debug")

MESSAGE("GTEST_LIB = ${GTEST_LIB}")
MESSAGE("GTESTMAIN_LIB = ${GTESTMAIN_LIB}")

#cpu test sources
set(CPU_SOURCES cpu/gridding3D_tests.cpp 
				cpu/gridding3D_kernel_tests.cpp 
				../src/griddingFunctions.cpp 
				../src/cpu/gridding3D_cpu.cpp)

#gpu test sources
set(GPU_SOURCES gpu/gridding3DGPU_conv_tests.cpp
				gpu/gridding3DGPU_fft_tests.cpp
				gpu/gridding3DGPU_deapo_tests.cpp
				gpu/forward/gridding3DGPU_forward_tests.cpp)

#add source dir
add_executable(runUnitTests ${CPU_SOURCES} ../inc/gridding_cpu.hpp)
target_link_libraries(runUnitTests ${GTEST_LIB} ${GTESTMAIN_LIB})
set_target_properties(runUnitTests PROPERTIES LINK_FLAGS -lpthread)

cuda_add_executable(runGPUUnitTests ${GPU_SOURCES} ../inc/gridding_gpu.hpp ../inc/gridding_kernels.hpp)
target_link_libraries(runGPUUnitTests ${GRID_LIB_NAME} ${GTEST_LIB} ${GTESTMAIN_LIB})

if(GEN_ATOMIC)
	cuda_add_executable(runGPUATMUnitTests ${GPU_SOURCES} ../inc/gridding_gpu.hpp ../inc/gridding_kernels.hpp)
	target_link_libraries(runGPUATMUnitTests ${GRID_LIB_ATM_NAME} ${GTEST_LIB} ${GTESTMAIN_LIB})
endif()