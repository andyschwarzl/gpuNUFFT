#setup testframework
MESSAGE("------ generating cpu tests -------")

#cpu test sources
set(CPU_SOURCES gpuNUFFT_tests.cpp 
				gpuNUFFT_kernel_tests.cpp
				gpuNUFFT_precomputation_tests.cpp
				gpuNUFFT_operator_factory_tests.cpp
				../../src/gpuNUFFT_utils.cpp 
				../../src/cpu/gpuNUFFT_cpu.cpp)

include_directories(${CUDA_INCLUDE_DIRS})
#add source dir
add_executable(runUnitTests ${CPU_SOURCES} ${GPUNUFFT_SOURCES} ../../inc/gpuNUFFT_cpu.hpp ../../inc/gpuNUFFT_utils.hpp ../../inc/gpuNUFFT_operator_factory.hpp ../../inc/gpuNUFFT_operator.hpp ../../inc/gpuNUFFT_kernels.hpp)
target_link_libraries(runUnitTests ${GRID_LIB_ATM_NAME} ${GTEST_LIB} ${GTESTMAIN_LIB})
set_target_properties(runUnitTests PROPERTIES LINK_FLAGS -lpthread)
