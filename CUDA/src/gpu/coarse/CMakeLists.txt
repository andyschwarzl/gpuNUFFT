MESSAGE("Coarse GPU GpuNUFFT Library")
#add exe to build from gpuNUFFT

include_directories(${GPUNUFFT_INC_DIR})
													
cuda_include_directories(${GPUNUFFT_INC_DIR})

set(GPU_CU_SOURCES	${CMAKE_CURRENT_SOURCE_DIR}/coarse_gpuNUFFT.cu 
					#${CMAKE_CURRENT_SOURCE_DIR}/gpuNUFFT_kernels.cu
					#${CMAKE_CURRENT_SOURCE_DIR}/../std_gpuNUFFT_kernels.cu
									 )

if (NOT GEN_ATOMIC)
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    SET(CUDA_NVCC_FLAGS "-arch;sm_11;--ptxas-options=-v")
  else(CMAKE_BUILD_TYPE)
    SET(CUDA_NVCC_FLAGS "-arch;sm_11;")
  endif()
endif(NOT GEN_ATOMIC)

if(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_NAME} ${GPU_CU_SOURCES} ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE} ${GPUNUFFT_INC_DIR}/cufft_config.hpp)
else(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_NAME} ${GPU_CU_SOURCES} ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE} ${GPUNUFFT_INC_DIR}/cufft_config.hpp SHARED)
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(${GRID_LIB_NAME})
CUDA_ADD_CUBLAS_TO_TARGET(${GRID_LIB_NAME})
