MESSAGE("Coarse GPU GpuNUFFT Library")
#add exe to build from gpuNUFFT

include_directories(${GPUNUFFT_INC_DIR})
													
cuda_include_directories(${GPUNUFFT_INC_DIR})

set(GPU_CU_SOURCES	${CMAKE_CURRENT_SOURCE_DIR}/coarse_gpuNUFFT.cu 
					#${CMAKE_CURRENT_SOURCE_DIR}/gpuNUFFT_kernels.cu
					#${CMAKE_CURRENT_SOURCE_DIR}/../std_gpuNUFFT_kernels.cu
									 )

if(GPU_DOUBLE_PREC)
  SET(CUDA_NVCC_FLAGS "-arch;sm_20;--ptxas-options=-v")
else(GPU_DOUBLE_PREC)
  if (NOT GEN_ATOMIC)
    SET(CUDA_NVCC_FLAGS "-arch;sm_11;--ptxas-options=-v")
  endif(GEN_ATOMIC)
endif(GPU_DOUBLE_PREC)

if(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_NAME} ${GPU_CU_SOURCES} ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE} ${GPUNUFFT_INC_DIR}/cufft_config.hpp)
else(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_NAME} ${GPU_CU_SOURCES} ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE} ${GPUNUFFT_INC_DIR}/cufft_config.hpp SHARED)
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(${GRID_LIB_NAME})
CUDA_ADD_CUBLAS_TO_TARGET(${GRID_LIB_NAME})
