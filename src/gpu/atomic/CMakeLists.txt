MESSAGE("Atomic GPU Gridding Library")
#add exe to build from gridding3D

include_directories(${GRIDDING_INC_DIR})
													
cuda_include_directories(${GRIDDING_INC_DIR})

set(GPU_CU_SOURCES	${CMAKE_CURRENT_SOURCE_DIR}/atomic_gridding_kernels.cu 
					#${CMAKE_CURRENT_SOURCE_DIR}/../std_gridding_kernels.cu
									 )
SET(CUDA_NVCC_FLAGS "-arch;sm_20;--ptxas-options=-v")
									 
if(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_ATM_NAME} ${GPU_CU_SOURCES}  ${GPU_GRIDDING_SOURCES} ${GRIDDING_SOURCES} ${GRIDDING_INCLUDE})
else(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_ATM_NAME} ${GPU_CU_SOURCES}  ${GPU_GRIDDING_SOURCES} ${GRIDDING_SOURCES} ${GRIDDING_INCLUDE} SHARED)
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(${GRID_LIB_ATM_NAME})
CUDA_ADD_CUBLAS_TO_TARGET(${GRID_LIB_ATM_NAME})
