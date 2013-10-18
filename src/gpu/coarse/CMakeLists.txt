MESSAGE("Coarse GPU Gridding Library")
#add exe to build from gridding3D

include_directories(${GRIDDING_INC_DIR})
													
cuda_include_directories(${GRIDDING_INC_DIR})

set(GPU_CU_SOURCES	${CMAKE_CURRENT_SOURCE_DIR}/gridding_kernels.cu 
					#${CMAKE_CURRENT_SOURCE_DIR}/../std_gridding_kernels.cu
									 )

if(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_NAME} ${GPU_CU_SOURCES} ${GPU_GRIDDING_SOURCES} ${GRIDDING_INCLUDE} ${GRIDDING_INC_DIR}/cufft_config.hpp)
else(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_NAME} ${GPU_CU_SOURCES} ${GPU_GRIDDING_SOURCES} ${GRIDDING_INCLUDE} ${GRIDDING_INC_DIR}/cufft_config.hpp SHARED)
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(${GRID_LIB_NAME})
CUDA_ADD_CUBLAS_TO_TARGET(${GRID_LIB_NAME})
