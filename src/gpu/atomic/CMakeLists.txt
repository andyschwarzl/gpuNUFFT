MESSAGE("Atomic GPU Gridding Library")
#add exe to build from gridding3D

if(WIN32)
	CUDA_ADD_LIBRARY(${GRID_LIB_ATM_NAME} atomic_gridding_gpu.cu ../../griddingFunctions.cpp ${GRIDDING_INCLUDE})
else(WIN32)
	CUDA_ADD_LIBRARY(${GRID_LIB_ATM_NAME} atomic_gridding_gpu.cu ../../griddingFunctions.cpp ${GRIDDING_INCLUDE} SHARED) 
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(${GRID_LIB_ATM_NAME})
CUDA_ADD_CUBLAS_TO_TARGET(${GRID_LIB_ATM_NAME})
