MESSAGE("Coarse GPU Gridding Library")
#add exe to build from gridding3D
if(WIN32)
	CUDA_ADD_LIBRARY(gridding3D gridding_gpu.cu gridding_kernels.cu  gridding_kernels.hpp  ../../griddingFunctions.cpp ${GRIDDING_INCLUDE})
else(WIN32)
	CUDA_ADD_LIBRARY(gridding3D gridding_gpu.cu gridding_kernels.cu gridding_kernels.hpp ../../griddingFunctions.cpp ${GRIDDING_INCLUDE} SHARED) 
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(gridding3D)
CUDA_ADD_CUBLAS_TO_TARGET(gridding3D)
