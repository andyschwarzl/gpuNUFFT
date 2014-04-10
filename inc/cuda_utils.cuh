#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH
#include "griddingFunctions.hpp"

__constant__ GriddingND::GriddingInfo GI;

__constant__ DType KERNEL[5000];

//texture<DType,1,cudaReadModeNormalizedFloat> texKERNEL;
texture<DType> texKERNEL;

#if __CUDA_ARCH__ < 200
	#define THREAD_BLOCK_SIZE 128 
#else
	#define THREAD_BLOCK_SIZE 128
#endif

#endif
