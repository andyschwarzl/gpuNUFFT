#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH
#include "gridding_gpu.hpp"

__constant__ GriddingInfo GI;
__constant__ DType KERNEL[5000];


#if __CUDA_ARCH__ < 200
	#define THREAD_BLOCK_SIZE 128 
#else
	#define THREAD_BLOCK_SIZE 256
#endif


#endif
