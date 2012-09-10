#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH
#include "gridding_gpu.hpp"

__constant__ GriddingInfo GI;
__constant__ DType KERNEL[5000];
#endif
