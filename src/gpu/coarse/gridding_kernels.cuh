#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH
#include "cuda_utils.cuh"

__global__ void griddingKernel( DType* data, 
							    DType* crds, 
							    CufftType* gdata,
							    DType* kernel, 
							    int* sectors, 
								int* sector_centers,
								DType* temp_gdata
								);

__global__ void composeOutput(DType* temp_gdata, CufftType* gdata, int* sector_centers);

__global__ void performDeapodization(DType* gdata);

#endif