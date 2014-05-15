#ifndef GRIDDING_CPU_H_
#define GRIDDING_CPU_H_

#include "gpuNUFFT_utils.hpp"

void gpuNUFFT_cpu(DType* data, 
				DType* crds, 
				DType* gdata,
				DType* kernel, 
				int* sectors, 
				int sector_count, 
				int* sector_centers,
				int sector_width,
				int kernel_width, 
				int kernel_count, 
				int width);

#endif  // GRIDDING_CPU_H_
