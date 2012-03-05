#ifndef GRIDDING_GPU_HPP_
#define GRIDDING_GPU_HPP_

#include "griddingFunctions.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

void gridding3D_gpu(float* data, 
				float* crds, 
				float* gdata,
				float* kernel, 
				int* sectors, 
				int sector_count, 
				int* sector_centers,
				int sector_width,
				int kernel_width, 
				int kernel_count, 
				int width);

#endif  // GRIDDING_GPU_HPP_*/
