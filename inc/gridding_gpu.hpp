#ifndef GRIDDING_GPU_HPP_
#define GRIDDING_GPU_HPP_

#include "griddingFunctions.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __DOUBLE_PREC__
	typedef double DType;
	typedef double3 DType3;
#else
	typedef float DType;
	typedef float3 DType3;
#endif

void gridding3D_gpu(DType* data, 
					int data_cnt,
					DType* crds, 
					DType* gdata,
					int gdata_cnt,
					DType* kernel,
					int kernel_cnt,
					int* sectors, 
					int sector_count, 
					int* sector_centers,
					int sector_width,
					int kernel_width, 
					int kernel_count, 
					int width);

struct GriddingInfo 
{
	int sector_count;
	int sector_width;
	
	int kernel_width; 
	int kernel_count;
	DType kernel_radius;

	int width;
	
	int sector_dim;
	int sector_pad_width;
	int sector_offset;

	DType radiusSquared;
	DType dist_multiplier;
};

#endif  // GRIDDING_GPU_HPP_*/
