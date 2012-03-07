#ifndef GRIDDING_GPU_HPP_
#define GRIDDING_GPU_HPP_

#include "griddingFunctions.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void gridding3D_gpu(float* data, 
					int data_cnt,
					float* crds, 
					float* gdata,
					int gdata_cnt,
					float* kernel,
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
	float kernel_radius;

	int width;
	
	int sector_dim;
	int sector_pad_width;
	int sector_offset;

	float radiusSquared;
	float dist_multiplier;
};

#endif  // GRIDDING_GPU_HPP_*/
