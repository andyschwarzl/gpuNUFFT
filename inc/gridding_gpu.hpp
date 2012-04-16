#ifndef GRIDDING_GPU_HPP_
#define GRIDDING_GPU_HPP_

#include "griddingFunctions.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum GriddingOutput
{
	CONVOLUTION,
	FFT,
	DEAPODIZATION
};

void gridding3D_gpu(DType* data, 
					int data_cnt,
					DType* crds, 
					CufftType* gdata,
					int gdata_cnt,
					DType* kernel,
					int kernel_cnt,
					int* sectors, 
					int sector_count, 
					int* sector_centers,
					int sector_width,
					int kernel_width, 
					int kernel_count, 
					int width,
					const GriddingOutput gridding_out);

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
