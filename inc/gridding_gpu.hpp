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

enum FFTShiftDir
{
	FORWARD,
	INVERSE
};

struct GriddingInfo 
{
	int data_count;
	int kernel_width; 
	int kernel_widthSquared;
	DType kernel_widthInvSquared;
	int kernel_count;
	DType kernel_radius;

	int grid_width;		
	int grid_width_dim;  
	int grid_width_offset;
	DType grid_width_inv;

	int im_width;
	int im_width_dim;
	int im_width_offset;

	DType osr;
	
	int sector_count;
	int sector_width;
	int sector_dim;
	int sector_pad_width;
	int sector_pad_max;
	int sector_offset;

	DType radiusSquared;
	DType dist_multiplier;
};

#endif  // GRIDDING_GPU_HPP_*/
