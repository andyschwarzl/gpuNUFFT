#ifndef GRIDDING_GPU_HPP_
#define GRIDDING_GPU_HPP_

#include "griddingFunctions.hpp"
//#include "cuda_utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_THREADS_PER_SECTOR 1 //16x16
#define MAX_SECTOR_WIDTH 12 // 8x8x8 + Kernel with Width 5 -> 12x12x12
#define MAX_SECTOR_DIM 1728 // 12x12x12

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
					DType osr,
					const GriddingOutput gridding_out);

struct GriddingInfo 
{
	int sector_count;
	int sector_width;
	
	int kernel_width; 
	int kernel_widthSquared;
	DType kernel_widthInvSquared;
	int kernel_count;
	DType kernel_radius;

	int width;
	DType width_inv;
	DType osr;
	
	int sector_dim;
	int sector_pad_width;
	int sector_offset;

	DType radiusSquared;
	DType dist_multiplier;
};


#endif  // GRIDDING_GPU_HPP_*/
