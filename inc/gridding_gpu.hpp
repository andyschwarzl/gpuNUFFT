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


void gridding3D_gpu(CufftType**	data,			// kspace output data array 
					int			data_count,		// data count, samples per trajectory
					int			n_coils,		// number of coils 
					DType*		crds,			//
					DType2*		imdata,			// image input data array
					int			imdata_count,	//			
					int			grid_width,		//
					DType*		kernel,			//
					int			kernel_count,	//
					int			kernel_width,	//
					int*		sectors,		//
					int			sector_count,	//
					int*		sector_centers,	//
					int			sector_width,	//
					int			im_width,		//
					DType		osr,			// oversampling ratio
					const GriddingOutput gridding_out);

void gridding3D_gpu_adj(DType2*		data,			// kspace input data array
						int			data_count,		// data count, samples per trajectory
						int			n_coils,		// number of coils 
						DType*		crds,			// 
						CufftType**	imdata,			// image output data array
						int			imdata_count,	//			
						int			grid_width,		//
						DType*		kernel,			//
						int			kernel_count,	//
						int			kernel_width,	//
						int*		sectors,		//
						int			sector_count,	//
						int*		sector_centers,	//
						int			sector_width,	//
						int			im_width,		//
						DType		osr,			// oversampling ratio
						bool    do_comp,
						DType*  density_comp,
						const GriddingOutput gridding_out);

#endif  // GRIDDING_GPU_HPP_*/
