#ifndef GRIDDING_KERNELS_H
#define GRIDDING_KERNELS_H
#include "gridding_gpu.hpp"

//forward declarations
void performConvolution( DType* data_d, 
						 DType* crds_d, 
						 CufftType* gdata_d,
						 DType* kernel_d, 
						 int* sectors_d, 
						 int* sector_centers_d,
						 DType* temp_gdata_d,
						 dim3 grid_dim,
						 dim3 block_dim,
						 GriddingInfo* gi_host
						);

void composeOutput(DType* temp_gdata_d, 
				   CufftType* gdata_d, 
				   int* sector_centers_d,
				   dim3 grid_dim,
				   dim3 block_dim);

void performFFTShift(CufftType* gdata_d,
					 FFTShiftDir shift_dir,
					 int width);

void performDeapodization(CufftType* gdata,
						 dim3 grid_dim,
						 dim3 block_dim,
						 GriddingInfo* gi_host);

#endif