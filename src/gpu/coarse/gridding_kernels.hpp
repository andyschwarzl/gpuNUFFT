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
						 GriddingInfo* gi_host
						);

void composeOutput(DType* temp_gdata_d, 
				   CufftType* gdata_d, 
				   int* sector_centers_d,
				   GriddingInfo* gi_host);

void performFFTShift(CufftType* gdata_d,
					 FFTShiftDir shift_dir,
					 int width);

void performDeapodization(CufftType* gdata,
						 GriddingInfo* gi_host);

#endif