#ifndef GRIDDING_KERNELS_H
#define GRIDDING_KERNELS_H
#include "gridding_gpu.hpp"

//INVERSE Operations
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

void performCrop(CufftType* gdata_d,
				 CufftType* imdata_d,
				 GriddingInfo* gi_host);

void performDeapodization(CufftType* imdata_d,
						 GriddingInfo* gi_host);

//FORWARD Operations

void performForwardDeapodization(DType* imdata_d,
						  GriddingInfo* gi_host);

void performPadding(DType*			imdata_d,
					CufftType*		gdata_d,					
					GriddingInfo*	gi_host);

void performForwardConvolution( CufftType* data_d, 
								 DType* crds_d, 
								 CufftType* gdata_d,
								 DType* kernel_d, 
								 int* sectors_d, 
								 int* sector_centers_d,
								 GriddingInfo* gi_host
								);



#endif
