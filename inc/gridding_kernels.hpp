#ifndef GRIDDING_KERNELS_H
#define GRIDDING_KERNELS_H
#include "gridding_gpu.hpp"
#include "cuda_utils.hpp"

//INVERSE Operations

// gridding function prototypes
void performConvolution( DType2* data_d, 
						 DType* crds_d, 
						 CufftType* gdata_d,
						 DType* kernel_d, 
						 int* sectors_d, 
						 int* sector_centers_d,
						 GriddingInfo* gi_host
						);
																	
void performForwardConvolution( CufftType*		data_d, 
								DType*			crds_d, 
								CufftType*		gdata_d,
								DType*			kernel_d, 
								int*			sectors_d, 
								int*			sector_centers_d,
								GriddingInfo*	gi_host
								);

void performFFTScaling(CufftType* data,
	                   int N, 
					   GriddingInfo* gi_host);

void performDensityCompensation(DType2* data, DType* density_comp, GriddingInfo* gi_host);

void performFFTShift(CufftType* gdata_d,
					 FFTShiftDir shift_dir,
					 int width);

void performCrop(CufftType* gdata_d,
				 CufftType* imdata_d,
				 GriddingInfo* gi_host);

void performDeapodization(CufftType* imdata_d,
						 GriddingInfo* gi_host);

void performDeapodization(CufftType* imdata_d,
													DType* deapo_d,
													GriddingInfo* gi_host);
//FORWARD Operations

void performForwardDeapodization(DType2* imdata_d,
								 GriddingInfo* gi_host);

void performForwardDeapodization(DType2* imdata_d,
								 DType* deapo_d,
								 GriddingInfo* gi_host);

void performPadding(DType2* imdata_d,
					CufftType* gdata_d,					
					GriddingInfo* gi_host);

void precomputeDeapodization(DType* deapo_d,
							 GriddingInfo* gi_host);

#endif
