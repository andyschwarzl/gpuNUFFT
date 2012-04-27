#ifndef GRIDDING_KERNELS_H
#define GRIDDING_KERNELS_H
#include "gridding_gpu.hpp"
/*__global__ void convolutionKernel( DType* data, 
							    DType* crds, 
							    CufftType* gdata,
							    DType* kernel, 
							    int* sectors, 
								int* sector_centers,
								DType* temp_gdata
								);

__global__ void composeOutputKernel(DType* temp_gdata, CufftType* gdata, int* sector_centers);
*/
//forward declarations

void performConvolution( DType* data_d, 
						 DType* crds_d, 
						 CufftType* gdata_d,
						 DType* kernel_d, 
						 int* sectors_d, 
						 int* sector_centers_d,
						 DType* temp_gdata_d,
						 dim3 grid_dim,
						 dim3 block_dim
						);

void composeOutput(DType* temp_gdata_d, 
				   CufftType* gdata_d, 
				   int* sector_centers_d,
				   dim3 grid_dim,
				   dim3 block_dim);

void performDeapodization(DType* gdata_d,
						 dim3 grid_dim,
						 dim3 block_dim);

#endif