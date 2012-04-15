#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cufft.h"
#include "gridding_gpu.hpp"

__constant__ GriddingInfo GI;

#define HANDLE_ERROR(err) { \
	if (err != cudaSuccess) \
	{ \
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), \
                __FILE__, __LINE__ ); \
        exit( EXIT_FAILURE ); \
	}}\

template<typename TypeName>
inline void allocateDeviceMem(TypeName** device_ptr, int num_elements)
{
	HANDLE_ERROR(cudaMalloc(device_ptr,num_elements*sizeof(TypeName)));
}

inline void freeDeviceMem(void* device_ptr)
{
	HANDLE_ERROR(cudaFree(device_ptr));
}

template<typename TypeName>
inline void copyToDevice(TypeName* host_ptr, TypeName* device_ptr, int num_elements)
{
	HANDLE_ERROR(cudaMemcpy(device_ptr, host_ptr,num_elements*sizeof(TypeName),cudaMemcpyHostToDevice ));
}

template<typename TypeName>
inline void allocateAndCopyToDeviceMem(TypeName** device_ptr, TypeName* host_ptr, int num_elements)
{
	allocateDeviceMem<TypeName>(device_ptr,num_elements);
	copyToDevice<TypeName>(host_ptr,*device_ptr,num_elements);
}

template<typename TypeName>
inline void copyFromDevice(TypeName* device_ptr, TypeName* host_ptr, int num_elements)
{
	HANDLE_ERROR(cudaMemcpy(host_ptr, device_ptr,num_elements*sizeof(TypeName),cudaMemcpyDeviceToHost ));
}

__device__ inline float atomicFloatAdd(float* address, float value)
{
  float old = value;  
  float ret=atomicExch(address, 0.0f);
  float new_old=ret+old;
  while ((old = atomicExch(address, new_old))!=0.0f)
  {
        new_old = atomicExch(address, 0.0f);
        new_old += old;
  }
  return ret;
};

GriddingInfo* initAndCopyGriddingInfo(int sector_count, 
							 int sector_width,
							 int kernel_width,
							 int kernel_count, 
							 int width)
{
	GriddingInfo* gi_host = (GriddingInfo*)malloc(sizeof(GriddingInfo));

	gi_host->sector_count = sector_count;
	gi_host->sector_width = sector_width;
	
	gi_host->kernel_width = kernel_width; 
	gi_host->kernel_count = kernel_count;
	gi_host->width = width;

	DType kernel_radius = static_cast<DType>(kernel_width) / 2.0f;
	DType radius = kernel_radius / static_cast<DType>(width);
	DType width_inv = 1.0f / width;
	DType radiusSquared = radius * radius;
	DType kernelRadius_invSqr = 1 / radiusSquared;
	DType dist_multiplier = (kernel_count - 1) * kernelRadius_invSqr;
	printf("radius rel. to grid width %f\n",radius);
	int sector_pad_width = 10;//sector_width + 2*(int)(floor(kernel_width / 2.0f));
	int sector_dim = sector_pad_width  * sector_pad_width  * sector_pad_width ;
	int sector_offset = (int)(floor(sector_pad_width / 2.0f));

	gi_host->kernel_radius = kernel_radius;
	gi_host->sector_pad_width = sector_pad_width;
	gi_host->sector_dim = sector_dim;
	gi_host->sector_offset = sector_offset;
	gi_host->radiusSquared = radiusSquared;
	gi_host->dist_multiplier = dist_multiplier;

	printf("sector offset = %d\n",sector_offset);
	
	gi_host->sector_pad_width = sector_pad_width;
	
	printf("copy Gridding Info to symbol memory...\n");
	cudaMemcpyToSymbol(GI, gi_host,sizeof(GriddingInfo));
	//free(gi_host);
	printf("...done!\n");
	return gi_host;
}

#endif