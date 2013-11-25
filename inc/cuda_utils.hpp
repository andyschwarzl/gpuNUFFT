#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cufft.h"
#include "gridding_gpu.hpp"
#include <stdarg.h>

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) >= (Y) ? (X) : (Y))

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
inline void allocateAndSetMem(TypeName** device_ptr, int num_elements,int value)
{
	allocateDeviceMem<TypeName>(device_ptr,num_elements);
	HANDLE_ERROR(cudaMemset(*device_ptr,value,num_elements*sizeof(TypeName)));
}

template<typename TypeName>
inline void copyFromDevice(TypeName* device_ptr, TypeName* host_ptr, int num_elements)
{
	HANDLE_ERROR(cudaMemcpy(host_ptr, device_ptr,num_elements*sizeof(TypeName),cudaMemcpyDeviceToHost ));
}

inline void freeTotalDeviceMemory(void* ptr,...)
{
	va_list list;
	va_start(list,ptr); 
	void* p = ptr;
	int i =0;
	while(true)
	{	  
	  if(p==NULL)
		   break;
	  //printf("free dev ptr...%p %d\n",p,i);
	  freeDeviceMem((void*)p);
	  i++;
	  p= va_arg(list,void*);
	}  
	cudaThreadSynchronize();
	if (DEBUG)
		printf("%d device pointers freed\n",i);
	
    va_end(list);
}

/*__device__ inline float atomicFloatAdd(float* address, float value)
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
}*/

inline dim3 getOptimalGridDim(long im_dim, long thread_count)
{
	return dim3(MIN((im_dim+thread_count-1)/thread_count,128*128));//128*128 empiric, max is 256*256 = 65536
}

inline void showMemoryInfo(bool force, FILE* stream)
{
	size_t free_mem = 0;
	size_t total_mem = 0;
	cudaMemGetInfo(&free_mem, &total_mem);
	if (DEBUG || force)
		fprintf(stream,"memory usage, free: %lu total: %lu\n",free_mem,total_mem);
}
	
inline void showMemoryInfo(bool force)
{
	showMemoryInfo(force,stdout);
}	

inline void showMemoryInfo()
{
	showMemoryInfo(false);
}

// prototype
// implementation in cu file
void initConstSymbol(const char* symbol, const void* src, IndType count);


inline GriddingInfo* initAndCopyGriddingInfo(int sector_count, 
									  int sector_width,
									  int kernel_width,
									  int kernel_count, 
									  int grid_width,
									  int im_width,
									  DType osr,
									  int data_count)
{
	GriddingInfo* gi_host = (GriddingInfo*)malloc(sizeof(GriddingInfo));
    gi_host->data_count = data_count;
	gi_host->sector_count = sector_count;
	gi_host->sector_width = sector_width;
	
	gi_host->kernel_width = kernel_width; 
	gi_host->kernel_widthSquared = kernel_width * kernel_width;
	gi_host->kernel_count = kernel_count;
	gi_host->grid_width = grid_width;
	gi_host->grid_width_dim = grid_width * grid_width * grid_width;
	gi_host->grid_width_offset= (int)(floor(grid_width / (DType)2.0));

	gi_host->im_width = im_width;
	gi_host->im_width_dim = im_width * im_width * im_width;
	gi_host->im_width_offset = (int)(floor(im_width / (DType)2.0));

	DType kernel_radius = static_cast<DType>(kernel_width) / (DType)2.0;
	DType radius = kernel_radius / static_cast<DType>(grid_width);
	DType width_inv = (DType)1.0 / static_cast<DType>(grid_width);

	DType kernel_width_inv = (DType)1.0 / static_cast<DType>(kernel_width);

	DType radiusSquared = radius * radius;
	DType kernelRadius_invSqr = (DType)1.0 / radiusSquared;
	DType dist_multiplier = (kernel_count - 1) * kernelRadius_invSqr;
	if (DEBUG)
		printf("radius rel. to grid width %f\n",radius);
	int sector_pad_width = sector_width + 2*(int)(floor(kernel_width / (DType)2.0));
	int sector_dim = sector_pad_width  * sector_pad_width  * sector_pad_width ;
	int sector_offset = (int)(floor(sector_pad_width / (DType)2.0));

	gi_host->grid_width_inv = width_inv;
	gi_host->kernel_widthInvSquared = kernel_width_inv * kernel_width_inv;
	gi_host->osr = osr;
	
	gi_host->kernel_radius = kernel_radius;
	gi_host->sector_pad_width = sector_pad_width;
	gi_host->sector_pad_max = sector_pad_width - 1;
	gi_host->sector_dim = sector_dim;
	gi_host->sector_offset = sector_offset;
	gi_host->radiusSquared = radiusSquared;
	gi_host->dist_multiplier = dist_multiplier;
	
	if (DEBUG)
		printf("copy Gridding Info to symbol memory... size = %d \n",sizeof(GriddingInfo));

	initConstSymbol("GI",gi_host,sizeof(GriddingInfo));

	//free(gi_host);
	if (DEBUG)
		printf("...done!\n");
	return gi_host;
}


#endif
