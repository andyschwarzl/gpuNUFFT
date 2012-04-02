#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

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
};*/

#endif