#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cufft.h"
#include "griddingFunctions.hpp"
#include "gridding_operator.hpp"
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

// prototypes
// for function
// implementations that have to reside in cu file
void initConstSymbol(const char* symbol, const void* src, IndType count);

void initTexture(const char* symbol, cudaArray** devicePtr, GriddingND::Array<DType> hostTexture);

void freeTexture(const char* symbol,cudaArray* devicePtr);
#endif
