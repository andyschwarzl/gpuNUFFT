#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cufft.h"
#include "gpuNUFFT_utils.hpp"
#include "gpuNUFFT_operator.hpp"
#include <stdarg.h>

/**
 * @file
 * \brief Util functions for CUDA memory and texture management.
 *
 *
 */

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) >= (Y) ? (X) : (Y))

#define HANDLE_ERROR(err)                                                      \
  {                                                                            \
    if (err != cudaSuccess)                                                    \
    {                                                                          \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,       \
             __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

/** \brief Allocation of device memory of the given extent for the defined type.
 *
 * @param device_ptr device pointer
 * @param num_elements amount of elements of size TypeName
 */
template <typename TypeName>
inline void allocateDeviceMem(TypeName **device_ptr, IndType num_elements)
{
  HANDLE_ERROR(cudaMalloc(device_ptr, num_elements * sizeof(TypeName)));
}

/** \brief Free of device memory.*/
inline void freeDeviceMem(void *device_ptr)
{
  HANDLE_ERROR(cudaFree(device_ptr));
}

/** \brief CUDA memcpy call to copy data from host to device
 *
 * @param host_ptr      host data pointer
 * @param device_ptr    device pointer
 * @param num_elements  amount of elements of size TypeName
 */
template <typename TypeName>
inline void copyToDevice(TypeName *host_ptr, TypeName *device_ptr,
                         IndType num_elements)
{
  HANDLE_ERROR(cudaMemcpy(device_ptr, host_ptr, num_elements * sizeof(TypeName),
                          cudaMemcpyHostToDevice));
}
/** \brief CUDA memcpy call to copy data from host to device
 *
 * @param host_ptr      host data pointer
 * @param device_ptr    device pointer
 * @param num_elements  amount of elements of size TypeName
 */
template <typename TypeName>
inline void copyToDeviceAsync(TypeName *host_ptr, TypeName *device_ptr,
                         IndType num_elements, cudaStream_t stream=0)
{
  HANDLE_ERROR(cudaMemcpyAsync(device_ptr, host_ptr, num_elements * sizeof(TypeName),
                          cudaMemcpyHostToDevice, stream));
}
/** \brief CUDA memory allocation and memcpy call to copy data from host to
 *device
 *
 * @param host_ptr      host data pointer
 * @param device_ptr    device pointer
 * @param num_elements  amount of elements of size TypeName
 */
template <typename TypeName>
inline void allocateAndCopyToDeviceMem(TypeName **device_ptr,
                                       TypeName *host_ptr, IndType num_elements)
{
  allocateDeviceMem<TypeName>(device_ptr, num_elements);
  copyToDevice<TypeName>(host_ptr, *device_ptr, num_elements);
}

/** \brief CUDA memory allocation call and initialization with given scalar
 *value
 *
 * @param device_ptr    device pointer
 * @param num_elements  amount of elements of size TypeName
 * @param value         Scalar value to initialize elements with
 */
template <typename TypeName>
inline void allocateAndSetMem(TypeName **device_ptr, IndType num_elements,
                              int value)
{
  allocateDeviceMem<TypeName>(device_ptr, num_elements);
  HANDLE_ERROR(cudaMemset(*device_ptr, value, num_elements * sizeof(TypeName)));
}

/** \brief CUDA memcpy call to copy data from device ptr to device ptr
 *
 * @param device_ptr_src   source device pointer
 * @param device_ptr_dest  destination device pointer
 * @param num_elements     amount of elements of size TypeName
 */
template <typename TypeName>
inline void copyDeviceToDevice(TypeName *device_ptr_src,
                               TypeName *device_ptr_dest, IndType num_elements)
{
  HANDLE_ERROR(cudaMemcpy(device_ptr_dest, device_ptr_src,
                          num_elements * sizeof(TypeName),
                          cudaMemcpyDeviceToDevice));
}

/** \brief Copy CUDA memory from device to host
 *
 * @param device_ptr    device pointer
 * @param host_ptr      host pointer
 * @param num_elements  amount of elements of size TypeName
 */
template <typename TypeName>
inline void copyFromDevice(TypeName *device_ptr, TypeName *host_ptr,
                           IndType num_elements)
{
  HANDLE_ERROR(cudaMemcpy(host_ptr, device_ptr, num_elements * sizeof(TypeName),
                          cudaMemcpyDeviceToHost));
}
/** \brief Copy CUDA memory from device to host
 *
 * @param device_ptr    device pointer
 * @param host_ptr      host pointer
 * @param num_elements  amount of elements of size TypeName
 */
template <typename TypeName>
inline void copyFromDeviceAsync(TypeName *device_ptr, TypeName *host_ptr,
                           IndType num_elements, cudaStream_t stream=0)
{
  HANDLE_ERROR(cudaMemcpyAsync(host_ptr, device_ptr, num_elements * sizeof(TypeName),
                          cudaMemcpyDeviceToHost, stream));
}
/** \brief Free variable list of device pointers. Use NULL as stopping element
 *
 * e.g.: freeTotalDeviceMemory(ptr1*, ptr2*,NULL);
 *
 * @param ptr Device pointer list
 */
inline void freeTotalDeviceMemory(void *ptr, ...)
{
  va_list list;
  va_start(list, ptr);
  void *p = ptr;
  int i = 0;
  while (true)
  {
    if (p == NULL)
      break;
    // printf("free dev ptr...%p %d\n",p,i);
    freeDeviceMem((void *)p);
    i++;
    p = va_arg(list, void *);
  }
  cudaDeviceSynchronize();
  if (DEBUG)
    printf("%d device pointers freed\n", i);

  va_end(list);
}

/** \brief Compute optimal grid dimensions for given thread count
 *
 * @param N problem size
 * @param thread_count count of threads used
 */
inline dim3 getOptimalGridDim(long N, long thread_count)
{
  return dim3(MIN((N + thread_count - 1) / thread_count,
                  128 * 128));  // 128*128 empiric, max is 256*256 = 65536
}

/** \brief Compute optimal 2d block dimensions for given thread count in y
 *direction
 *
 * @param threadsX required thread size in x direction
 * @param threadsY necessary thread size in y direction
 */
inline dim3 getOptimal2DBlockDim(long threadsX, long threadsY)
{
  return dim3(MIN(threadsX * threadsY, 1024) / threadsY, threadsY);
}

/** \brief Debug short device memory information (free/total) to stream if DEBUG
 *flag is set to true.
 *
 * @param force always print output
 * @param stream output stream
 */
inline void showMemoryInfo(bool force, FILE *stream)
{
  size_t free_mem = 0;
  size_t total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  if (DEBUG || force)
    fprintf(stream, "memory usage, free: %lu total: %lu\n", (SizeType)(free_mem),
    (SizeType)(total_mem));
}

/** \brief Debug short device memory information (free/total) to stdout if DEBUG
 *flag is set to true.
 *
 * @param force always print output
 */
inline void showMemoryInfo(bool force)
{
  showMemoryInfo(force, stdout);
}

/** \brief Debug short device memory information (free/total) to stream if DEBUG
 * flag is set to true.
 */
inline void showMemoryInfo()
{
  showMemoryInfo(false);
}

// CUDA kernel prototypes for function
// implementations that have to reside in cu file

/** \brief Initialize constant symbol on device
 *
 * CUDA Kernel function prototype.
 *
 * @param symbol Const symbol name
 */
void initConstSymbol(const char *symbol, const void *src, IndType count, cudaStream_t stream=0);

/** \brief Initialize texture memory on device
 *
 * CUDA Kernel function prototype.
 *
 * @param symbol Texture symbol name
 */
void initTexture(const char *symbol, cudaArray **devicePtr,
                 gpuNUFFT::Array<DType> hostTexture);

/** \brief Bind to 1-d texture on device
 *
 * CUDA Kernel function prototype.
 *
 * @param symbol Texture symbol name
 */
void bindTo1DTexture(const char *symbol, void *devicePtr, IndType count);

/** \brief Unbind from device texture
 *
 * CUDA Kernel function prototype.
 *
 * @param symbol Texture symbol name
 */
void unbindTexture(const char *symbol);

/** \brief Free texture memory on device
 *
 * CUDA Kernel function prototype.
 *
 * @param symbol Texture symbol name
 */
void freeTexture(const char *symbol, cudaArray *devicePtr);

#endif
