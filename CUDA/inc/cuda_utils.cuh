#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH
#include "gpuNUFFT_types.hpp"
#include "gpuNUFFT_utils.hpp"

__constant__ gpuNUFFT::GpuNUFFTInfo GI;

__constant__ DType KERNEL[10000];

#if __CUDA_ARCH__ < 200
#define THREAD_BLOCK_SIZE 256
#else
#define THREAD_BLOCK_SIZE 256
#endif

// From NVIDIA devtalk
#ifdef GPU_DOUBLE_PREC
__inline__ __device__ double atomicAdd(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

#endif
