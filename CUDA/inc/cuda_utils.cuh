#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH
#include "gpuNUFFT_types.hpp"
#include "gpuNUFFT_utils.hpp"

__constant__ gpuNUFFT::GpuNUFFTInfo GI;

__constant__ DType KERNEL[10000];

texture<float, 1, cudaReadModeElementType> texKERNEL;
texture<float, 2, cudaReadModeElementType> texKERNEL2D;
texture<float, 3, cudaReadModeElementType> texKERNEL3D;

texture<float2> texDATA;
texture<cufftComplex> texGDATA;

__inline__ __device__ float compute1DTextureLookup(float x, float y)
{
  return tex1D(texKERNEL, x) * tex1D(texKERNEL, y);
}

__inline__ __device__ float compute1DTextureLookup(float x, float y, float z)
{
  return tex1D(texKERNEL, x) * tex1D(texKERNEL, y) * tex1D(texKERNEL, z);
}

__inline__ __device__ float compute2DTextureLookup(float x, float y)
{
  return (float)tex2D(texKERNEL2D, (float)x, (float)y);
}

__inline__ __device__ float compute2DTextureLookup(float x, float y, float z)
{
  return (float)tex2D(texKERNEL2D, (float)x, (float)y) *
         tex2D(texKERNEL2D, (float)z, 0);
}

__inline__ __device__ float compute3DTextureLookup(float x, float y)
{
  return tex3D(texKERNEL3D, x, y, 0);
}

__inline__ __device__ float compute3DTextureLookup(float x, float y, float z)
{
  return tex3D(texKERNEL3D, x, y, z);
}

__inline__ __device__ float computeTextureLookup(float x, float y)
{
  // wired to 2d
  return compute2DTextureLookup((float)x, (float)y);
  // switch(GI.interpolationType)
  //{
  //  case 1: return compute1DTextureLookup(x,y);
  //  case 2: return compute2DTextureLookup(x,y);
  //  case 3: return compute3DTextureLookup(x,y);
  //  default: return (float)0.0;
  //}
}

__inline__ __device__ float computeTextureLookup(float x, float y, float z)
{
  // wired to 2d
  return compute2DTextureLookup(x, y, z);
  // switch(GI.interpolationType)
  //{
  //  case 1: return compute1DTextureLookup(x,y,z);
  //  case 2: return compute2DTextureLookup(x,y,z);
  //  case 3: return compute3DTextureLookup(x,y,z);
  //  default: return (float)0.0;
  //}
}

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
