#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH
#include "gpuNUFFT_types.hpp"
#include "gpuNUFFT_utils.hpp"

__constant__ gpuNUFFT::GpuNUFFTInfo GI;

__constant__ DType KERNEL[5000];

texture<DType,1,cudaReadModeElementType> texKERNEL;
texture<DType,2,cudaReadModeElementType> texKERNEL2D;
texture<DType,3,cudaReadModeElementType> texKERNEL3D;

texture<DType2> texDATA;
texture<CufftType> texGDATA;

__inline__ __device__ DType compute1DTextureLookup(DType x, DType y)
{
  return tex1D(texKERNEL,x)*tex1D(texKERNEL,y);
}

__inline__ __device__ DType compute1DTextureLookup(DType x, DType y, DType z)
{
  return tex1D(texKERNEL,x)*tex1D(texKERNEL,y)*tex1D(texKERNEL,z);
}

__inline__ __device__ DType compute2DTextureLookup(DType x, DType y)
{
  return tex2D(texKERNEL2D,x,y);
}

__inline__ __device__ DType compute2DTextureLookup(DType x, DType y, DType z)
{
  return tex2D(texKERNEL2D,x,y)*tex2D(texKERNEL2D,z,0);
}

__inline__ __device__ DType compute3DTextureLookup(DType x, DType y)
{
  return tex3D(texKERNEL3D,x,y,0);
}

__inline__ __device__ DType compute3DTextureLookup(DType x, DType y, DType z)
{
  return tex3D(texKERNEL3D,x,y,z);
}

__inline__ __device__ DType computeTextureLookup(DType x, DType y)
{
  //wired to 2d
  return compute2DTextureLookup(x,y);
  //switch(GI.interpolationType)
  //{
  //  case 1: return compute1DTextureLookup(x,y);
  //  case 2: return compute2DTextureLookup(x,y);
  //  case 3: return compute3DTextureLookup(x,y);
  //  default: return (DType)0.0;
  //}
}

__inline__ __device__ DType computeTextureLookup(DType x, DType y, DType z)
{
  //wired to 2d
  return compute2DTextureLookup(x,y,z);
  //switch(GI.interpolationType)
  //{
  //  case 1: return compute1DTextureLookup(x,y,z);
  //  case 2: return compute2DTextureLookup(x,y,z);
  //  case 3: return compute3DTextureLookup(x,y,z);
  //  default: return (DType)0.0;
  //}
}

#if __CUDA_ARCH__ < 200
	#define THREAD_BLOCK_SIZE 256 
#else
	#define THREAD_BLOCK_SIZE 256
#endif

#endif
