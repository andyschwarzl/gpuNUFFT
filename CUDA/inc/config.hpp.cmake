@WARNING@

#ifndef CONFIG_H
#define CONFIG_H
#include "cufft.h"

/**
 * @file
 * \brief Definition of types used in gpuNUFFT
 *
 * Depends on CMAKE build parameters MATLAB_DEBUG, DEBUG, GPU_DOUBLE_PREC
 *
 */

#define MATLAB_DEBUG @MATLAB_DEBUG@
#define DEBUG @DEBUG@

#cmakedefine GPU_DOUBLE_PREC

#ifdef GPU_DOUBLE_PREC
  typedef double DType;
  typedef double2 DType2;
  typedef double3 DType3;
  typedef cufftDoubleComplex CufftType;
#else
  typedef float DType;
  typedef float2 DType2;
  typedef float3 DType3;
  typedef cufftComplex CufftType;
#endif

typedef unsigned long int SizeType;
typedef unsigned long int IndType;
typedef uint2 IndType2;
typedef uint3 IndType3;

#endif  // CONFIG_H
