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

typedef unsigned int IndType;

/** \brief Combined 2-tuple (x,y) of IndType */
typedef struct IndType2
{
  IndType x;
  IndType y;
  IndType2()
  {
  }
  IndType2(IndType x, IndType y) : x(x), y(y)
  {
  }
} IndType2;

/** \brief Combined 3-tuple (x,y,z) of IndType */
typedef struct IndType3
{
  IndType x;
  IndType y;
  IndType z;
  //      IndType3(){}
  //      IndType3(IndType x, IndType y, IndType z): x(x),y(y),z(z){}
} IndType3;

#endif  // CONFIG_H
