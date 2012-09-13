#ifndef CONFIG_H
#define CONFIG_H
#include "cufft.h"
 
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

#endif // CONFIG_H