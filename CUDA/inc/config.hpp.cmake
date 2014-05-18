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

	typedef unsigned int IndType;
	
	typedef
	struct IndType2 { IndType x;
	                  IndType y;
      IndType2(){}
      IndType2(IndType x, IndType y): x(x),y(y){}
	} IndType2;

	typedef 
  struct IndType3 {	 IndType x;
								 IndType y;
								 IndType z;
//      IndType3(){}
//      IndType3(IndType x, IndType y, IndType z): x(x),y(y),z(z){}
	} IndType3;

#endif // CONFIG_H
