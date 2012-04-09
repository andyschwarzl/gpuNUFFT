#ifndef CONFIG_H
#define CONFIG_H
 
#define MATLAB_DEBUG FALSE

/* #undef GPU_DOUBLE_PREC */

#ifdef GPU_DOUBLE_PREC
	typedef double DType;
	typedef double3 DType3;
#else
	typedef float DType;
	typedef float3 DType3;
#endif

#endif // CONFIG_H
