#ifndef CONFIG_H
#define CONFIG_H
 
<<<<<<< HEAD
#define MATLAB_DEBUG false
=======
#define MATLAB_DEBUG TRUE
>>>>>>> 343a45f60c08897c310b6a8cd4337b83f4bc81d6

/* #undef GPU_DOUBLE_PREC */

#ifdef GPU_DOUBLE_PREC
	typedef double DType;
	typedef double3 DType3;
#else
	typedef float DType;
	typedef float3 DType3;
#endif

#endif // CONFIG_H
