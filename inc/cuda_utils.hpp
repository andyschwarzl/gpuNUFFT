#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define HANDLE_ERROR(err) { \
	if (err != cudaSuccess) \
	{ \
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), \
                __FILE__, __LINE__ ); \
        exit( EXIT_FAILURE ); \
	}}\


#endif