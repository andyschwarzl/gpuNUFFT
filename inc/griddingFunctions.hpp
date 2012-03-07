#ifndef GRIDDING_FUNCTIONS_H_
#define GRIDDING_FUNCTIONS_H_

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h> 

#ifdef _WIN32 
	#define _USE_MATH_DEFINES	
#endif

#include <math.h>
#include <assert.h>

#define DEFAULT_OVERSAMPLING_RATIO				1.0f
#define DEFAULT_KERNEL_WIDTH					3.0f  
#define DEFAULT_KERNEL_RADIUS		((DEFAULT_KERNEL_WIDTH) / 2.0f)

#define DEFAULT_WINDOW_LENGTH			1.0f

#define MAXIMUM_ALIASING_ERROR			0.001f

#define round(x) floor((x) + 0.5)

__inline__ __device__ __host__ void set_minmax (double x, int *min, int *max, int maximum, double radius)
{
	*min = (int) ceil (x - radius);
	*max = (int) floor (x + radius);
	//check boundaries
	if (*min < 0) *min = 0;
	if (*max >= maximum) *max = maximum;
}

long calculateGrid3KernelSize();
long calculateGrid3KernelSize(float osr, float kernel_radius);

void loadGrid3Kernel(float *kernTab,long kernel_entries, int kernel_width, float osr);
void loadGrid3Kernel(float *kernTab,long kernel_entries);
void loadGrid3Kernel(float *kernTab);

__inline__ __device__ __host__ int getIndex(int x, int y, int z, int gwidth)
{
	return x + gwidth * (y + gwidth * z);
}

__inline__ __device__ __host__ bool isOutlier(int x, int y, int z, int center_x, int center_y, int center_z, int width, int sector_offset)
{
		return ((center_x - sector_offset + x) >= width ||
						(center_x - sector_offset + x) < 0 ||
						(center_y - sector_offset + y) >= width ||
						(center_y - sector_offset + y) < 0 ||
						(center_z - sector_offset + z) >= width ||
						(center_z - sector_offset + z) < 0);
}
#endif  // GRIDDING_FUNCTIONS_H_
