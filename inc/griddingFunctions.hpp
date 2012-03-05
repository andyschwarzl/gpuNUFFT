#ifndef GRIDDING_FUNCTIONS_H_
#define GRIDDING_FUNCTIONS_H_

#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32 
	#define _USE_MATH_DEFINES	
#endif

#include <math.h>
#include <assert.h>

#define DEFAULT_OVERSAMPLING_RATIO				1
#define DEFAULT_KERNEL_WIDTH					3.0  
#define DEFAULT_KERNEL_RADIUS		((DEFAULT_KERNEL_WIDTH) / 2.0)

#define DEFAULT_WINDOW_LENGTH			1.0

#define MAXIMUM_ALIASING_ERROR			0.001


#define round(x) floor((x) + 0.5)

void set_minmax (double x, int *min, int *max, int maximum, double radius);

long calculateGrid3KernelSize();
long calculateGrid3KernelSize(float osr, float kernel_radius);

void loadGrid3Kernel(float *kernTab,long kernel_entries, int kernel_width, float osr);
void loadGrid3Kernel(float *kernTab,long kernel_entries);
void loadGrid3Kernel(float *kernTab);

inline int getIndex(int x, int y, int z, int gwidth)
{
	return x + gwidth * (y + gwidth * z);
}

bool isOutlier(int x, int y, int z, int center_x, int center_y, int center_z, int width, int sector_offset);

#endif  // GRIDDING_FUNCTIONS_H_
