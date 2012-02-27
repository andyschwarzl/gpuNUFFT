#ifndef GRIDDING_FUNCTIONS_H_
#define GRIDDING_FUNCTIONS_H_

#include <stdlib.h>
#include <stdio.h>

#define DEFAULT_OVERSAMPLING_RATIO				1
#define DEFAULT_KERNEL_WIDTH					3.0  
#define DEFAULT_KERNEL_RADIUS		((DEFAULT_KERNEL_WIDTH) / 2.0)

#define DEFAULT_WINDOW_LENGTH			1.0

#define MAXIMUM_ALIASING_ERROR			0.001


void gridding3D(float* data, 
				float* crds, 
				float* gdata,
				float* kernel, 
				int* sectors, 
				int sector_count, 
				int* sector_centers,
				int sector_width,
				int kernel_width, 
				int kernel_count, 
				int width);

long calculateGrid3KernelSize();
long calculateGrid3KernelSize(float osr, float kernel_radius);

void loadGrid3Kernel(float *kernTab,long kernel_entries, int kernel_width, float osr);
void loadGrid3Kernel(float *kernTab,long kernel_entries);
void loadGrid3Kernel(float *kernTab);

#endif  // GRIDDING_FUNCTIONS_H_
