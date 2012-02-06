#ifndef GRIDDING_FUNCTIONS_H_
#define GRIDDING_FUNCTIONS_H_

#include <stdlib.h>
#include <stdio.h>

#define OVERSAMPLING_RATIO				1.5
#define KERNEL_WIDTH					5.0  
#define DEFAULT_RADIUS_FOV_PRODUCT		((KERNEL_WIDTH) / 2.0)
#define DEFAULT_KERNEL_TABLE_SIZE		800
#define DEFAULT_WINDOW_LENGTH			1.0

void gridding3D(float* data, 
				float* crds, 
				float* gdata,
				float* kernel, 
				int* sectors, 
				int sector_count, 
				int* sector_centers,
				int kernel_width, 
				int kernel_count, 
				int width);


void loadGrid3Kernel(float *kernTab, int kernel_entries);

#endif  // GRIDDING_FUNCTIONS_H_
