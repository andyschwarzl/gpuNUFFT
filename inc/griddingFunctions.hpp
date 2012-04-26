#ifndef GRIDDING_FUNCTIONS_H_
#define GRIDDING_FUNCTIONS_H_

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h> 

#include "config.hpp" 

#ifdef _WIN32 
	#define _USE_MATH_DEFINES	
#endif

#include <math.h>
#include <assert.h>

#define DEFAULT_OVERSAMPLING_RATIO				1.0f
#define DEFAULT_KERNEL_WIDTH					3
#define DEFAULT_KERNEL_RADIUS		((DEFAULT_KERNEL_WIDTH) / 2.0f)

#define DEFAULT_WINDOW_LENGTH			1.0f

#define MAXIMUM_ALIASING_ERROR			0.001f

#define round(x) floor((x) + 0.5)

#define sqr(__se) ((__se)*(__se))
#define BETA(__kw,__osr) (M_PI*sqrt(sqr(__kw/__osr*(__osr-0.5f))-0.8f))
#define I0_BETA(__kw,__osr)	(i0(BETA(__kw,__osr)))
#define kernel(__radius,__kw,__osr) (i0 (BETA(__kw,__osr) * sqrt (1 - sqr(__radius))) / I0_BETA(__kw,__osr))

/*BEGIN Zwart*/
/**************************************************************************
 *  FROM GRID_UTILS.C
 *
 *  Author: Nick Zwart, Dallas Turley, Ken Johnson, Jim Pipe 
 *  Date: 2011 apr 11
 *  Rev: 2011 aug 21
 * ...
*/
/************************************************************************** KERNEL */
/* 
 *	Summary: Allocates the 3D spherically symmetric kaiser-bessel function 
 *	         for kernel table lookup.
 *  
 *	         This lookup table is with respect to the radius squared.
 *	         and is based on the work described in Beatty et al. MRM 24, 2005
 */
static DType i0( DType x )
{
	float ax = fabs(x);
	float ans;
	float y;

	if (ax < 3.75f) 
    {
		y=x/3.75f,y=y*y;
		ans=1.0f+y*(3.5156229f+y*(3.0899424f+y*(1.2067492f
			   +y*(0.2659732f+y*(0.360768e-1f+y*0.45813e-2f)))));
	} 
    else 
    {
		y=3.75f/ax;
		ans=(exp(ax)/sqrt(ax))*(0.39894228f+y*(0.1328592e-1f
				+y*(0.225319e-2f+y*(-0.157565e-2f+y*(0.916281e-2f
				+y*(-0.2057706e-1f+y*(0.2635537e-1f+y*(-0.1647633e-1f
				+y*0.392377e-2f))))))));
	}
	return (ans);
}


/* LOADGRID3KERNEL()
 * Loads a radius of the circularly symmetric kernel into a 1-D array, with
 * respect to the kernel radius squared.
 */

/*END Zwart*/

__inline__ __device__ __host__ void set_minmax (DType x, int *min, int *max, int maximum, DType radius)
{
	*min = (int) ceil (x - radius);
	*max = (int) floor (x + radius);
	//check boundaries
	if (*min < 0) *min = 0;
	if (*max >= maximum) *max = maximum;
}

long calculateGrid3KernelSize();
long calculateGrid3KernelSize(DType osr, DType kernel_radius);

void loadGrid3Kernel(DType *kernTab,long kernel_entries, int kernel_width, DType osr);
void loadGrid3Kernel(DType *kernTab,long kernel_entries);
void loadGrid3Kernel(DType *kernTab);

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

__inline__ __device__ __host__ DType calculateDeapodizationAt(int x, int y, int z, DType grid_width_inv, DType osr, DType kernel_width)
{
	DType poly_x = sqrt(sqr((DType)M_PI) * sqr(kernel_width) * sqr(grid_width_inv) * sqr(x) - sqr(BETA(kernel_width,osr)));
	DType poly_y = sqrt(sqr((DType)M_PI) * sqr(kernel_width) * sqr(grid_width_inv) * sqr(y) - sqr(BETA(kernel_width,osr)));
	DType poly_z = sqrt(sqr((DType)M_PI) * sqr(kernel_width) * sqr(grid_width_inv) * sqr(z) - sqr(BETA(kernel_width,osr)));
	
	DType val_x = sin(poly_x)/poly_x;
	DType val_y = sin(poly_y)/poly_y;
	DType val_z = sin(poly_z)/poly_z;
	
	return val_x * val_y * val_z;
}


#endif  // GRIDDING_FUNCTIONS_H_
