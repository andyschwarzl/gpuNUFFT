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
#define MAXIMUM_ALIASING_ERROR_LIN_INT			0.0001f

#define round(x) floor((x) + 0.5)

#define sqr(__se) ((__se)*(__se))
#define BETA(__kw,__osr) (M_PI*sqrt(sqr(__kw/__osr*(__osr-0.5f))-0.8f))
#define I0_BETA(__kw,__osr)	(i0(BETA(__kw,__osr)))
#define kernel(__radius,__kw,__osr) (i0 (BETA(__kw,__osr) * sqrt (1 - sqr(__radius))) / I0_BETA(__kw,__osr))

/**************************************************************************
*  Lookup table creation extracted from GRID_UTILS.C
*
*  Author: Nick Zwart, Dallas Turley, Ken Johnson, Jim Pipe 
*  Date: 2011 apr 11
*  Rev: 2011 aug 21
*  In: grid3_dct_11aug
*/
/*  KERNEL 
*	Summary: Allocates the 3D spherically symmetric kaiser-bessel function 
*	         for kernel table lookup.
*  
*	         This lookup table is with respect to the radius squared.
*	         and is based on the work described in Beatty et al. MRM 24, 2005
*/
static double i0( double x )
{
  double ax = abs(x);
  double ans;
  double y;

  if (ax < 3.75) 
  {
    y=x/3.75,y=y*y;
    ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
      +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
  } 
  else 
  {
    y=3.75/ax;
    ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
      +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
      +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
      +y*0.392377e-2))))))));
  }
  return (ans);
}


/* LOADGRID3KERNEL()
* Loads a radius of the circularly symmetric kernel into a 1-D array, with
* respect to the kernel radius squared.
*/

__inline__ __device__ __host__ void set_minmax (DType *x, int *min, int *max, int maximum, DType radius)
{
  *min = (int) ceil (*x - radius);
  *max = (int) floor (*x + radius);
  //check boundaries
  if (*min < 0) *min = 0;
  if (*max >= maximum) *max = maximum;
  //if (*x >= (DType)maximum) *x = (DType)(maximum-radius);
  if (*min >= (DType)maximum) *min = (int)(maximum-2*radius);

}

long calculateGrid3KernelSize();
long calculateGrid3KernelSize(DType osr, DType kernel_radius);
long calculateKernelSizeLinInt(double osr, double kernel_radius);

void loadGrid3Kernel(DType *kernTab,long kernel_entries, int kernel_width, DType osr);
void loadGrid3Kernel(DType *kernTab,long kernel_entries);
void loadGrid3Kernel(DType *kernTab);
/*END Zwart*/

void load2DKernel(DType *kernTab,long kernel_entries, int kernel_width, DType osr);
void load3DKernel(DType *kernTab,long kernel_entries, int kernel_width, DType osr);


__inline__ __device__ __host__ int getIndex(int x, int y, int z, int gwidth)
{
  return x + gwidth * (y + gwidth * z);
}

__inline__ __device__ __host__ int getIndex2D(int x, int y, int gwidth)
{
  return x + gwidth * (y);
}

__inline__ __device__ __host__ void getCoordsFromIndex(int index, int* x, int* y, int* z, int w)
{
  *x = index % w;
  *z = (int)(index / (w*w)) ;
  int r = index - *z * w * w;
  *y = (int)(r / w);	
}

__inline__ __device__ __host__ void getCoordsFromIndex(int index, int* x, int* y, int* z, int w_x, int w_y, int w_z)
{
  *x = index % w_x;
  *z = (int)(index / (w_x*w_y)) ;
  int r = index - *z * w_x * w_y;
  *y = (int)(r / w_x);	
}

__inline__ __device__ __host__ void getCoordsFromIndex2D(int index, int* x, int* y, int w)
{
  *x = index % w;
  *y = (int)(index / w);        
}

__inline__ __device__ __host__ void getCoordsFromIndex2D(int index, int* x, int* y,  int w_x, int w_y)
{
  *x = index % w_x;
  *y = (int)(index / w_y);        
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

__inline__ __device__ __host__ bool isOutlier(int x, int y, int z, int center_x, int center_y, int center_z, IndType3 dim, int sector_offset)
{
  return ((center_x - sector_offset + x) >= (int)dim.x ||
    (center_x - sector_offset + x) < 0 ||
    (center_y - sector_offset + y) >= (int)dim.y ||
    (center_y - sector_offset + y) < 0 ||
    (center_z - sector_offset + z) >= (int)dim.z ||
    (center_z - sector_offset + z) < 0);
}


__inline__ __device__ __host__ bool isOutlier2D(int x, int y, int center_x, int center_y, int width, int sector_offset)
{
  return ((center_x - sector_offset + x) >= width ||
    (center_x - sector_offset + x) < 0 ||
    (center_y - sector_offset + y) >= width ||
    (center_y - sector_offset + y) < 0);
}

__inline__ __device__ __host__ bool isOutlier2D(int x, int y, int center_x, int center_y, IndType3 dim, int sector_offset)
{
  return ((center_x - sector_offset + x) >= (int)dim.x ||
    (center_x - sector_offset + x) < 0 ||
    (center_y - sector_offset + y) >= (int)dim.y ||
    (center_y - sector_offset + y) < 0);
}

__inline__ __device__ __host__ int calculateOppositeIndex(int coord,int center,int width, int offset)
{
  //return (center - offset + coord) % width;
  if ((center - offset + coord) >= width)
    return (center - offset + coord) - width;
  else if ((center - offset + coord) < 0)
    return (center - offset + coord + width);
  else
    return center - offset + coord;
}

__inline__ __device__ __host__ DType calculateDeapodizationValue(int coord, DType grid_width_inv, int kernel_width, DType beta)
{
  DType poly = sqr((DType)M_PI) * sqr(kernel_width) * sqr(grid_width_inv) * sqr(coord) - sqr(beta);
  DType val;
  //sqrt for negative values not defined -> workaround with sinh
  if (poly >= 0)
    val = sin(sqrt(poly))/sqrt(poly);
  else
    val = sinh(sqrt((DType)-1.0 * poly))/sqrt((DType)-1.0 * poly);

  return val;
}

__inline__ __device__ __host__ DType calculateDeapodizationAt(int x, int y, int z, IndType3 width_offset, DType3 grid_width_inv, int kernel_width, DType beta, DType norm_val)
{
  int x_shifted = x - (int)width_offset.x;
  int y_shifted = y - (int)width_offset.y;
  int z_shifted = z - (int)width_offset.z;

  DType val_x = calculateDeapodizationValue(x_shifted,grid_width_inv.x,kernel_width,beta);
  DType val_y = calculateDeapodizationValue(y_shifted,grid_width_inv.y,kernel_width,beta);
  DType val_z = calculateDeapodizationValue(z_shifted,grid_width_inv.z,kernel_width,beta);

  return val_x * val_y *  val_z / norm_val;
}

__inline__ __device__ __host__ DType calculateDeapodizationAt2D(int x, int y,IndType3 width_offset, DType3 grid_width_inv, int kernel_width, DType beta, DType norm_val)
{
  int x_shifted = x - (int)width_offset.x;
  int y_shifted = y - (int)width_offset.y;

  DType val_x = calculateDeapodizationValue(x_shifted,grid_width_inv.x,kernel_width,beta);
  DType val_y = calculateDeapodizationValue(y_shifted,grid_width_inv.y,kernel_width,beta);

  return val_x * val_y / norm_val;
}

#endif  // GRIDDING_FUNCTIONS_H_
