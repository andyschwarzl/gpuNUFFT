#ifndef GPUNUFFT_FUNCTIONS_H_
#define GPUNUFFT_FUNCTIONS_H_

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "config.hpp"

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <cmath>
#include <math.h>
#include <assert.h>

/**
 * @file
 * \brief Definition of util functions used in gridding operations.
 *
 * The functions are based on the grid3 work by Nick Zwart
 *
 * Lookup table creation extracted from GRID_UTILS.C
 *
 * Author: Nick Zwart, Dallas Turley, Ken Johnson, Jim Pipe
 *  Date: 2011 apr 11
 *  Rev: 2011 aug 21
 *  In: grid3_dct_11aug
 */

/** \brief Default oversampling ratio */
#define DEFAULT_OVERSAMPLING_RATIO 1.0f

/** \brief Default kernel width used for interpolation */
#define DEFAULT_KERNEL_WIDTH 3

/** \brief Default kernel radius */
#define DEFAULT_KERNEL_RADIUS ((DEFAULT_KERNEL_WIDTH) / 2.0f)

/** \brief Default window length */
#define DEFAULT_WINDOW_LENGTH 1.0f

/** \brief Maximum aliasing amplitude (nearest neighbor table lookup) defined by
 * Beatty et al. */
#define MAXIMUM_ALIASING_ERROR 0.001f
/** \brief Maximum aliasing amplitude (linear interpolation table lookup)
 * defined by Beatty et al. */
#define MAXIMUM_ALIASING_ERROR_LIN_INT 0.0001f

/** \brief Square value */
#define sqr(__se) ((__se) * (__se))

/**
 * \brief Modified Kaiser Bessel function of zero-th order. */
DType i0(DType x);

/** \brief beta function used in interpolation function. See Beatty et al. */
__inline__ DType BETA(IndType kernelWidth, DType osr)
{
  return (DType)M_PI * sqrt(std::pow((DType)kernelWidth / osr * (osr - (DType)0.5), (DType)2.0) - (DType)0.8);
}

/** \brief I_0 function used in interpolation function. See Beatty et al. */
__inline__ DType I0_BETA(IndType kernelWidth, DType osr)
{
  return i0(BETA(kernelWidth, osr));
}

/** \brief Interpolation Kernel evaluation for radius */
__inline__ DType kernel(DType radius, IndType kernelWidth, DType osr)
{
  return i0(BETA(kernelWidth, osr) * sqrt((DType)1.0 - std::pow(radius, (DType)2.0))) / I0_BETA(kernelWidth, osr);
}

/*  KERNEL
*	Summary: Allocates the 3D spherically symmetric kaiser-bessel function
*	         for kernel table lookup.
*
*	         This lookup table is with respect to the radius squared.
*	         and is based on the work described in Beatty et al. MRM 24, 2005
*/

/**
  * \brief Compute the boundary indices for interpolation of the value x with
 * respect to the kernel radius.
  */
__inline__ __device__ __host__ void set_minmax(DType *x, int *min, int *max,
                                               int maximum, DType radius)
{
  *min = (int)ceil(*x - radius);
  *max = (int)floor(*x + radius);
  // check boundaries
  if (*min < 0)
    *min = 0;
  if (*max < 0)
    *max = 0;
  if (*max >= maximum)
    *max = maximum;
  // if (*x >= (DType)maximum) *x = (DType)(maximum-radius);
  if (*min >= (DType)maximum)
    *min = (int)(maximum - 2 * radius);
}

long calculateGrid3KernelSize();
long calculateGrid3KernelSize(DType osr, IndType kernel_width);
long calculateKernelSizeLinInt(DType osr, IndType kernel_width);

/** \brief Loads a radius of the circularly symmetric kernel into a 1-d array,
* with
* respect to the kernel radius squared.
*/
void loadGrid3Kernel(DType *kernTab, long kernel_entries);

/** \brief Loads a radius of the circularly symmetric kernel into a 1-d array,
* with
* respect to the kernel radius squared.
*/
void loadGrid3Kernel(DType *kernTab);

// -- END Zwart

/** \brief Loads a radius of the circularly symmetric kernel into a 1-d array,
* with
* respect to the kernel radius squared.
*/
void load1DKernel(DType *kernTab, long kernel_entries, int kernel_width,
                  DType osr);

/** \brief Loads a radius of the circularly symmetric kernel into a 2-d array,
* with
* respect to the kernel radius squared.
*/
void load2DKernel(DType *kernTab, long kernel_entries, int kernel_width,
                  DType osr);

/** \brief Loads a radius of the circularly symmetric kernel into a 3-d array, with
* respect to the kernel radius squared. 
*/ void
load3DKernel(DType *kernTab, long kernel_entries, int kernel_width, DType osr);

/** \brief Convert position (x,y,z) to index in linear array */
__inline__ __device__ __host__ int getIndex(int x, int y, int z, int gwidth)
{
  return x + gwidth * (y + gwidth * z);
}

/** \brief Convert position (x,y) to index in linear array */
__inline__ __device__ __host__ int getIndex2D(int x, int y, int gwidth)
{
  return x + gwidth * (y);
}

/** \brief Compute position (x,y,z) from linear index */
__inline__ __device__ __host__ void getCoordsFromIndex(int index, int *x,
                                                       int *y, int *z, int w)
{
  *x = index % w;
  *z = (int)(index / (w * w));
  int r = index - *z * w * w;
  *y = (int)(r / w);
}

/** \brief Compute position (x,y,z) in grid defined by (w_x,w_y,w_z) from linear
 * index */
__inline__ __device__ __host__ void
getCoordsFromIndex(int index, int *x, int *y, int *z, int w_x, int w_y, int w_z)
{
  *x = index % w_x;
  *z = (int)(index / (w_x * w_y));
  int r = index - *z * w_x * w_y;
  *y = (int)(r / w_x);
}

/** \brief Compute position (x,y) in grid defined by (w,w) from linear index */
__inline__ __device__ __host__ void getCoordsFromIndex2D(int index, int *x,
                                                         int *y, int w)
{
  *x = index % w;
  *y = (int)(index / w);
}

/** \brief Compute position (x,y) in grid defined by (w_x,w_y) from linear index
 */
__inline__ __device__ __host__ void
getCoordsFromIndex2D(int index, int *x, int *y, int w_x, int w_y)
{
  *x = index % w_x;
  *y = (int)(index / w_x);
}

/** \brief Compute relative grid position of the passed k-space data point. */
inline __device__ DType mapKSpaceToGrid(DType pos, IndType gridDim,
                                        IndType sectorCenter, int sectorOffset)
{
  return (pos * (DType)gridDim) + ((DType)0.5 * ((DType)gridDim /*-1*/)) -
         (DType)sectorCenter + (DType)sectorOffset;
}

/** \brief Compute relative k space position of the passed grid position. */
inline __device__ DType mapGridToKSpace(int gridPos, IndType gridDim,
                                        IndType sectorCenter, int sectorOffset)
{
  return static_cast<DType>((DType)gridPos + (DType)sectorCenter -
                            (DType)sectorOffset) /
             static_cast<DType>((DType)gridDim /* - 1*/) -
         (DType)0.5;
}

/** \brief Evaluate whether position (x,y,z) inside the defined sector located
 * at (center_x,center_y,center_z) lies outside the grid defined by
 * (width,width,width).  */
__inline__ __device__ __host__ bool isOutlier(int x, int y, int z, int center_x,
                                              int center_y, int center_z,
                                              int width, int sector_offset)
{
  return ((center_x - sector_offset + x) >= width ||
          (center_x - sector_offset + x) < 0 ||
          (center_y - sector_offset + y) >= width ||
          (center_y - sector_offset + y) < 0 ||
          (center_z - sector_offset + z) >= width ||
          (center_z - sector_offset + z) < 0);
}

/** \brief Evaluate whether position (x,y,z) inside the defined sector located
 * at (center_x,center_y,center_z) lies outside the grid defined by dim.  */
__inline__ __device__ __host__ bool isOutlier(int x, int y, int z, int center_x,
                                              int center_y, int center_z,
                                              IndType3 dim, int sector_offset)
{
  return ((center_x - sector_offset + x) >= (int)dim.x ||
          (center_x - sector_offset + x) < 0 ||
          (center_y - sector_offset + y) >= (int)dim.y ||
          (center_y - sector_offset + y) < 0 ||
          (center_z - sector_offset + z) >= (int)dim.z ||
          (center_z - sector_offset + z) < 0);
}

/** \brief Evaluate whether position (x,y) inside the defined sector located at
 * (center_x,center_y) lies outside the grid defined by (width,width).  */
__inline__ __device__ __host__ bool isOutlier2D(int x, int y, int center_x,
                                                int center_y, int width,
                                                int sector_offset)
{
  return ((center_x - sector_offset + x) >= width ||
          (center_x - sector_offset + x) < 0 ||
          (center_y - sector_offset + y) >= width ||
          (center_y - sector_offset + y) < 0);
}

/** \brief Evaluate whether position (x,y) inside the defined sector located at
 * (center_x,center_y) lies outside the grid defined by dim.  */
__inline__ __device__ __host__ bool isOutlier2D(int x, int y, int center_x,
                                                int center_y, IndType3 dim,
                                                int sector_offset)
{
  return ((center_x - sector_offset + x) >= (int)dim.x ||
          (center_x - sector_offset + x) < 0 ||
          (center_y - sector_offset + y) >= (int)dim.y ||
          (center_y - sector_offset + y) < 0);
}

/** \brief Calculate the coord array index on the opposite side of the grid.
  *
  * @see isOutlier
  */
__inline__ __device__ __host__ int calculateOppositeIndex(int coord, int center,
                                                          int width, int offset)
{
  // return (center - offset + coord) % width;
  if ((center - offset + coord) >= width)
    return (center - offset + coord) - width;
  else if ((center - offset + coord) < 0)
    return (center - offset + coord + width);
  else
    return center - offset + coord;
}

/** \brief Calculate the deapodization value for the specified grid position
  *coord
  *
  * See Beatty et al. for details
  *
  */
__inline__ __device__ __host__ DType
    calculateDeapodizationValue(int coord, DType grid_width_inv,
                                int kernel_width, DType beta)
{
  DType poly =
      sqr((DType)M_PI) * sqr(kernel_width) * sqr(grid_width_inv) * sqr(coord) -
      sqr(beta);
  DType val;
  // sqrt for negative values not defined -> workaround with sinh
  if (poly >= 0)
    val = sin(sqrt(poly)) / sqrt(poly);
  else
    val = sinh(sqrt((DType)-1.0 * poly)) / sqrt((DType)-1.0 * poly);

  return val;
}

/** \brief Calculate the deapodization value for the specified grid position
  *(x,y,z)
  *
  * See Beatty et al. for details
  *
  */
__inline__ __device__ __host__ DType
    calculateDeapodizationAt(int x, int y, int z, IndType3 width_offset,
                             DType3 grid_width_inv, int kernel_width,
                             DType beta, DType norm_val)
{
  int x_shifted = x - (int)width_offset.x;
  int y_shifted = y - (int)width_offset.y;
  int z_shifted = z - (int)width_offset.z;

  DType val_x = calculateDeapodizationValue(x_shifted, grid_width_inv.x,
                                            kernel_width, beta);
  DType val_y = calculateDeapodizationValue(y_shifted, grid_width_inv.y,
                                            kernel_width, beta);
  DType val_z = calculateDeapodizationValue(z_shifted, grid_width_inv.z,
                                            kernel_width, beta);

  return val_x * val_y * val_z / norm_val;
}

/** \brief Calculate the deapodization value for the specified grid position
  *(x,y)
  *
  * See Beatty et al. for details
  *
  */
__inline__ __device__ __host__ DType
    calculateDeapodizationAt2D(int x, int y, IndType3 width_offset,
                               DType3 grid_width_inv, int kernel_width,
                               DType beta, DType norm_val)
{
  int x_shifted = x - (int)width_offset.x;
  int y_shifted = y - (int)width_offset.y;

  DType val_x = calculateDeapodizationValue(x_shifted, grid_width_inv.x,
                                            kernel_width, beta);
  DType val_y = calculateDeapodizationValue(y_shifted, grid_width_inv.y,
                                            kernel_width, beta);

  return val_x * val_y / norm_val;
}

#endif  // GPUNUFFT_FUNCTIONS_H_
