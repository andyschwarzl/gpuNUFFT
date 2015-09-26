#ifndef PRECOMP_UTILS_H
#define PRECOMP_UTILS_H

#include "cuda_utils.hpp"
#include "gpuNUFFT_operator.hpp"
#include "gpuNUFFT_utils.hpp"
#include <cmath>

/**
  * @file
  * \brief UTIL functions for precomputation
  */

/** \brief Compute Sector Mapping for one coordinate based on the sector count
  */
__inline__ __device__ __host__ IndType
    computeSectorMapping(DType coord, IndType sectorCount)
{
  int sector = (int)std::floor((coord + 0.5) * sectorCount);
  if (sector >= (int)(sectorCount))
    sector = (int)(sectorCount - 1);
  if (sector < 0)
    sector = 0;
  return sector;
}

/** \brief Compute Sector Mapping for one coordinate based on the current grid
  *dimension and sector width
  *
  * This way is neccessary in order to support grid dimensions which are not
  *integer multiples of
  * the selected sector width.
  */
__inline__ __device__ __host__ IndType
    computeSectorMapping(DType coord, IndType gridDim, DType sectorWidth)
{
  // int sector = (int)std::floor((int)((coord + 0.5) * ((int)gridDim - 1))/
  // sectorWidth);
  int sector = (int)std::floor(
      round(coord * (DType)(gridDim /*-1*/) + 0.5 * (DType)(gridDim /*-1*/)) /
      sectorWidth);

  if (sector >= (int)std::ceil((DType)(gridDim /*-1*/) / sectorWidth))
    sector = (int)(std::ceil((DType)(gridDim /*-1*/) / sectorWidth) - 1);
  if (sector < 0)
    sector = 0;
  return sector;
}

/** \brief Compute Sector Mapping with the same resolution as in
 * resolutionSectorCount
  * but limit it to dimension in sectorCount
  * Used when z dimension is not the same as in x and y
  */
__inline__ __device__ __host__ IndType
    computeSectorMapping(DType coord, IndType sectorCount,
                         IndType resolutionSectorCount)
{
  double sectorKdim = 1.0 / resolutionSectorCount;
  double offset = ((resolutionSectorCount - sectorCount) / 2.0) * sectorKdim;

  int sector = (int)std::floor((coord + 0.5 - offset) * resolutionSectorCount);
  if (sector >= (int)(sectorCount))
    sector = (int)(sectorCount - 1);
  if (sector < 0)
    sector = 0;
  return sector;
}

/** \brief Compute Sector Mapping */
__inline__ __device__ __host__ IndType3
    computeSectorMapping(DType3 coord, gpuNUFFT::Dimensions sectorDims)
{
  IndType3 sector;
  sector.x = computeSectorMapping(coord.x, sectorDims.width);
  sector.y = computeSectorMapping(coord.y, sectorDims.height);
  sector.z = computeSectorMapping(coord.z, sectorDims.depth);
  return sector;
}

/** \brief Compute Sector Mapping */
__inline__ __device__ __host__ IndType3
    computeSectorMapping(DType3 coord, gpuNUFFT::Dimensions gridDims,
                         DType sectorWidth)
{
  IndType3 sector;
  sector.x = computeSectorMapping(coord.x, gridDims.width, sectorWidth);
  sector.y = computeSectorMapping(coord.y, gridDims.height, sectorWidth);
  sector.z = computeSectorMapping(coord.z, gridDims.depth, sectorWidth);
  return sector;
}

/** \brief Compute Sector Mapping for a 2D coord */
__inline__ __device__ __host__ IndType2
    computeSectorMapping(DType2 coord, gpuNUFFT::Dimensions sectorDims)
{
  IndType2 sector;
  sector.x = computeSectorMapping(coord.x, sectorDims.width);
  sector.y = computeSectorMapping(coord.y, sectorDims.height);
  return sector;
}

/** \brief Compute Sector Mapping for a 2D coord */
__inline__ __device__ __host__ IndType2
    computeSectorMapping(DType2 coord, gpuNUFFT::Dimensions gridDims,
                         DType sectorWidth)
{
  IndType2 sector;
  sector.x = computeSectorMapping(coord.x, gridDims.width, sectorWidth);
  sector.y = computeSectorMapping(coord.y, gridDims.height, sectorWidth);
  return sector;
}

/** \brief Compute linear index from 3-d point in grid defined by dim */
__inline__ __device__ __host__ int computeXYZ2Lin(int x, int y, int z,
                                                  gpuNUFFT::Dimensions dim)
{
  return x + (int)dim.width * (y + (int)dim.height * z);
}

/** \brief Compute linear index from 3-d point in grid defined by dim */
__inline__ __device__ __host__ int computeXYZ2Lin(int x, int y, int z,
                                                  IndType3 dim)
{
  return x + (int)dim.x * (y + (int)dim.y * z);
}

/** \brief Compute linear index from 2-d point in grid defined by dim */
__inline__ __device__ __host__ int computeXY2Lin(int x, int y,
                                                 gpuNUFFT::Dimensions dim)
{
  return x + (int)dim.width * y;
}

/** \brief Compute linear index from 2-d point in grid defined by dim */
__inline__ __device__ __host__ int computeXY2Lin(int x, int y, IndType3 dim)
{
  return x + (int)dim.x * y;
}

/** \brief Compute linear index from 3-d point in grid defined by dim */
__inline__ __device__ __host__ int computeInd32Lin(IndType3 sector,
                                                   gpuNUFFT::Dimensions dim)
{
  return (int)(sector.x + dim.width * (sector.y + dim.height * sector.z));
}

/** \brief Compute linear index from 2-d point in grid defined by dim */
__inline__ __device__ __host__ int computeInd22Lin(IndType2 sector,
                                                   gpuNUFFT::Dimensions dim)
{
  return (int)(sector.x + dim.width * sector.y);
}

#endif
