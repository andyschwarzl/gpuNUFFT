#ifndef PRECOMP_UTILS_H
#define PRECOMP_UTILS_H

#include "cuda_utils.hpp"
#include "gridding_operator.hpp"
#include <cmath>

// UTIL functions for precomputation
// 
//
__inline__ __device__ __host__ IndType computeSectorMapping(DType coord, IndType sectorCount)
{
  int sector = (int)std::floor((coord + 0.5) * sectorCount);
  if (sector >= (int)(sectorCount)) 
    sector = (int)(sectorCount -1);
  if (sector < 0)
    sector = 0;
  return sector;
}

// Compute Sector Mapping with the same resolution as in resolutionSectorCount
// but limit it to dimension in sectorCount
// Used when z dimension is not the same as in x and y
//
__inline__ __device__ __host__ IndType computeSectorMapping(DType coord, IndType sectorCount, IndType resolutionSectorCount)
{
  double sectorKdim = 1.0 / resolutionSectorCount;
  //int offset = (int)std::ceil((resolutionSectorCount - sectorCount)/2.0);
  double offset = ((resolutionSectorCount-sectorCount)/2.0)*sectorKdim;

  int sector = (int)std::floor((coord + 0.5 - offset) * resolutionSectorCount);
  if (sector >= (int)(sectorCount)) 
    sector = (int)(sectorCount-1);
  if (sector < 0)
    sector = 0;
  return sector;
}


__inline__ __device__ __host__ IndType3 computeSectorMapping(DType3 coord, GriddingND::Dimensions sectorDims)
{
  IndType3 sector;
  sector.x = computeSectorMapping(coord.x,sectorDims.width);
  sector.y  = computeSectorMapping(coord.y,sectorDims.height);
  sector.z  = computeSectorMapping(coord.z,sectorDims.depth,sectorDims.width);
  return sector;
}

__inline__ __device__ __host__ IndType2 computeSectorMapping(DType2 coord, GriddingND::Dimensions sectorDims)
{
  IndType2 sector;
  sector.x = computeSectorMapping(coord.x,sectorDims.width);
  sector.y  = computeSectorMapping(coord.y,sectorDims.height);
  return sector;
}

__inline__ __device__ __host__ int computeXYZ2Lin(int x, int y, int z, GriddingND::Dimensions dim)
{
  return x + dim.height * (y + dim.width * z);
}

__inline__ __device__ __host__ int computeXYZ2Lin(int x, int y, int z, IndType3 dim)
{
  return x + dim.y * (y + dim.x * z);
}

__inline__ __device__ __host__ int computeXY2Lin(int x, int y, GriddingND::Dimensions dim)
{
  return x + dim.height * y;
}

__inline__ __device__ __host__ int computeXY2Lin(int x, int y, IndType3 dim)
{
  return x + dim.y * y;
}

__inline__ __device__ __host__ int computeInd32Lin(IndType3 sector, GriddingND::Dimensions dim)
{
  return sector.x + dim.height * (sector.y + dim.width * sector.z);
}

__inline__ __device__ __host__ int computeInd22Lin(IndType2 sector, GriddingND::Dimensions dim)
{
  return sector.x + dim.height * sector.y ;
}

#endif
