#ifndef PRECOMP_UTILS_H
#define PRECOMP_UTILS_H

#include "cuda_utils.hpp"
#include "gridding_operator.hpp"
#include <cmath>

// UTIL functions for precomputation
//
__inline__ __device__ __host__ IndType computeSectorMapping(DType coord, IndType sectorCount)
{
  IndType sector = (IndType)std::floor(static_cast<DType>(coord + 0.5) * sectorCount);
  if (sector >= sectorCount) 
    sector = sectorCount -1;
  if (sector < 0)
    sector = 0;
  return sector;
}

__inline__ __device__ __host__ IndType3 computeSectorMapping(DType3 coord, GriddingND::Dimensions sectorDims)
{
  IndType3 sector;
  sector.x = computeSectorMapping(coord.x,sectorDims.width);
  sector.y  = computeSectorMapping(coord.y,sectorDims.height);
  sector.z  = computeSectorMapping(coord.z,sectorDims.depth);
  return sector;
}

__inline__ __device__ __host__ IndType2 computeSectorMapping(DType2 coord, GriddingND::Dimensions sectorDims)
{
  IndType2 sector;
  sector.x = computeSectorMapping(coord.x,sectorDims.width);
  sector.y  = computeSectorMapping(coord.y,sectorDims.height);
  return sector;
}

__inline__ __device__ __host__ IndType computeXYZ2Lin(IndType x, IndType y, IndType z, GriddingND::Dimensions dim)
{
  return x + dim.height * (y + dim.width * z);
}

__inline__ __device__ __host__ IndType computeXY2Lin(IndType x, IndType y, GriddingND::Dimensions dim)
{
  return x + dim.height * y;
}

__inline__ __device__ __host__ IndType computeInd32Lin(IndType3 sector, GriddingND::Dimensions dim)
{
  return sector.x + dim.height * (sector.y + dim.width * sector.z);
}

__inline__ __device__ __host__ IndType computeInd22Lin(IndType2 sector, GriddingND::Dimensions dim)
{
  return sector.x + dim.height * sector.y ;
}

#endif
