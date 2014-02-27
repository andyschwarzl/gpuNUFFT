#ifndef PRECOMP_UTILS_H
#define PRECOMP_UTILS_H

#include "cuda_utils.hpp"
#include "gridding_operator.hpp"

// UTIL functions for precomputation
//
    IndType computeSectorMapping(DType coord, IndType sectorCount);

    IndType3 computeSectorMapping(DType3 coord, GriddingND::Dimensions sectorDims);

    IndType2 computeSectorMapping(DType2 coord, GriddingND::Dimensions sectorDims);

    IndType computeXYZ2Lin(IndType x, IndType y, IndType z, GriddingND::Dimensions dim);

    IndType computeXY2Lin(IndType x, IndType y, GriddingND::Dimensions dim);

    IndType computeInd32Lin(IndType3 sector, GriddingND::Dimensions dim);

    IndType computeInd22Lin(IndType2 sector, GriddingND::Dimensions dim);
    
#endif
