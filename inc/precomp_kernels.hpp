#ifndef PRECOMP_KERNELS_H
#define PRECOMP_KERNELS_H

#include "cuda_utils.hpp"
#include "gridding_operator.hpp"
#include "precomp_utils.hpp"

// GPU Kernel for Precomputation
// Sector Assignment 
//
void assignSectorsGPU(GriddingND::GriddingOperator* griddingOp, GriddingND::Array<DType>& kSpaceTraj, IndType* assignedSectors);

#endif
