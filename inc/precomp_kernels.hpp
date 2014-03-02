#ifndef PRECOMP_KERNELS_H
#define PRECOMP_KERNELS_H

#include "cuda_utils.hpp"
#include "gridding_operator.hpp"
#include "precomp_utils.hpp"
#include <vector>

// GPU Kernel for Precomputation
// Sector Assignment 
//
void assignSectorsGPU(GriddingND::GriddingOperator* griddingOp, 
                      GriddingND::Array<DType>& kSpaceTraj, 
                      IndType* assignedSectors);

void sortArrays(GriddingND::GriddingOperator* griddingOp, 
                      std::vector<GriddingND::IndPair> assignedSectorsAndIndicesSorted,
                      IndType* assignedSectors, 
                      IndType* dataIndices,
                      GriddingND::Array<DType>& kSpaceTraj,
                      DType* trajectory,
                      DType* densCompData,
                      DType* densData);

#endif
