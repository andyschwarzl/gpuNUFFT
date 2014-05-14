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
  DType* trajSorted,
  DType* densCompData,
  DType* densData);

void selectOrderedGPU(DType2* data_d, 
  IndType* data_indices_d, 
  DType2* data_sorted_d,
  int N);

void writeOrderedGPU(GriddingND::Array<DType2>& destArray,
  GriddingND::Array<IndType> dataIndices,
  CufftType* data_sorted_d,
  int offset);

#endif
