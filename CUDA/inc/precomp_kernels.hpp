#ifndef PRECOMP_KERNELS_H
#define PRECOMP_KERNELS_H

#include "cuda_utils.hpp"
#include "gpuNUFFT_operator.hpp"
#include "precomp_utils.hpp"
#include <vector>

/**
  * @file
  *
  * \brief GPU Kernels for Precomputation (sector assignment)
  */

/**
  * \brief Perform assign sectors operation on GPU
  *
  * CUDA function prototype
  */
void assignSectorsGPU(gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp,
                      gpuNUFFT::Array<DType> &kSpaceTraj,
                      IndType *assignedSectors);

/**
  * \brief Select arrays by sort order previously computed
  *
  * CUDA function prototype
  */
void sortArrays(gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp,
                std::vector<gpuNUFFT::IndPair> assignedSectorsAndIndicesSorted,
                IndType *assignedSectors, IndType *dataIndices,
                gpuNUFFT::Array<DType> &kSpaceTraj, DType *trajSorted,
                DType *densCompData, DType *densData);

/**
  * \brief Select input array in order defined by data_indices.
  *
  * CUDA function prototype
  *
  * @param data_d Unsorted data array
  * @param data_indidces_d Indices of sort order
  * @param data_sorted_d Sorted data array
  * @param N count of elements
  */
void selectOrderedGPU(DType2 *data_d, IndType *data_indices_d,
                      DType2 *data_sorted_d, int N, int n_coils_cc = 1);

/**
  * \brief Write from sorted input array to unsorted data array.
  *
  * CUDA function prototype
  *
  * @param data_sorted_d Sorted data array
  * @param data_indidces_d Indices of sort order
  * @param data_d Unsorted data array
  * @param N count of elements
  */
void writeOrderedGPU(DType2 *data_sorted_d, IndType *data_indices_d,
                     CufftType *data_d, int N, int n_coils_cc = 1);

#endif
