#ifndef PRECOMP_KERNELS_H
#define PRECOMP_KERNELS_H

#include "cuda_utils.hpp"

// GPU Kernel for Precomputation
// Sector Assignment 
//
void performPrecomputation( DType2* data_d
                            DType2* assignedSectors
						              );

#endif
