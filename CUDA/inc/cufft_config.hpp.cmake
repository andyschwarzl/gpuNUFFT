@WARNING@
#ifndef CUFFT_CONFIG_H
#define CUFFT_CONFIG_H
#include "config.hpp"
#include "cufft.h"

/**
 * @file
 * \brief Definition of types and function pointers used for CUFFT calls.
 *
 * Depends on CMAKE build parameter GPU_DOUBLE_PREC
 *
 */

#ifdef GPU_DOUBLE_PREC
static cufftType_t CufftTransformType = CUFFT_Z2Z;

#ifdef WIN32
static cufftResult(__stdcall *pt2CufftExec)(cufftHandle, cufftDoubleComplex *,
                                            cufftDoubleComplex *,
                                            int) = &cufftExecZ2Z;
#else
static cufftResult (*pt2CufftExec)(cufftHandle, cufftDoubleComplex *,
                                   cufftDoubleComplex *, int) = &cufftExecZ2Z;
#endif
#else
    static cufftType_t CufftTransformType = CUFFT_C2C;

#ifdef WIN32
static cufftResult(__stdcall *pt2CufftExec)(cufftHandle, cufftComplex *,
                                            cufftComplex *,
                                            int) = &cufftExecC2C;
#else
static cufftResult (*pt2CufftExec)(cufftHandle, cufftComplex *, cufftComplex *,
                                   int) = &cufftExecC2C;
#endif
#endif

#endif  // CUFFT_CONFIG_H
