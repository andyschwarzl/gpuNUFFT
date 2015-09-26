#ifndef GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED
#define GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED

#include "config.hpp"
#include "gpuNUFFT_operator_factory.hpp"
#include <algorithm>  // std::sort
#include <vector>     // std::vector
#include "matlab_helper.h"
#include <string>

#include "mex.h"
#include "matrix.h"

namespace gpuNUFFT
{
/** \brief GpuNUFFTOperatorFactory Decorator
 *
 * Takes care of proper array initialization for MATLAB
 * output arrays.
 *
 */
class GpuNUFFTOperatorMatlabFactory : public GpuNUFFTOperatorFactory
{
 public:
  GpuNUFFTOperatorMatlabFactory() : GpuNUFFTOperatorFactory()
  {
  }

  GpuNUFFTOperatorMatlabFactory(const bool useTextures, const bool useGpu,
                                const bool balanceWorkload)
    : GpuNUFFTOperatorFactory(useTextures, useGpu, balanceWorkload)
  {
  }

  GpuNUFFTOperatorMatlabFactory(const bool useTextures, const bool useGpu)
    : GpuNUFFTOperatorFactory(useTextures, useGpu)
  {
  }

  GpuNUFFTOperatorMatlabFactory(const bool useTextures)
    : GpuNUFFTOperatorFactory(useTextures)
  {
  }

  ~GpuNUFFTOperatorMatlabFactory()
  {
  }

  GpuNUFFTOperator *
  createGpuNUFFTOperator(Array<DType> &kSpaceTraj, Array<DType> &densCompData,
                         Array<DType2> &sensData, const IndType &kernelWidth,
                         const IndType &sectorWidth, const DType &osf,
                         Dimensions &imgDims, mxArray *plhs[]);

 protected:
  Array<IndType> initDataIndices(GpuNUFFTOperator *gpuNUFFTOp,
                                 IndType coordCnt);
  Array<DType> initCoordsData(GpuNUFFTOperator *gpuNUFFTOp, IndType coordCnt);
  Array<IndType> initSectorCenters(GpuNUFFTOperator *gpuNUFFTOp,
                                   IndType sectorCnt);
  Array<IndType> initSectorDataCount(GpuNUFFTOperator *gpuNUFFTOp,
                                     IndType dataCount);
  Array<IndType2> initSectorProcessingOrder(GpuNUFFTOperator *gpuNUFFTOp,
                                            IndType sectorCnt);
  Array<DType> initDensData(GpuNUFFTOperator *gpuNUFFTOp, IndType coordCnt);

  void debug(const std::string &message);

 private:
  mxArray **plhs;
};
}

#endif  // GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED
