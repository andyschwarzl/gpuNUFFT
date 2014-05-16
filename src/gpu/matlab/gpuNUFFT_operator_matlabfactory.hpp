#ifndef GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED
#define GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED

#include "config.hpp"
#include "gpuNUFFT_operator_factory.hpp"
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include "matlab_helper.h"
#include <string>

#include "mex.h"
#include "matrix.h"

namespace gpuNUFFT
{
  // GpuNUFFTOperatorMatlabFactory
  // decorates GpuNUFFTOperatorFactory
  //
  // Manages the initialization of the GpuNUFFT Operator.
  // Distinguishes between two cases:
  //
  // * new calculation of "data - sector" mapping, sorting etc.
  //  
  // * reuse of previously calculated mapping
  // 
  class GpuNUFFTOperatorMatlabFactory : public GpuNUFFTOperatorFactory
  {
  public:

    GpuNUFFTOperatorMatlabFactory():
        GpuNUFFTOperatorFactory()
    {
    }

    GpuNUFFTOperatorMatlabFactory(const InterpolationType interpolationType,const bool useGpu, const bool balanceWorkload) 
      : GpuNUFFTOperatorFactory(interpolationType,useGpu,balanceWorkload)
    {
    }

    GpuNUFFTOperatorMatlabFactory(const InterpolationType interpolationType,const bool useGpu) 
      : GpuNUFFTOperatorFactory(interpolationType,useGpu)
    {
    }

    GpuNUFFTOperatorMatlabFactory(const InterpolationType interpolationType) 
      : GpuNUFFTOperatorFactory(interpolationType)
    {
    }

    ~GpuNUFFTOperatorMatlabFactory()
    {
    }

    GpuNUFFTOperator* createGpuNUFFTOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims,mxArray *plhs[]);

  protected:

    //abstract methods from GpuNUFFTOperatorFactory
    Array<IndType> initDataIndices(GpuNUFFTOperator* gpuNUFFTOp, IndType coordCnt);
    Array<DType> initCoordsData(GpuNUFFTOperator* gpuNUFFTOp, IndType coordCnt);
    Array<IndType> initSectorCenters(GpuNUFFTOperator* gpuNUFFTOp, IndType sectorCnt);
    Array<IndType> initSectorDataCount(GpuNUFFTOperator* gpuNUFFTOp, IndType dataCount);
    Array<IndType2> initSectorProcessingOrder(GpuNUFFTOperator* gpuNUFFTOp, IndType sectorCnt);
    Array<DType> initDensData(GpuNUFFTOperator* gpuNUFFTOp, IndType coordCnt);

    void debug(const std::string& message);

  private:
    mxArray **plhs;

  };

}

#endif //GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED
