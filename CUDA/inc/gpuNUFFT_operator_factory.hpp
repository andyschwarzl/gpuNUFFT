#ifndef GPUNUFFT_OPERATOR_FACTORY_H_INCLUDED
#define GPUNUFFT_OPERATOR_FACTORY_H_INCLUDED

#include "config.hpp"
#include "gpuNUFFT_operator.hpp"
#include "balanced_gpuNUFFT_operator.hpp"
#include "texture_gpuNUFFT_operator.hpp"
#include "balanced_texture_gpuNUFFT_operator.hpp"
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <string>
#include <cmath>

#include "cuda_utils.hpp"
#include "precomp_utils.hpp"

namespace gpuNUFFT
{
  // GpuNUFFTOperatorFactory
  // 
  // Manages the initialization of the GpuNUFFT Operator.
  // Distinguishes between two cases:
  //
  // * new calculation of "data - sector" mapping, sorting etc.
  //  
  // * reuse of previously calculated mapping
  // 
  class GpuNUFFTOperatorFactory
  {
  public:
    GpuNUFFTOperatorFactory() 
      : useTextures(true),useGpu(true), balanceWorkload(true)
    {
    }
    
    GpuNUFFTOperatorFactory(const bool useTextures) 
      : useTextures(useTextures), useGpu(true), balanceWorkload(true)
    {
    }

    GpuNUFFTOperatorFactory(const bool useTextures,const bool useGpu) 
      : useTextures(useTextures), useGpu(useGpu), balanceWorkload(true)
    {
    }

    GpuNUFFTOperatorFactory(const bool useTextures,const bool useGpu, bool balanceWorkload) 
      : useTextures(useTextures), useGpu(useGpu), balanceWorkload(balanceWorkload)
    {
    }

    ~GpuNUFFTOperatorFactory()
    {
    }

    GpuNUFFTOperator* createGpuNUFFTOperator(Array<DType>& kSpaceTraj, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    GpuNUFFTOperator* createGpuNUFFTOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    GpuNUFFTOperator* createGpuNUFFTOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    GpuNUFFTOperator* loadPrecomputedGpuNUFFTOperator(Array<DType>& kSpaceTraj, Array<IndType>& dataIndices, Array<IndType>& sectorDataCount,gpuNUFFT::Array<IndType2>& sectorProcessingOrder,Array<IndType>& sectorCenters, Array<DType>& densCompData, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    GpuNUFFTOperator* loadPrecomputedGpuNUFFTOperator(Array<DType>& kSpaceTraj, Array<IndType>& dataIndices, Array<IndType>& sectorDataCount,gpuNUFFT::Array<IndType2>& sectorProcessingOrder,Array<IndType>& sectorCenters, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    void setUseTextures(bool useTextures);

    void setBalanceWorkload(bool balanceWorkload);
    
  protected:
    Array<IndType> assignSectors(GpuNUFFTOperator* gpuNUFFTOp, Array<DType>& kSpaceTraj);

    template <typename T> 
    Array<T> initLinArray(IndType arrCount);

    virtual Array<IndType> initDataIndices(GpuNUFFTOperator* gpuNUFFTOp, IndType coordCnt);
    virtual Array<IndType> initSectorDataCount(GpuNUFFTOperator* gpuNUFFTOp, IndType coordCnt);
    virtual Array<IndType2> initSectorProcessingOrder(GpuNUFFTOperator* gpuNUFFTOp, IndType sectorCnt);
    virtual Array<DType> initDensData(GpuNUFFTOperator* gpuNUFFTOp, IndType coordCnt);
    virtual Array<DType> initCoordsData(GpuNUFFTOperator* gpuNUFFTOp, IndType coordCnt);
    virtual Array<IndType> initSectorCenters(GpuNUFFTOperator* gpuNUFFTOp, IndType sectorCnt);
    virtual void debug(const std::string& message);

    IndType computeSectorCountPerDimension(IndType dim, IndType sectorWidth);

    Dimensions computeSectorCountPerDimension(Dimensions dim, IndType sectorWidth);

    IndType computeTotalSectorCount(Dimensions dim, IndType sectorWidth);

    template <typename T>
    std::vector<IndPair> sortVector(Array<T> assignedSectors, bool descending=false);

    Array<IndType> computeSectorDataCount(gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp,gpuNUFFT::Array<IndType> assignedSectors);
    
    void computeProcessingOrder(GpuNUFFTOperator* gpuNUFFTOp);

    Array<IndType> computeSectorCenters(GpuNUFFTOperator *gpuNUFFTOp);
    Array<IndType> computeSectorCenters2D(GpuNUFFTOperator *gpuNUFFTOp);

    IndType computeSectorCenter(IndType var, IndType sectorWidth);

    GpuNUFFTOperator* createNewGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims);

  private:
    bool useTextures;

    bool useGpu;

    bool balanceWorkload;
  };

}

#endif //GPUNUFFT_OPERATOR_FACTORY_H_INCLUDED
