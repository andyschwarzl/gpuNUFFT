#ifndef BALANCED_GPUNUFFT_OPERATOR_H_INCLUDED
#define BALANCED_GPUNUFFT_OPERATOR_H_INCLUDED

#include "gpuNUFFT_types.hpp"
#include "gpuNUFFT_operator.hpp"
#include "balanced_operator.hpp"

namespace gpuNUFFT
{
/**
* \brief Balanced GpuNUFFTOperator inherited from gpuNUFFT::GpuNUFFTOperator
*
* Changes the behaviour of the default GpuNUFFTOperator by balancing the
* work load by sector to a maximum amount of samples per sector
*(MAXIMUM_PAYLOAD).
* Thus, sectors with a high density of data points are split into multiple ones,
* which are processed in parallel.
*
*/
class BalancedGpuNUFFTOperator : public GpuNUFFTOperator,
                                 public BalancedOperator
{
 public:
  BalancedGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf,
    Dimensions imgDims, bool matlabSharedMem = false)
    : GpuNUFFTOperator(kernelWidth, sectorWidth, osf, imgDims, true, BALANCED, matlabSharedMem)
  {
  }

  ~BalancedGpuNUFFTOperator()
  {
      freeLocalMemberArray(this->sectorProcessingOrder.data);
  }

  Array<IndType2> getSectorProcessingOrder()
  {
    return this->sectorProcessingOrder;
  }
  void setSectorProcessingOrder(Array<IndType2> sectorProcessingOrder)
  {
    this->sectorProcessingOrder = sectorProcessingOrder;
  }

  // OPERATIONS
  void performGpuNUFFTAdj(Array<DType2> kspaceData, Array<CufftType> &imgData,
                          GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);
  void performGpuNUFFTAdj(GpuArray<DType2> kspaceData_gpu,
                          GpuArray<CufftType> &imgData_gpu,
                          GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);

  void performForwardGpuNUFFT(Array<DType2> imgData,
                              Array<CufftType> &kspaceData,
                              GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);
  void performForwardGpuNUFFT(GpuArray<DType2> imgData_gpu,
                              GpuArray<CufftType> &kspaceData,
                              GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);

  OperatorType getType()
  {
    return gpuNUFFT::BALANCED;
  }

 protected:
  // sectorProcessingOrder
  Array<IndType2> sectorProcessingOrder;

  IndType2 *sector_processing_order_d;

  GpuNUFFTInfo *initAndCopyGpuNUFFTInfo(int n_coils_cc = 1);

  void adjConvolution(DType2 *data_d, DType *crds_d, CufftType *gdata_d,
                      DType *kernel_d, IndType *sectors_d,
                      IndType *sector_centers_d,
                      gpuNUFFT::GpuNUFFTInfo *gi_host);

  void forwardConvolution(CufftType *data_d, DType *crds_d, CufftType *gdata_d,
                          DType *kernel_d, IndType *sectors_d,
                          IndType *sector_centers_d,
                          gpuNUFFT::GpuNUFFTInfo *gi_host);
};
}

#endif  // BALANCED_GPUNUFFT_OPERATOR_H_INCLUDED
