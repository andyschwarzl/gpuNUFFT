#ifndef BALANCED_TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED
#define BALANCED_TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED

#include "gpuNUFFT_types.hpp"
#include "texture_gpuNUFFT_operator.hpp"
#include "balanced_operator.hpp"

namespace gpuNUFFT
{
/**
  * \brief GpuNUFFTOperator with load balancing and texture memory lookup
  *
  * Changes the behaviour of the default GpuNUFFTOperator by balancing the
  * work load by sector to a maximum amount of samples per sector
  *(MAXIMUM_PAYLOAD).
  * Thus, sectors with a high density of data points are split into multiple
  *ones,
  * which are processed in parallel.
  *
  * Furthermore, the kernel interpolation is performed by using gpu texture
  *memory.
  *
  */
class BalancedTextureGpuNUFFTOperator : public TextureGpuNUFFTOperator,
                                        public BalancedOperator
{
 public:
  BalancedTextureGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth,
                                  DType osf, Dimensions imgDims,
                                  InterpolationType interpolationType = TEXTURE2D_LOOKUP,
                                  bool matlabSharedMem = false)
    : TextureGpuNUFFTOperator(kernelWidth, sectorWidth, osf, imgDims,
                              interpolationType, matlabSharedMem)
  {
  }

  ~BalancedTextureGpuNUFFTOperator()
  {
     freeLocalMemberArray(this->sectorProcessingOrder.data);
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

  // Getter and Setter for Processing Order
  Array<IndType2> getSectorProcessingOrder()
  {
    return this->sectorProcessingOrder;
  }
  void setSectorProcessingOrder(Array<IndType2> sectorProcessingOrder)
  {
    this->sectorProcessingOrder = sectorProcessingOrder;
  }

  OperatorType getType()
  {
    return gpuNUFFT::BALANCED_TEXTURE;
  }
  // OPERATIONS
 private:
  GpuNUFFTInfo *initAndCopyGpuNUFFTInfo(int n_coils_cc = 1);

  // sectorProcessingOrder
  Array<IndType2> sectorProcessingOrder;

  IndType2 *sector_processing_order_d;

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

#endif  // BALANCED_TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED
