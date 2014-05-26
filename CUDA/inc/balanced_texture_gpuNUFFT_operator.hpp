#ifndef BALANCED_TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED
#define BALANCED_TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED

#include "gpuNUFFT_types.hpp"
#include "texture_gpuNUFFT_operator.hpp"
#include "balanced_operator.hpp"

namespace gpuNUFFT
{
  class BalancedTextureGpuNUFFTOperator : public TextureGpuNUFFTOperator, public BalancedOperator
  {
  public:
    
    BalancedTextureGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims,InterpolationType interpolationType): 
    TextureGpuNUFFTOperator(kernelWidth,sectorWidth,osf,imgDims,interpolationType)
    {
      initKernel();	
    }

    BalancedTextureGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims): 
     TextureGpuNUFFTOperator(kernelWidth,sectorWidth,osf,imgDims)
    {
      initKernel();	
    }

    ~BalancedTextureGpuNUFFTOperator()
    {
    }

    // OPERATIONS
    void performGpuNUFFTAdj(Array<DType2> kspaceData, Array<CufftType>& imgData, GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);
    void performForwardGpuNUFFT(Array<DType2> imgData,Array<CufftType>& kspaceData, GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);
    
    //Getter and Setter for Processing Order
    Array<IndType2>  getSectorProcessingOrder(){return this->sectorProcessingOrder;}
    void setSectorProcessingOrder(Array<IndType2> sectorProcessingOrder)	{this->sectorProcessingOrder = sectorProcessingOrder;}

    OperatorType getType() {return gpuNUFFT::BALANCED_TEXTURE;}
    // OPERATIONS
  private:
    GpuNUFFTInfo* initAndCopyGpuNUFFTInfo();

	  // sectorProcessingOrder
    Array<IndType2> sectorProcessingOrder;
    
    IndType2* sector_processing_order_d;

    void adjConvolution(DType2* data_d, 
      DType* crds_d, 
      CufftType* gdata_d,
      DType* kernel_d, 
      IndType* sectors_d, 
      IndType* sector_centers_d,
      gpuNUFFT::GpuNUFFTInfo* gi_host);

    void forwardConvolution(CufftType*		data_d, 
      DType*			crds_d, 
      CufftType*		gdata_d,
      DType*			kernel_d, 
      IndType*		sectors_d, 
      IndType*		sector_centers_d,
      gpuNUFFT::GpuNUFFTInfo* gi_host);
    
  };
}

#endif //BALANCED_TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED
