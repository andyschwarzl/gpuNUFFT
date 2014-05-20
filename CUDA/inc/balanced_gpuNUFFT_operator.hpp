#ifndef BALANCED_GPUNUFFT_OPERATOR_H_INCLUDED
#define BALANCED_GPUNUFFT_OPERATOR_H_INCLUDED

#include "gpuNUFFT_types.hpp"
#include "gpuNUFFT_operator.hpp"

namespace gpuNUFFT
{
  class BalancedGpuNUFFTOperator : public GpuNUFFTOperator
  {
  public:

    BalancedGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims): 
    GpuNUFFTOperator(kernelWidth,sectorWidth,osf,imgDims,true,BALANCED)
    {
    }

    ~BalancedGpuNUFFTOperator()
    {
    }

    Array<IndType2>  getSectorProcessingOrder(){return this->sectorProcessingOrder;}
    void setSectorProcessingOrder(Array<IndType2> sectorProcessingOrder)	{this->sectorProcessingOrder = sectorProcessingOrder;}

    // OPERATIONS
    void performGpuNUFFTAdj(Array<DType2> kspaceData, Array<CufftType>& imgData, GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);
    void performForwardGpuNUFFT(Array<DType2> imgData,Array<CufftType>& kspaceData, GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);
        
    OperatorType getType() {return gpuNUFFT::BALANCED;}

  protected:
	
    // sectorProcessingOrder
    Array<IndType2> sectorProcessingOrder;
    
    IndType2* sector_processing_order_d;
    
    GpuNUFFTInfo* initAndCopyGpuNUFFTInfo();

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

#endif //BALANCED_GPUNUFFT_OPERATOR_H_INCLUDED
