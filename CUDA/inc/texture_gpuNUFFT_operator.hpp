#ifndef TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED
#define TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED

#include "gpuNUFFT_types.hpp"
#include "gpuNUFFT_operator.hpp"

namespace gpuNUFFT
{
  class TextureGpuNUFFTOperator : public GpuNUFFTOperator
  {
  public:
    
    TextureGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims,InterpolationType interpolationType): 
    GpuNUFFTOperator(kernelWidth,sectorWidth,osf,imgDims,false,TEXTURE),interpolationType(interpolationType)
    {
      initKernel();	
    }

    TextureGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims): 
    GpuNUFFTOperator(kernelWidth,sectorWidth,osf,imgDims,false,TEXTURE),interpolationType(gpuNUFFT::TEXTURE2D_LOOKUP)
    {
      initKernel();	
    }

    ~TextureGpuNUFFTOperator()
    {
    }

    virtual OperatorType getType() {return gpuNUFFT::TEXTURE;}

  protected:
    void initKernel();
    
		cudaArray* kernel_d;
    InterpolationType interpolationType;
    const char* getInterpolationTypeName();

    // OPERATIONS
  private:
    GpuNUFFTInfo* initAndCopyGpuNUFFTInfo();

    virtual void adjConvolution(DType2* data_d, 
      DType* crds_d, 
      CufftType* gdata_d,
      DType* kernel_d, 
      IndType* sectors_d, 
      IndType* sector_centers_d,
      gpuNUFFT::GpuNUFFTInfo* gi_host);
    virtual void forwardConvolution(CufftType*		data_d, 
      DType*			crds_d, 
      CufftType*		gdata_d,
      DType*			kernel_d, 
      IndType*		sectors_d, 
      IndType*		sector_centers_d,
      gpuNUFFT::GpuNUFFTInfo* gi_host);
    
    void initLookupTable();
    void freeLookupTable();
  };
}

#endif //TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED
