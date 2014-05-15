#ifndef TEXTURE_GRIDDING_OPERATOR_H_INCLUDED
#define TEXTURE_GRIDDING_OPERATOR_H_INCLUDED

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
     GpuNUFFTOperator(kernelWidth,sectorWidth,osf,imgDims,false,TEXTURE),interpolationType(TEXTURE_LOOKUP)
    {
      initKernel();	
    }

    ~TextureGpuNUFFTOperator()
    {
    }

    OperatorType getType() {return gpuNUFFT::TEXTURE;}
    // OPERATIONS
  private:
    void initKernel();

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
    
    void initLookupTable();
    void freeLookupTable();

		cudaArray* kernel_d;
    InterpolationType interpolationType;
    const char* getInterpolationTypeName();

  };
}

#endif //TEXTURE_GRIDDING_OPERATOR_H_INCLUDED
