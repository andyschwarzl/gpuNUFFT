#ifndef TEXTURE_GRIDDING_OPERATOR_H_INCLUDED
#define TEXTURE_GRIDDING_OPERATOR_H_INCLUDED

#include "gridding_types.hpp"
#include "gridding_operator.hpp"

namespace GriddingND
{
  class TextureGriddingOperator : public GriddingOperator
  {
  public:

    TextureGriddingOperator():
        interpolationType(InterpolationType::TEXTURE_LOOKUP)
    {
    }

    TextureGriddingOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims): 
    GriddingOperator(kernelWidth,sectorWidth,osf,imgDims),interpolationType(InterpolationType::TEXTURE_LOOKUP)
    {
    }

    ~TextureGriddingOperator()
    {
    }

    // OPERATIONS

  private:
    void initKernel();

    GriddingInfo* initAndCopyGriddingInfo();

    void adjConvolution(DType2* data_d, 
      DType* crds_d, 
      CufftType* gdata_d,
      DType* kernel_d, 
      IndType* sectors_d, 
      IndType* sector_centers_d,
      GriddingND::GriddingInfo* gi_host);
    void forwardConvolution(CufftType*		data_d, 
      DType*			crds_d, 
      CufftType*		gdata_d,
      DType*			kernel_d, 
      IndType*		sectors_d, 
      IndType*		sector_centers_d,
      GriddingND::GriddingInfo* gi_host);
    
    void initLookupTable();
    void freeLookupTable();

		cudaArray* kernel_d;
    InterpolationType interpolationType;
    const char* getInterpolationTypeName();
  };
}

#endif //TEXTURE_GRIDDING_OPERATOR_H_INCLUDED
