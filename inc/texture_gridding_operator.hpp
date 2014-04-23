#ifndef TEXTURE_GRIDDING_OPERATOR_H_INCLUDED
#define TEXTURE_GRIDDING_OPERATOR_H_INCLUDED

#include "gridding_types.hpp"
#include "gridding_operator.hpp"

namespace GriddingND
{
  class TextureGriddingOperator : public GriddingOperator
  {
  public:
    
    TextureGriddingOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims,InterpolationType interpolationType): 
    GriddingOperator(kernelWidth,sectorWidth,osf,imgDims,false,TEXTURE),interpolationType(interpolationType)
    {
        initKernel();	
    }

    TextureGriddingOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims): 
     GriddingOperator(kernelWidth,sectorWidth,osf,imgDims,false,TEXTURE),interpolationType(TEXTURE_LOOKUP)
    {
      initKernel();	
    }

    ~TextureGriddingOperator()
    {
    }

    OperatorType getType() {return OperatorType::TEXTURE;}
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
