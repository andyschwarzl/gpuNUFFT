
#include "texture_gridding_operator.hpp"

void GriddingND::TextureGriddingOperator::initKernel()
{
  IndType kernelSize = (interpolationType > 1) ? calculateKernelSizeLinInt(osf, kernelWidth/2.0f) : calculateGrid3KernelSize(osf, kernelWidth/2.0f);
  this->kernel.dim.width = kernelSize;
  this->kernel.dim.height = interpolationType > 1 ? kernelSize : 1;
  this->kernel.dim.depth = interpolationType > 2 ? kernelSize : 1;
  this->kernel.data = (DType*) calloc(this->kernel.count(),sizeof(DType));

  switch (interpolationType)
  {
  case 1:   load1DKernel(this->kernel.data,(int)kernelSize,(int)kernelWidth,osf);break;
  case 2:   load2DKernel(this->kernel.data,(int)kernelSize,(int)kernelWidth,osf);break;
  case 3:   load3DKernel(this->kernel.data,(int)kernelSize,(int)kernelWidth,osf);break;
  }

}

const char* GriddingND::TextureGriddingOperator::getInterpolationTypeName()
{
  switch (interpolationType)
  {
  case 1:   return "texKERNEL";
  case 2:   return "texKERNEL2D";
  case 3:   return "texKERNEL3D";
  }

}

GriddingND::GriddingInfo* GriddingND::TextureGriddingOperator::initAndCopyGriddingInfo()
{
  GriddingND::GriddingInfo* gi_host = initAndCopyGriddingInfo();

  gi_host->interpolationType = interpolationType;

  if (DEBUG)
    printf("copy Gridding Info to symbol memory... size = %ld \n",sizeof(GriddingND::GriddingInfo));

  initConstSymbol("GI",gi_host,sizeof(GriddingND::GriddingInfo));

  if (DEBUG)
    printf("...done!\n");
  return gi_host;
}

void GriddingND::TextureGriddingOperator::adjConvolution(DType2* data_d, 
      DType* crds_d, 
      CufftType* gdata_d,
      DType* kernel_d, 
      IndType* sectors_d, 
      IndType* sector_centers_d,
  GriddingND::GriddingInfo* gi_host)
{
  performTextureConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,gi_host);
}

void GriddingND::TextureGriddingOperator::forwardConvolution(CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  GriddingND::GriddingInfo* gi_host)
{
  performTextureForwardConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,gi_host);
}

void GriddingND::TextureGriddingOperator::initLookupTable()
{
  initTexture(getInterpolationTypeName(),&kernel_d,this->kernel);
}

void GriddingND::TextureGriddingOperator::freeLookupTable()
{
	freeTexture(getInterpolationTypeName(),kernel_d);
}