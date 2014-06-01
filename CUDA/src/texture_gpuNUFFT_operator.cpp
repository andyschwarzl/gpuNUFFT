
#include "texture_gpuNUFFT_operator.hpp"

void gpuNUFFT::TextureGpuNUFFTOperator::initKernel()
{
  IndType kernelSize = (interpolationType > 1) ? calculateKernelSizeLinInt(osf, kernelWidth/2.0f) : calculateGrid3KernelSize(osf, kernelWidth/2.0f);
  this->kernel.dim.width = kernelSize;
  this->kernel.dim.height = interpolationType > 1 ? kernelSize : 1;
  this->kernel.dim.depth = interpolationType > 2 ? kernelSize : 1;
  this->kernel.data = (DType*) calloc(this->kernel.count(),sizeof(DType));

  switch (interpolationType)
  {
    case TEXTURE_LOOKUP:   load1DKernel(this->kernel.data,(int)kernelSize,(int)kernelWidth,osf);break;
    case TEXTURE2D_LOOKUP:   load2DKernel(this->kernel.data,(int)kernelSize,(int)kernelWidth,osf);break;
    case TEXTURE3D_LOOKUP:   load3DKernel(this->kernel.data,(int)kernelSize,(int)kernelWidth,osf);break;
    default: load1DKernel(this->kernel.data,(int)kernelSize,(int)kernelWidth,osf);
  }

}

const char* gpuNUFFT::TextureGpuNUFFTOperator::getInterpolationTypeName()
{
  switch (interpolationType)
  {
  case TEXTURE_LOOKUP:   return "texKERNEL";
  case TEXTURE2D_LOOKUP:   return "texKERNEL2D";
  case TEXTURE3D_LOOKUP:   return "texKERNEL3D";
  default: return "KERNEL";
  }

}

gpuNUFFT::GpuNUFFTInfo* gpuNUFFT::TextureGpuNUFFTOperator::initAndCopyGpuNUFFTInfo()
{
  gpuNUFFT::GpuNUFFTInfo* gi_host = initGpuNUFFTInfo();

  gi_host->interpolationType = interpolationType;

  if (DEBUG)
    printf("copy GpuNUFFT Info to symbol memory... size = %ld \n",sizeof(gpuNUFFT::GpuNUFFTInfo));

  initConstSymbol("GI",gi_host,sizeof(gpuNUFFT::GpuNUFFTInfo));

  if (DEBUG)
    printf("...done!\n");
  return gi_host;
}

void gpuNUFFT::TextureGpuNUFFTOperator::adjConvolution(DType2* data_d, 
      DType* crds_d, 
      CufftType* gdata_d,
      DType* kernel_d, 
      IndType* sectors_d, 
      IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  bindTo1DTexture("texDATA",data_d,this->kSpaceTraj.count());
  
  performTextureConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,minmax_bounds_d,gi_host);

  unbindTexture("texDATA");
}

void gpuNUFFT::TextureGpuNUFFTOperator::forwardConvolution(CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  bindTo1DTexture("texGDATA",gdata_d,gi_host->grid_width_dim);

  performTextureForwardConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,minmax_bounds_d,gi_host);

  unbindTexture("texGDATA");
}

void gpuNUFFT::TextureGpuNUFFTOperator::initLookupTable()
{
  initTexture(getInterpolationTypeName(),&kernel_d,this->kernel);
}

void gpuNUFFT::TextureGpuNUFFTOperator::freeLookupTable()
{
	freeTexture(getInterpolationTypeName(),kernel_d);
}

void gpuNUFFT::TextureGpuNUFFTOperator::initMinMaxBounds()
{
  int minmax_count = gi_host->data_count*getImageDimensionCount();
  if (DEBUG)
    printf("allocate minmax bounds data of size %d...\n",minmax_count);
  allocateDeviceMem<uchar2>(&minmax_bounds_d,minmax_count);
  computeMinMaxBounds(crds_d,sectors_d,sector_centers_d,minmax_bounds_d,gi_host);

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error at adj thread synchronization minmax %s\n",cudaGetErrorString(cudaGetLastError()));
}

void gpuNUFFT::TextureGpuNUFFTOperator::freeMinMaxBounds()
{
  freeTotalDeviceMemory(minmax_bounds_d,NULL);//NULL as stop token
}

void gpuNUFFT::TextureGpuNUFFTOperator::performGpuNUFFTAdj(gpuNUFFT::Array<DType2> kspaceData, gpuNUFFT::Array<CufftType>& imgData, GpuNUFFTOutput gpuNUFFTOut)
{ 
  //TODO
  initDeviceMemory(kspaceData.dim.channels);

  initMinMaxBounds();

  GpuNUFFTOperator::performGpuNUFFTAdj(kspaceData,imgData,gpuNUFFTOut);

  freeMinMaxBounds();
}

void gpuNUFFT::TextureGpuNUFFTOperator::performForwardGpuNUFFT(gpuNUFFT::Array<DType2> imgData,gpuNUFFT::Array<CufftType>& kspaceData, GpuNUFFTOutput gpuNUFFTOut)
{
  initDeviceMemory(kspaceData.dim.channels);

  initMinMaxBounds();

  GpuNUFFTOperator::performForwardGpuNUFFT(imgData,kspaceData,gpuNUFFTOut);

  freeMinMaxBounds();
}