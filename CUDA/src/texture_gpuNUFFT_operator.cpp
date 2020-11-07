
#include "texture_gpuNUFFT_operator.hpp"

void gpuNUFFT::TextureGpuNUFFTOperator::initKernel()
{
  IndType kernelSize = (interpolationType > 1)
                           ? calculateKernelSizeLinInt(osf, kernelWidth)
                           : calculateGrid3KernelSize(osf, kernelWidth);
  this->kernel.dim.width = kernelSize;
  this->kernel.dim.height = interpolationType > 1 ? kernelSize : 1;
  this->kernel.dim.depth = interpolationType > 2 ? kernelSize : 1;
  if (this->kernel.data != NULL)
    free(this->kernel.data);
  this->kernel.data = (DType *)calloc(this->kernel.count(), sizeof(DType));

  switch (interpolationType)
  {
  case TEXTURE_LOOKUP:
    load1DKernel(this->kernel.data, (int)kernelSize, (int)kernelWidth, osf);
    break;
  case TEXTURE2D_LOOKUP:
    load2DKernel(this->kernel.data, (int)kernelSize, (int)kernelWidth, osf);
    break;
  case TEXTURE3D_LOOKUP:
    load3DKernel(this->kernel.data, (int)kernelSize, (int)kernelWidth, osf);
    break;
  default:
    load1DKernel(this->kernel.data, (int)kernelSize, (int)kernelWidth, osf);
  }
}

const char *gpuNUFFT::TextureGpuNUFFTOperator::getInterpolationTypeName()
{
  switch (interpolationType)
  {
  case TEXTURE_LOOKUP:
    return "texKERNEL";
  case TEXTURE2D_LOOKUP:
    return "texKERNEL2D";
  case TEXTURE3D_LOOKUP:
    return "texKERNEL3D";
  default:
    return "KERNEL";
  }
}

gpuNUFFT::GpuNUFFTInfo *
gpuNUFFT::TextureGpuNUFFTOperator::initAndCopyGpuNUFFTInfo(int n_coils_cc)
{
  gpuNUFFT::GpuNUFFTInfo *gi_host = initGpuNUFFTInfo(n_coils_cc);

  gi_host->interpolationType = interpolationType;
  gi_host->sectorsToProcess = gi_host->sector_count;

  if (DEBUG)
    printf("copy GpuNUFFT Info to symbol memory... size = %lu \n",
      (SizeType)sizeof(gpuNUFFT::GpuNUFFTInfo));

  initConstSymbol("GI", gi_host, sizeof(gpuNUFFT::GpuNUFFTInfo));

  if (DEBUG)
    printf("...done!\n");
  return gi_host;
}

void gpuNUFFT::TextureGpuNUFFTOperator::adjConvolution(
    DType2 *data_d, DType *crds_d, CufftType *gdata_d, DType *kernel_d,
    IndType *sectors_d, IndType *sector_centers_d,
    gpuNUFFT::GpuNUFFTInfo *gi_host)
{
  bindTo1DTexture("texDATA", data_d,
                  this->kSpaceTraj.count() * gi_host->n_coils_cc);

  performTextureConvolution(data_d, crds_d, gdata_d, kernel_d, sectors_d,
                            sector_centers_d, gi_host);

  unbindTexture("texDATA");
}

void gpuNUFFT::TextureGpuNUFFTOperator::forwardConvolution(
    CufftType *data_d, DType *crds_d, CufftType *gdata_d, DType *kernel_d,
    IndType *sectors_d, IndType *sector_centers_d,
    gpuNUFFT::GpuNUFFTInfo *gi_host)
{
  bindTo1DTexture("texGDATA", gdata_d,
                  gi_host->grid_width_dim * gi_host->n_coils_cc);

  performTextureForwardConvolution(data_d, crds_d, gdata_d, kernel_d, sectors_d,
                                   sector_centers_d, gi_host);

  unbindTexture("texGDATA");
}

void gpuNUFFT::TextureGpuNUFFTOperator::initLookupTable()
{
  initTexture(getInterpolationTypeName(), &kernel_d, this->kernel);
}

void gpuNUFFT::TextureGpuNUFFTOperator::freeLookupTable()
{
  if (kernel_d != NULL)
    freeTexture(getInterpolationTypeName(), kernel_d);
}
