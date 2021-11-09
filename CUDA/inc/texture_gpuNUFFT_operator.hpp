#ifndef TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED
#define TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED

#include <typeinfo>
#include <stdexcept>
#include "gpuNUFFT_types.hpp"
#include "gpuNUFFT_operator.hpp"

namespace gpuNUFFT
{
/**
* \brief GpuNUFFTOperator with texture memory lookup
*
* Changes the behaviour of the default GpuNUFFTOperator by using gpu texture
*memory
* in the kernel interpolation step.
*
*/
class TextureGpuNUFFTOperator : public GpuNUFFTOperator
{
 public:
  TextureGpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf,
                          Dimensions imgDims,
                          InterpolationType interpolationType = TEXTURE2D_LOOKUP,
                          bool matlabSharedMem = false)
    : GpuNUFFTOperator(kernelWidth, sectorWidth, osf, imgDims, false, TEXTURE, matlabSharedMem),
    interpolationType(interpolationType), kernel_d(NULL)
  {
    if (typeid(DType) == typeid(double))
      throw std::runtime_error(
          "Double precision textures are not supported yet!");

    initKernel();
  }

  ~TextureGpuNUFFTOperator()
  {
  }

  virtual OperatorType getType()
  {
    return gpuNUFFT::TEXTURE;
  }

 protected:
  void initKernel();

  cudaArray *kernel_d;
  InterpolationType interpolationType;
  const char *getInterpolationTypeName();

  // OPERATIONS
 private:
  GpuNUFFTInfo *initAndCopyGpuNUFFTInfo(int n_coils_cc = 1);

  virtual void adjConvolution(DType2 *data_d, DType *crds_d, CufftType *gdata_d,
                              DType *kernel_d, IndType *sectors_d,
                              IndType *sector_centers_d,
                              gpuNUFFT::GpuNUFFTInfo *gi_host);
  virtual void forwardConvolution(CufftType *data_d, DType *crds_d,
                                  CufftType *gdata_d, DType *kernel_d,
                                  IndType *sectors_d, IndType *sector_centers_d,
                                  gpuNUFFT::GpuNUFFTInfo *gi_host);

  void initLookupTable();
  void freeLookupTable();
};
}

#endif  // TEXTURE_GPUNUFFT_OPERATOR_H_INCLUDED
