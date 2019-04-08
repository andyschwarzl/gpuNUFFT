
#include "balanced_gpuNUFFT_operator.hpp"

gpuNUFFT::GpuNUFFTInfo *
gpuNUFFT::BalancedGpuNUFFTOperator::initAndCopyGpuNUFFTInfo(int n_coils_cc)
{
  gpuNUFFT::GpuNUFFTInfo *gi_host = initGpuNUFFTInfo(n_coils_cc);

  gi_host->sectorsToProcess = sectorProcessingOrder.count();

  if (DEBUG)
    printf("copy GpuNUFFT Info to symbol memory... size = %lu \n",
           (SizeType)sizeof(gpuNUFFT::GpuNUFFTInfo));

  initConstSymbol("GI", gi_host, sizeof(gpuNUFFT::GpuNUFFTInfo));

  if (DEBUG)
    printf("...done!\n");
  return gi_host;
}

void gpuNUFFT::BalancedGpuNUFFTOperator::adjConvolution(
    DType2 *data_d, DType *crds_d, CufftType *gdata_d, DType *kernel_d,
    IndType *sectors_d, IndType *sector_centers_d,
    gpuNUFFT::GpuNUFFTInfo *gi_host)
{
  performConvolution(data_d, crds_d, gdata_d, kernel_d, sectors_d,
                     sector_processing_order_d, sector_centers_d, gi_host);
}

void gpuNUFFT::BalancedGpuNUFFTOperator::forwardConvolution(
    CufftType *data_d, DType *crds_d, CufftType *gdata_d, DType *kernel_d,
    IndType *sectors_d, IndType *sector_centers_d,
    gpuNUFFT::GpuNUFFTInfo *gi_host)
{
  performForwardConvolution(data_d, crds_d, gdata_d, kernel_d, sectors_d,
                            sector_processing_order_d, sector_centers_d,
                            gi_host);
}

// Adds behaviour of GpuNUFFTOperator by
// adding a sector processing order
void gpuNUFFT::BalancedGpuNUFFTOperator::performGpuNUFFTAdj(
    gpuNUFFT::Array<DType2> kspaceData, gpuNUFFT::Array<CufftType> &imgData,
    GpuNUFFTOutput gpuNUFFTOut)
{
  if (DEBUG)
    printf(
        "BGpuNUFFT: allocate and copy sector processing order of size %d...\n",
        this->sectorProcessingOrder.count());
  allocateAndCopyToDeviceMem<IndType2>(&sector_processing_order_d,
                                       this->sectorProcessingOrder.data,
                                       this->sectorProcessingOrder.count());

  GpuNUFFTOperator::performGpuNUFFTAdj(kspaceData, imgData, gpuNUFFTOut);

  freeTotalDeviceMemory(sector_processing_order_d, NULL);  // NULL as stop token
}

void gpuNUFFT::BalancedGpuNUFFTOperator::performGpuNUFFTAdj(
    GpuArray<DType2> kspaceData, GpuArray<CufftType> &imgData,
    GpuNUFFTOutput gpuNUFFTOut)
{
  if (DEBUG)
    printf(
        "BGpuNUFFT: allocate and copy sector processing order of size %d...\n",
        this->sectorProcessingOrder.count());
  allocateAndCopyToDeviceMem<IndType2>(&sector_processing_order_d,
                                       this->sectorProcessingOrder.data,
                                       this->sectorProcessingOrder.count());

  GpuNUFFTOperator::performGpuNUFFTAdj(kspaceData, imgData, gpuNUFFTOut);

  freeTotalDeviceMemory(sector_processing_order_d, NULL);  // NULL as stop token
}

void gpuNUFFT::BalancedGpuNUFFTOperator::performForwardGpuNUFFT(
    gpuNUFFT::Array<DType2> imgData, gpuNUFFT::Array<CufftType> &kspaceData,
    GpuNUFFTOutput gpuNUFFTOut)
{
  if (DEBUG)
    printf(
        "BGpuNUFFT: allocate and copy sector processing order of size %d...\n",
        this->sectorProcessingOrder.count());
  allocateAndCopyToDeviceMem<IndType2>(&sector_processing_order_d,
                                       this->sectorProcessingOrder.data,
                                       this->sectorProcessingOrder.count());

  GpuNUFFTOperator::performForwardGpuNUFFT(imgData, kspaceData, gpuNUFFTOut);

  freeTotalDeviceMemory(sector_processing_order_d, NULL);  // NULL as stop token
}

void gpuNUFFT::BalancedGpuNUFFTOperator::performForwardGpuNUFFT(
    GpuArray<DType2> imgData, GpuArray<CufftType> &kspaceData,
    GpuNUFFTOutput gpuNUFFTOut)
{
  if (DEBUG)
    printf(
        "BGpuNUFFT: allocate and copy sector processing order of size %d...\n",
        this->sectorProcessingOrder.count());
  allocateAndCopyToDeviceMem<IndType2>(&sector_processing_order_d,
                                       this->sectorProcessingOrder.data,
                                       this->sectorProcessingOrder.count());

  GpuNUFFTOperator::performForwardGpuNUFFT(imgData, kspaceData, gpuNUFFTOut);

  freeTotalDeviceMemory(sector_processing_order_d, NULL);  // NULL as stop token
}

