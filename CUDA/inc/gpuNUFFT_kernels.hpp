#ifndef GPUNUFFT_KERNELS_H
#define GPUNUFFT_KERNELS_H
#include "gpuNUFFT_utils.hpp"
#include "cuda_utils.hpp"

//INVERSE Operations

// gpuNUFFT function prototypes
void performConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType* kernel_d,
  IndType* sectors_d,
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  );

//Balanced Version
void performConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType* kernel_d,
  IndType* sectors_d, 
  IndType2* sector_processing_order_d,
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  );

void performTextureConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType*			kernel_d, 
  IndType* sectors_d, 
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  );

//Balanced Version
void performTextureConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType*			kernel_d, 
  IndType* sectors_d, 
  IndType2* sector_processing_order_d,
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  );

void performForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo*	gi_host
  );

//Balanced Version
void performForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*	gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d,   
  IndType2*   sector_processing_order_d,
  IndType*		sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo*	gi_host
  );

void performTextureForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*  gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo*	gi_host
  );

//Balanced Version
void performTextureForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*  gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType2*   sector_processing_order_d,
  IndType*		sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo*	gi_host
  );

void performFFTScaling(CufftType* data,
  int N, 
  gpuNUFFT::GpuNUFFTInfo* gi_host);

void performDensityCompensation(DType2* data, DType* density_comp, gpuNUFFT::GpuNUFFTInfo* gi_host);

void performSensMul(CufftType* imdata_d,
  DType2* sens_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host,
  bool conjugate);

void performFFTShift(CufftType* gdata_d,
  gpuNUFFT::FFTShiftDir shift_dir,
  gpuNUFFT::Dimensions gridDims,
  gpuNUFFT::GpuNUFFTInfo* gi_host);

void performCrop(CufftType* gdata_d,
  CufftType* imdata_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host);

void performDeapodization(CufftType* imdata_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host);

void performDeapodization(CufftType* imdata_d,
  DType* deapo_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host);
//FORWARD Operations

void performForwardDeapodization(DType2* imdata_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host);

void performForwardDeapodization(DType2* imdata_d,
  DType* deapo_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host);

void performPadding(DType2* imdata_d,
  CufftType* gdata_d,					
  gpuNUFFT::GpuNUFFTInfo* gi_host);

void precomputeDeapodization(DType* deapo_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host);

void computeMinMaxBounds(DType* crds_d, 
  IndType* sectors_d, 
  IndType* sector_centers_d,
  uchar2* minmax_bounds_d, 
  gpuNUFFT::GpuNUFFTInfo* gi_host);

#endif
