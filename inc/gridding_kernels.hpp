#ifndef GRIDDING_KERNELS_H
#define GRIDDING_KERNELS_H
#include "gridding_utils.hpp"
#include "cuda_utils.hpp"

//INVERSE Operations

// gridding function prototypes
void performConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType* kernel_d,
  IndType* sectors_d,
  IndType* sector_centers_d,
  GriddingND::GriddingInfo* gi_host
  );

void performConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType* kernel_d,
  IndType* sectors_d, 
  IndType* sector_processing_order_d,
  IndType* sector_centers_d,
  GriddingND::GriddingInfo* gi_host
  );

void performTextureConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType*			kernel_d, 
  IndType* sectors_d, 
  IndType* sector_centers_d,
  GriddingND::GriddingInfo* gi_host
  );

void performForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  GriddingND::GriddingInfo*	gi_host
  );

void performTextureForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  GriddingND::GriddingInfo*	gi_host
  );

void performFFTScaling(CufftType* data,
  int N, 
  GriddingND::GriddingInfo* gi_host);

void performDensityCompensation(DType2* data, DType* density_comp, GriddingND::GriddingInfo* gi_host);

void performFFTShift(CufftType* gdata_d,
  GriddingND::FFTShiftDir shift_dir,
  GriddingND::Dimensions gridDims,
  GriddingND::GriddingInfo* gi_host);

void performCrop(CufftType* gdata_d,
  CufftType* imdata_d,
  GriddingND::GriddingInfo* gi_host);

void performDeapodization(CufftType* imdata_d,
  GriddingND::GriddingInfo* gi_host);

void performDeapodization(CufftType* imdata_d,
  DType* deapo_d,
  GriddingND::GriddingInfo* gi_host);
//FORWARD Operations

void performForwardDeapodization(DType2* imdata_d,
  GriddingND::GriddingInfo* gi_host);

void performForwardDeapodization(DType2* imdata_d,
  DType* deapo_d,
  GriddingND::GriddingInfo* gi_host);

void performPadding(DType2* imdata_d,
  CufftType* gdata_d,					
  GriddingND::GriddingInfo* gi_host);

void precomputeDeapodization(DType* deapo_d,
  GriddingND::GriddingInfo* gi_host);

#endif
