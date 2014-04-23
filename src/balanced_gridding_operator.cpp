
#include "balanced_gridding_operator.hpp"

void GriddingND::BalancedGriddingOperator::adjConvolution(DType2* data_d, 
      DType* crds_d, 
      CufftType* gdata_d,
      DType* kernel_d, 
      IndType* sectors_d, 
      IndType* sector_centers_d,
  GriddingND::GriddingInfo* gi_host)
{
  performConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_processing_order_d,sector_centers_d,gi_host);
}

void GriddingND::BalancedGriddingOperator::forwardConvolution(CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  GriddingND::GriddingInfo* gi_host)
{
  performTextureForwardConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,gi_host);
}

// Adds behaviour of GriddingOperator by 
// adding a sector processing order 
void GriddingND::BalancedGriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingND::Array<CufftType>& imgData, GriddingOutput griddingOut)
{
  if (DEBUG)
    printf("allocate and copy sector processing order of size %d...\n",this->sectorProcessingOrder.count());
  allocateAndCopyToDeviceMem<IndType>(&sector_processing_order_d,this->sectorProcessingOrder.data,this->sectorProcessingOrder.count());

  GriddingOperator::performGriddingAdj(kspaceData,imgData,griddingOut);

  freeTotalDeviceMemory(sector_processing_order_d,NULL);//NULL as stop token
}