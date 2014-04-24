
#include "balanced_gridding_operator.hpp"

GriddingND::GriddingInfo* GriddingND::BalancedGriddingOperator::initAndCopyGriddingInfo()
{
  GriddingND::GriddingInfo* gi_host = initGriddingInfo();

  gi_host->sectorsToProcess = sectorProcessingOrder.count();

  if (DEBUG)
    printf("copy Gridding Info to symbol memory... size = %ld \n",sizeof(GriddingND::GriddingInfo));

  initConstSymbol("GI",gi_host,sizeof(GriddingND::GriddingInfo));

  if (DEBUG)
    printf("...done!\n");
  return gi_host;
}


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
  performForwardConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,gi_host);
}

// Adds behaviour of GriddingOperator by 
// adding a sector processing order 
void GriddingND::BalancedGriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingND::Array<CufftType>& imgData, GriddingOutput griddingOut)
{
  if (DEBUG)
    printf("allocate and copy sector processing order of size %d...\n",this->sectorProcessingOrder.count());
  allocateAndCopyToDeviceMem<IndType2>(&sector_processing_order_d,this->sectorProcessingOrder.data,this->sectorProcessingOrder.count());

  GriddingOperator::performGriddingAdj(kspaceData,imgData,griddingOut);

  freeTotalDeviceMemory(sector_processing_order_d,NULL);//NULL as stop token
}