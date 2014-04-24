#ifndef BALANCED_GRIDDING_OPERATOR_H_INCLUDED
#define BALANCED_GRIDDING_OPERATOR_H_INCLUDED

#include "gridding_types.hpp"
#include "gridding_operator.hpp"

namespace GriddingND
{
  class BalancedGriddingOperator : public GriddingOperator
  {
  public:

    BalancedGriddingOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims): 
    GriddingOperator(kernelWidth,sectorWidth,osf,imgDims,true,BALANCED)
    {
    }

    ~BalancedGriddingOperator()
    {
    }

    Array<IndType2>  getSectorProcessingOrder(){return this->sectorProcessingOrder;}
    void setSectorProcessingOrder(Array<IndType2> sectorProcessingOrder)	{this->sectorProcessingOrder = sectorProcessingOrder;}

    // OPERATIONS
    void performGriddingAdj(Array<DType2> kspaceData, Array<CufftType>& imgData, GriddingOutput griddingOut = DEAPODIZATION);
    
    OperatorType getType() {return OperatorType::BALANCED;}

  protected:
	
    // sectorProcessingOrder
    Array<IndType2> sectorProcessingOrder;
    
    IndType2* sector_processing_order_d;
    
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
  };
}

#endif //BALANCED_GRIDDING_OPERATOR_H_INCLUDED
