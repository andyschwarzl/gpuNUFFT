
#include "gridding_operator_factory.hpp"
#include <iostream>


GriddingND::GriddingOperatorFactory* GriddingND::GriddingOperatorFactory::instance = NULL;


template <typename T>
GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<T> kSpaceCoords, size_t kernelWidth, size_t sectorWidth, DType osf)
{
    GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernelWidth,sectorWidth,osf);
	
    //griddingOp->setKspaceCoords(kSpaceCoords);

    /*griddingOp->setDataCount(data_entries);

	griddingOp->setChnCount(n_coils);	

	griddingOp->setSectorCount(sector_count);

	griddingOp->setOsf(osr);

	griddingOp->setKspaceCoords(coords);

	griddingOp->setSectors((size_t*)sectors);

	griddingOp->setSectorCenters((size_t*)sector_centers);

	griddingOp->setKSpaceWidth(im_width);

	griddingOp->setKSpaceHeight(im_width);

    griddingOp->setKSpaceDepth(im_width);*/

	return griddingOp;
}
