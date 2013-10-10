
#include "gridding_operator_factory.hpp"
#include <iostream>

static GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator()
{
	GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernel_width,sector_width,osr);
	
	griddingOp->setDataCount(data_entries);

	griddingOp->setChnCount(n_coils);	

	griddingOp->setSectorCount(sector_count);

	griddingOp->setOsf(osr);

	griddingOp->setKspaceCoords(coords);

	griddingOp->setSectors((size_t*)sectors);

	griddingOp->setSectorCenters((size_t*)sector_centers);

	griddingOp->setKSpaceWidth(im_width);

	griddingOp->setKSpaceHeight(im_width);

	griddingOp->setKSpaceDepth(im_width);

	return griddingOp;
}