
#include "gridding_operator_factory.hpp"
#include <iostream>


GriddingND::GriddingOperatorFactory* GriddingND::GriddingOperatorFactory::instance = NULL;

GriddingND::GriddingOperatorFactory* GriddingND::GriddingOperatorFactory::getInstance()
{
          if (instance == NULL)
            instance = new GriddingOperatorFactory();
          return instance;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType> kSpaceCoords, size_t kernelWidth, size_t sectorWidth, DType osf)
{
    GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernelWidth,sectorWidth,osf);
	std::cout << "create gridding operator" << std::endl;
    griddingOp->setKspaceCoords(kSpaceCoords);

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
