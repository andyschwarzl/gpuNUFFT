
#include "gridding_operator.hpp"

#include <iostream>

void GriddingND::GriddingOperator::performGriddingAdj(CufftType** imgData)
{
	performGriddingAdj(imgData,DEAPODIZATION);
}

void GriddingND::GriddingOperator::performGriddingAdj(CufftType** imgData, GriddingOutput griddingOut)
{
	std::cout << "performing gridding adjoint!!!" << std::endl;

	std::cout << "test " << this->kspaceDim.width << std::endl;

	std::cout << "dataCount: " << this->dataCount << " chnCount: " << this->chnCount << std::endl;
	std::cout << "imgCount: " << this->getImgCount() << " gridWidth: " << this->getGridWidth() << std::endl;
	gridding3D_gpu_adj(this->data,this->dataCount,this->chnCount,this->kspaceCoords,imgData,this->getImgCount(),this->getGridWidth(),this->kernel,this->kernelCount,this->kernelWidth,(int*)this->sectors,this->sectorCount,(int*)this->sectorCenters,this->sectorWidth, this->kspaceDim.width,this->osf,this->applyDensComp(),this->dens,griddingOut);
}