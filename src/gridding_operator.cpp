
#include "gridding_operator.hpp"

#include <iostream>

void GriddingND::GriddingOperator::performGriddingAdj(DType2* kspaceData, CufftType** imgData)
{
	performGriddingAdj(kspaceData,imgData,DEAPODIZATION);
}

void GriddingND::GriddingOperator::performGriddingAdj(DType2* kspaceData, CufftType** imgData, GriddingOutput griddingOut)
{
	std::cout << "performing gridding adjoint!!!" << std::endl;

	std::cout << "test " << this->kspaceDim.width << std::endl;

	std::cout << "dataCount: " << this->dataCount << " chnCount: " << this->chnCount << std::endl;
	std::cout << "imgCount: " << this->getImgCount() << " gridWidth: " << this->getGridWidth() << std::endl;
	gridding3D_gpu_adj(kspaceData,this->dataCount,this->chnCount,this->kspaceCoords,imgData,this->getImgCount(),this->getGridWidth(),this->kernel,this->kernelCount,this->kernelWidth,(int*)this->sectors,this->sectorCount,(int*)this->sectorCenters,this->sectorWidth, this->kspaceDim.width,this->osf,this->applyDensComp(),this->dens,griddingOut);
}

void GriddingND::GriddingOperator::performForwardGridding(DType2* imgData, CufftType** kspaceData)
{
	performForwardGridding(imgData,kspaceData,CONVOLUTION);
}

void GriddingND::GriddingOperator::performForwardGridding(DType2* imgData, CufftType** kspaceData, GriddingOutput griddingOut)
{
	std::cout << "performing forward gridding!!!" << std::endl;

	std::cout << "test " << this->kspaceDim.width << std::endl;

	std::cout << "dataCount: " << this->dataCount << " chnCount: " << this->chnCount << std::endl;
	std::cout << "imgCount: " << this->getImgCount() << " gridWidth: " << this->getGridWidth() << std::endl;
	gridding3D_gpu(kspaceData,this->dataCount,this->chnCount,this->kspaceCoords,imgData,this->getImgCount(),this->getGridWidth(),this->kernel,this->kernelCount,this->kernelWidth,(int*)this->sectors,this->sectorCount,(int*)this->sectorCenters,this->sectorWidth, this->kspaceDim.width,this->osf,CONVOLUTION);
}