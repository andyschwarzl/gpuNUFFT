
#include "gridding_operator.hpp"

#include <iostream>

void GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, CufftType** imgData)
{
	performGriddingAdj(kspaceData,imgData,DEAPODIZATION);
}

void GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, CufftType** imgData, GriddingOutput griddingOut)
{
	std::cout << "performing gridding adjoint!!!" << std::endl;

    std::cout << "test " << this->kSpaceCoords.dim.width << std::endl;

	std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
	std::cout << "imgCount: " << this->getImgCount() << " gridWidth: " << this->getGridWidth() << std::endl;
    gridding3D_gpu_adj(kspaceData.data,kspaceData.count(),kspaceData.dim.channels,this->kSpaceCoords.data,imgData,this->getImgCount(),this->getGridWidth(),this->kernel,this->kernelCount,this->kernelWidth,(int*)this->sectors,this->sectorCount,(int*)this->sectorCenters,this->sectorWidth, this->kSpaceCoords.dim.width,this->osf,this->applyDensComp(),this->dens,griddingOut);
}

void GriddingND::GriddingOperator::performForwardGridding(DType2* imgData, GriddingND::Array<CufftType> kspaceData)
{
	performForwardGridding(imgData,kspaceData,CONVOLUTION);
}

void GriddingND::GriddingOperator::performForwardGridding(DType2* imgData, GriddingND::Array<CufftType> kspaceData, GriddingOutput griddingOut)
{
	std::cout << "performing forward gridding!!!" << std::endl;

    std::cout << "test " << this->kSpaceCoords.dim.width << std::endl;

	std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
	std::cout << "imgCount: " << this->getImgCount() << " gridWidth: " << this->getGridWidth() << std::endl;
    gridding3D_gpu(&kspaceData.data,kspaceData.count(),kspaceData.dim.channels,this->kSpaceCoords.data,imgData,this->getImgCount(),this->getGridWidth(),this->kernel,this->kernelCount,this->kernelWidth,(int*)this->sectors,this->sectorCount,(int*)this->sectorCenters,this->sectorWidth, this->kSpaceCoords.dim.width,this->osf,CONVOLUTION);
}
