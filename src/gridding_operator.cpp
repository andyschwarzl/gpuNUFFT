
#include "gridding_operator.hpp"

#include <iostream>

void GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingND::Array<CufftType> imgData)
{
	performGriddingAdj(kspaceData,imgData,DEAPODIZATION);
}

void GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingND::Array<CufftType> imgData, GriddingOutput griddingOut)
{
	std::cout << "performing gridding adjoint!!!" << std::endl;

    std::cout << "test " << this->kSpaceCoords.dim.width << std::endl;

	std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
	std::cout << "imgCount: " << imgData.count() << " gridWidth: " << this->getGridWidth() << std::endl;
	std::cout << "apply density comp: " << this->applyDensComp() << std::endl;
	std::cout << "kernel: " << this->kernel.data[3] << std::endl;
	std::cout << (this->dens.data == NULL) << std::endl; 

    gridding3D_gpu_adj(kspaceData.data,kspaceData.count(),kspaceData.dim.channels,this->kSpaceCoords.data,&imgData.data,imgData.count(),this->getGridWidth(),this->kernel.data,this->kernel.count(),this->kernelWidth,(int*)this->sectors.data,this->sectors.count(),(int*)this->sectorCenters.data,this->sectorWidth, this->kSpaceCoords.dim.width,this->osf,this->applyDensComp(),this->dens.data,griddingOut);
}

void GriddingND::GriddingOperator::performForwardGridding(Array<DType2> imgData, GriddingND::Array<CufftType> kspaceData)
{
	performForwardGridding(imgData,kspaceData,CONVOLUTION);
}

void GriddingND::GriddingOperator::performForwardGridding(Array<DType2> imgData, GriddingND::Array<CufftType> kspaceData, GriddingOutput griddingOut)
{
	std::cout << "performing forward gridding!!!" << std::endl;

    std::cout << "test " << this->kSpaceCoords.dim.width << std::endl;

	std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
	std::cout << "imgCount: " << imgData.count() << " gridWidth: " << this->getGridWidth() << std::endl;
    gridding3D_gpu(&kspaceData.data,kspaceData.count(),kspaceData.dim.channels,this->kSpaceCoords.data,imgData.data,imgData.count(),this->getGridWidth(),this->kernel.data,this->kernel.count(),this->kernelWidth,(int*)this->sectors.data,this->sectors.count(),(int*)this->sectorCenters.data,this->sectorWidth, this->kSpaceCoords.dim.width,this->osf,CONVOLUTION);
}
