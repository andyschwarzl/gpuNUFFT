
#include "gridding_operator.hpp"

#include <iostream>

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData)
{
	return performGriddingAdj(kspaceData,DEAPODIZATION);
}

template <typename T>
T* GriddingND::GriddingOperator::selectOrdered(GriddingND::Array<T> dataArray)
{
	T* dataSorted = (T*) calloc(dataArray.count(),sizeof(T)); //2* re + im

	for (int i=0; i<dataIndices.count();i++)
	{
		dataSorted[i] = dataArray.data[dataIndices.data[i]];
	}
	return dataSorted;
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingOutput griddingOut)
{
	std::cout << "performing gridding adjoint!!!" << std::endl;

	// init result
	GriddingND::Array<CufftType> imgData;
	imgData.data = (CufftType*)calloc(imgDims.count(),sizeof(CufftType));
	imgData.dim = this->getImageDims();

	// select data ordered
	DType2* dataSorted = selectOrdered<DType2>(kspaceData);

    std::cout << "test " << imgData.dim.width << std::endl;

	std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
	std::cout << "imgCount: " << imgData.count() << " gridWidth: " << this->getGridWidth() << std::endl;
	std::cout << "apply density comp: " << this->applyDensComp() << std::endl;
	std::cout << "kernel: " << this->kernel.data[3] << std::endl;
	std::cout << (this->dens.data == NULL) << std::endl; 

    gridding3D_gpu_adj(dataSorted,kspaceData.count(),kspaceData.dim.channels,this->kSpaceCoords.data,
		               &imgData.data,imgData.count(),this->getGridWidth(),this->kernel.data,this->kernel.count(),
					   this->kernelWidth,this->sectorDataCount.data,this->sectorDims.count(),
					   (IndType*)this->sectorCenters.data,this->sectorWidth, imgData.dim.width,
					   this->osf,this->applyDensComp(),this->dens.data,griddingOut);
	return imgData;
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performForwardGridding(Array<DType2> imgData)
{
	return performForwardGridding(imgData,CONVOLUTION);
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performForwardGridding(Array<DType2> imgData,GriddingOutput griddingOut)
{
	std::cout << "performing forward gridding!!!" << std::endl;

	GriddingND::Array<CufftType> kspaceData;
	kspaceData.data = (CufftType*)calloc(this->kSpaceCoords.count(),sizeof(CufftType));
	kspaceData.dim = this->kSpaceCoords.dim;

    std::cout << "test " << this->kSpaceCoords.dim.width << std::endl;

	std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
	std::cout << "imgCount: " << imgData.count() << " gridWidth: " << this->getGridWidth() << std::endl;
    gridding3D_gpu(&kspaceData.data,kspaceData.count(),kspaceData.dim.channels,this->kSpaceCoords.data,
		           imgData.data,imgData.count(),this->getGridWidth(),this->kernel.data,this->kernel.count(),
				   this->kernelWidth,this->sectorDataCount.data,this->sectorDims.count(),
				   (IndType*)this->sectorCenters.data,this->sectorWidth, imgData.dim.width,this->osf,griddingOut);
	return kspaceData;
}
