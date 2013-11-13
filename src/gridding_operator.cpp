
#include "gridding_operator.hpp"

#include <iostream>

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData)
{
	return performGriddingAdj(kspaceData,DEAPODIZATION);
}

template <typename T>
T* GriddingND::GriddingOperator::selectOrdered(GriddingND::Array<T>& dataArray)
{
	T* dataSorted = (T*) calloc(dataArray.count(),sizeof(T)); //2* re + im

	for (int i=0; i<dataIndices.count();i++)
	{
		dataSorted[i] = dataArray.data[dataIndices.data[i]];
	}
	return dataSorted;
}

template <typename T>
void GriddingND::GriddingOperator::writeOrdered(GriddingND::Array<T>& destArray, T* sortedArray)
{
	for (int i=0; i<dataIndices.count();i++)
	{
		destArray.data[dataIndices.data[i]] = sortedArray[i];
	}
}

void GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingND::Array<CufftType>& imgData, GriddingOutput griddingOut)
{
	std::cout << "performing gridding adjoint!!!" << std::endl;

	// select data ordered
	DType2* dataSorted = selectOrdered<DType2>(kspaceData);
	DType* densSorted = NULL;
	if (this->applyDensComp())
		densSorted = selectOrdered<DType>(this->dens);

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
					   this->osf,this->applyDensComp(),densSorted,griddingOut);
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingOutput griddingOut)
{
	// init result
	GriddingND::Array<CufftType> imgData;
	imgData.data = (CufftType*)calloc(imgDims.count(),sizeof(CufftType));
	imgData.dim = this->getImageDims();

	performGriddingAdj(kspaceData,imgData,griddingOut);

	return imgData;
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performForwardGridding(Array<DType2> imgData)
{
	return performForwardGridding(imgData,CONVOLUTION);
}

void GriddingND::GriddingOperator::performForwardGridding(GriddingND::Array<DType2> imgData,GriddingND::Array<CufftType>& kspaceData, GriddingOutput griddingOut)
{
	std::cout << "test " << this->kSpaceCoords.dim.width << std::endl;

	std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
	std::cout << "imgCount: " << imgData.count() << " gridWidth: " << this->getGridWidth() << std::endl;
    
	CufftType* kspaceDataSorted = (CufftType*) calloc(kspaceData.count(),sizeof(CufftType));

	gridding3D_gpu(&kspaceDataSorted,kspaceData.count(),kspaceData.dim.channels,this->kSpaceCoords.data,
		           imgData.data,imgData.count(),this->getGridWidth(),this->kernel.data,this->kernel.count(),
				   this->kernelWidth,this->sectorDataCount.data,this->sectorDims.count(),
				   (IndType*)this->sectorCenters.data,this->sectorWidth, imgData.dim.width,this->osf,griddingOut);

	writeOrdered<CufftType>(kspaceData,kspaceDataSorted);
	free(kspaceDataSorted);
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performForwardGridding(Array<DType2> imgData,GriddingOutput griddingOut)
{
	std::cout << "performing forward gridding!!!" << std::endl;

	GriddingND::Array<CufftType> kspaceData;
	kspaceData.data = (CufftType*)calloc(this->kSpaceCoords.count(),sizeof(CufftType));
	kspaceData.dim = this->kSpaceCoords.dim;

	performForwardGridding(imgData,kspaceData,griddingOut);

	return kspaceData;
}
