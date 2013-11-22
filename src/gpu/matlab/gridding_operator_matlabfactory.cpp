
#include "gridding_operator_matlabfactory.hpp"
#include <iostream>


GriddingND::GriddingOperatorMatlabFactory GriddingND::GriddingOperatorMatlabFactory::instance;

GriddingND::GriddingOperatorMatlabFactory& GriddingND::GriddingOperatorMatlabFactory::getInstance()
{
	return instance;
}

void GriddingND::GriddingOperatorMatlabFactory::debug(const std::string& message)
{
	if (MATLAB_DEBUG)
		mexPrintf(message.c_str());
}

mxArray* createIndicesArray(const size_t arrSize)
{
	mwSize indSize[] = {1,arrSize};
	return mxCreateNumericArray(2,indSize,mxUINT64_CLASS,mxREAL);
}

mxArray* createSectorDataArray(const size_t arrSize)
{
	mwSize secSize[] = {1,arrSize};	
	return mxCreateNumericArray(2,secSize,mxUINT64_CLASS,mxREAL);
}

mxArray* createDensArray(const size_t arrSize)
{
	mwSize densSize[] = {1,arrSize}; //scaling factor
	return mxCreateNumericArray(2,densSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createCoordsArray(const size_t arrSize)
{
	mwSize coordSize[] = {3,arrSize};//row major in matlab, SoA (DType3)
	return mxCreateNumericArray(2,coordSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createSectorCentersArray(const size_t arrSize)
{
	mwSize secSize[] = {3,arrSize};//IndType3
	return mxCreateNumericArray(2,secSize,mxUINT64_CLASS,mxREAL);
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorMatlabFactory::initDataIndices(GriddingND::GriddingOperator* griddingOp, size_t coordCnt)
{	
	if (MATLAB_DEBUG)
		mexPrintf("init Data Index Output Array: %d\n",coordCnt);

	plhs[0] = createIndicesArray(coordCnt);
	
	GriddingND::Array<IndType> dataIndices;
	dataIndices.data = (IndType*)mxGetData(plhs[0]);
	dataIndices.dim.length = coordCnt;
	
	if (dataIndices.data == NULL)
		mexErrMsgTxt("Could not create output mxArray.\n");

	return dataIndices;
}

GriddingND::Array<DType> GriddingND::GriddingOperatorMatlabFactory::initCoordsData(GriddingND::GriddingOperator* griddingOp, size_t coordCnt)
{	
	if (MATLAB_DEBUG)
		mexPrintf("init Coords Output Array: %d\n",coordCnt);

	plhs[3] = createCoordsArray(coordCnt);

	GriddingND::Array<DType> coordsData;
	coordsData.data = (DType*)mxGetData(plhs[3]);
	coordsData.dim.length = coordCnt;
	return coordsData;
}

GriddingND::Array<IndType3> GriddingND::GriddingOperatorMatlabFactory::initSectorCenters(GriddingND::GriddingOperator* griddingOp, size_t sectorCnt)
{
	if (MATLAB_DEBUG)
		mexPrintf("init Sector Centers Output Array: %d\n",sectorCnt);
	plhs[4] = createSectorCentersArray(sectorCnt);

	GriddingND::Array<IndType3> sectorCenters;
	sectorCenters.data = (IndType3*)mxGetData(plhs[4]);
	sectorCenters.dim.length = sectorCnt;
	return sectorCenters;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorMatlabFactory::initSectorDataCount(GriddingND::GriddingOperator* griddingOp, size_t dataCount)
{
	if (MATLAB_DEBUG)
		mexPrintf("init SectorData Output Array: %d\n",dataCount);
	plhs[1] = createSectorDataArray(dataCount);

	Array<IndType> sectorDataCount;
	sectorDataCount.data = (IndType*)mxGetData(plhs[1]);
	sectorDataCount.dim.length = dataCount;
	return sectorDataCount;

}

GriddingND::Array<DType> GriddingND::GriddingOperatorMatlabFactory::initDensData(GriddingND::GriddingOperator* griddingOp, size_t coordCnt)
{
	if (MATLAB_DEBUG)
		mexPrintf("init Dens Output Array: %d\n",coordCnt);
	plhs[2] = createDensArray(coordCnt);

	Array<DType> densData;
	densData.data = (DType*)mxGetData(plhs[2]);
	densData.dim.length = coordCnt;

	return densData;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorMatlabFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType2>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims,mxArray *plhs[])
{
	if (MATLAB_DEBUG)
		mexPrintf("Start init of Gridding Operator\n");

	this->plhs = plhs;
	
	GriddingOperator* griddingOp = GriddingOperatorFactory::createGriddingOperator(kSpaceTraj,densCompData,sensData,kernelWidth,sectorWidth,osf,imgDims);

	if (!griddingOp->applyDensComp())
	{		
		if (MATLAB_DEBUG)
			mexPrintf("returning empty dens array\n");
		plhs[2] = createDensArray(1);
	}

	return griddingOp;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorMatlabFactory::loadPrecomputedGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<IndType>& dataIndices, GriddingND::Array<IndType>& sectorDataCount,GriddingND::Array<IndType3>& sectorCenters, GriddingND::Array<DType>& densCompData, GriddingND::Array<DType2>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
	GriddingOperator* griddingOp = new GriddingOperator(kernelWidth,sectorWidth,osf,imgDims);
	griddingOp->setSectorDims(GriddingOperatorFactory::computeSectorCountPerDimension(griddingOp->getGridDims(),griddingOp->getSectorWidth()));
	
	griddingOp->setKSpaceTraj(kSpaceTraj);
	griddingOp->setDataIndices(dataIndices);
	griddingOp->setSectorDataCount(sectorDataCount);
	griddingOp->setSectorCenters(sectorCenters);
	griddingOp->setDens(densCompData);
	griddingOp->setSens(sensData);
	return griddingOp;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorMatlabFactory::loadPrecomputedGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<IndType>& dataIndices, GriddingND::Array<IndType>& sectorDataCount,GriddingND::Array<IndType3>& sectorCenters, GriddingND::Array<DType2>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
	GriddingOperator* griddingOp = new GriddingOperator(kernelWidth,sectorWidth,osf,imgDims);
	griddingOp->setSectorDims(GriddingOperatorFactory::computeSectorCountPerDimension(griddingOp->getGridDims(),griddingOp->getSectorWidth()));
	
	griddingOp->setKSpaceTraj(kSpaceTraj);
	griddingOp->setDataIndices(dataIndices);
	griddingOp->setSectorDataCount(sectorDataCount);
	griddingOp->setSectorCenters(sectorCenters);
	griddingOp->setSens(sensData);
	return griddingOp;
}