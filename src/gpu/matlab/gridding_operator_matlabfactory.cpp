
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

mxArray* createIndicesArray(const IndType arrSize)
{
	mwSize indSize[] = {1,arrSize};
	return mxCreateNumericArray(2,indSize,mxUINT64_CLASS,mxREAL);
}

mxArray* createSectorDataArray(const IndType arrSize)
{
	mwSize secSize[] = {1,arrSize};	
	return mxCreateNumericArray(2,secSize,mxUINT64_CLASS,mxREAL);
}

mxArray* createDensArray(const IndType arrSize)
{
	mwSize densSize[] = {1,arrSize}; //scaling factor
	return mxCreateNumericArray(2,densSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createCoordsArray(const IndType arrSize, const IndType nDim)
{
	mwSize coordSize[] = {nDim,arrSize};//row major in matlab, SoA (DType3 or DType2)
	return mxCreateNumericArray(2,coordSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createSectorCentersArray(const IndType arrSize, const IndType nDim)
{
	mwSize secSize[] = {nDim,arrSize};//IndType3 or IndType2
	return mxCreateNumericArray(2,secSize,mxUINT64_CLASS,mxREAL);
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorMatlabFactory::initDataIndices(GriddingND::GriddingOperator* griddingOp, IndType coordCnt)
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

GriddingND::Array<DType> GriddingND::GriddingOperatorMatlabFactory::initCoordsData(GriddingND::GriddingOperator* griddingOp, IndType coordCnt)
{	
	if (MATLAB_DEBUG)
		mexPrintf("init Coords Output Array: %d\n",coordCnt);

	plhs[3] = createCoordsArray(coordCnt,griddingOp->getImageDimensionCount());

	GriddingND::Array<DType> coordsData;
	coordsData.data = (DType*)mxGetData(plhs[3]);
	coordsData.dim.length = coordCnt;
	return coordsData;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorMatlabFactory::initSectorCenters(GriddingND::GriddingOperator* griddingOp, IndType sectorCnt)
{
	if (MATLAB_DEBUG)
		mexPrintf("init Sector Centers Output Array: %d\n",sectorCnt);
	plhs[4] = createSectorCentersArray(sectorCnt,griddingOp->getImageDimensionCount());

	GriddingND::Array<IndType> sectorCenters;
	sectorCenters.data = (IndType*)mxGetData(plhs[4]);
	sectorCenters.dim.length = sectorCnt;
	return sectorCenters;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorMatlabFactory::initSectorDataCount(GriddingND::GriddingOperator* griddingOp, IndType dataCount)
{
	if (MATLAB_DEBUG)
		mexPrintf("init SectorData Output Array: %d\n",dataCount);
	plhs[1] = createSectorDataArray(dataCount);

	Array<IndType> sectorDataCount;
	sectorDataCount.data = (IndType*)mxGetData(plhs[1]);
	sectorDataCount.dim.length = dataCount;
	return sectorDataCount;

}

GriddingND::Array<DType> GriddingND::GriddingOperatorMatlabFactory::initDensData(GriddingND::GriddingOperator* griddingOp, IndType coordCnt)
{
	if (MATLAB_DEBUG)
		mexPrintf("init Dens Output Array: %d\n",coordCnt);
	plhs[2] = createDensArray(coordCnt);

	Array<DType> densData;
	densData.data = (DType*)mxGetData(plhs[2]);
	densData.dim.length = coordCnt;

	return densData;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorMatlabFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims,mxArray *plhs[])
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
