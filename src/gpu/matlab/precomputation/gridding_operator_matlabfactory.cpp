
#include "gridding_operator_matlabfactory.hpp"
#include <iostream>

GriddingND::GriddingOperatorMatlabFactory GriddingND::GriddingOperatorMatlabFactory::instance;

GriddingND::GriddingOperatorMatlabFactory& GriddingND::GriddingOperatorMatlabFactory::getInstance()
{
	return instance;
}

mxArray* createIndicesArray(const size_t arrSize)
{
	mwSize indSize[] = {arrSize};
	return mxCreateNumericArray(1,indSize,mxUINT64_CLASS,mxREAL);
}

mxArray* createSectorDataArray()
{
	mwSize secSize[] = {10};	
	return mxCreateNumericArray(1,secSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createDensArray()
{
	mwSize densSize[] = {1,10};
	return mxCreateNumericArray(2,densSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createCoordsArray(const size_t arrSize)
{
	mwSize coordSize[] = {3,arrSize};//row major in matlab, SoA
	return mxCreateNumericArray(2,coordSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createSectorCentersArray(const size_t arrSize)
{
	mwSize secSize[] = {arrSize*3};
	return mxCreateNumericArray(1,secSize,mxUINT64_CLASS,mxREAL);
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorMatlabFactory::initDataIndices(GriddingND::GriddingOperator* griddingOp, size_t coordCnt)
{
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
	plhs[3] = createCoordsArray(coordCnt);

	GriddingND::Array<DType> coordsData;
	coordsData.data = (DType*)mxGetData(plhs[3]);
	coordsData.dim.length = coordCnt;
	return coordsData;
}

GriddingND::Array<IndType3> GriddingND::GriddingOperatorMatlabFactory::initSectorCenters(GriddingND::GriddingOperator* griddingOp, size_t sectorCnt)
{
	plhs[4] = createSectorCentersArray(sectorCnt);

	GriddingND::Array<IndType3> sectorCenters;
	sectorCenters.data = (IndType3*)mxGetData(plhs[4]);
	sectorCenters.dim.length = sectorCnt;
	return sectorCenters;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorMatlabFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims,mxArray *plhs[])
{
	IndType* dataSectorDataCount = NULL;
	DType*   dataSortedDens = NULL;
	DType*   dataSortedCoords = NULL;
	IndType* sectorCenters = NULL;
	mexPrintf("trying to create index array with size %d\n",kSpaceTraj.count());
	plhs[1] = createSectorDataArray();
    plhs[2] = createDensArray();
	this->plhs = plhs;
	
	dataSectorDataCount = (IndType*)mxGetData(plhs[1]);
	dataSortedDens   = (DType*)mxGetData(plhs[2]);

	return GriddingOperatorFactory::createGriddingOperator(kSpaceTraj,densCompData,sensData,kernelWidth,sectorWidth,osf,imgDims);
}

