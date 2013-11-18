
#include "gridding_operator_matlabfactory.hpp"
#include <iostream>

GriddingND::GriddingOperatorMatlabFactory GriddingND::GriddingOperatorMatlabFactory::instance;

GriddingND::GriddingOperatorMatlabFactory& GriddingND::GriddingOperatorMatlabFactory::getInstance()
{
	return instance;
}

mxArray* createIndicesArray()
{
	mwSize indSize[] = {1};
	return mxCreateNumericArray(1,indSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createSectorDataArray()
{
	mwSize secSize[] = {1};	
	return mxCreateNumericArray(1,secSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createDensArray()
{
	mwSize densSize[] = {1,1};
	return mxCreateNumericArray(2,densSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createCoordsArray()
{
	mwSize coordSize[] = {1,1};
	return mxCreateNumericArray(2,coordSize,mxSINGLE_CLASS,mxREAL);
}

mxArray* createSectorCentersArray()
{
	mwSize secSize[] = {1,1,1};
	return mxCreateNumericArray(3,secSize,mxSINGLE_CLASS,mxREAL);
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorMatlabFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims,mxArray *plhs[])
{
	IndType* dataIndicesMatlab = NULL;
	IndType* dataSectorDataCount = NULL;
	DType*   dataSortedDens = NULL;
	DType*   dataSortedCoords = NULL;
	IndType* sectorCenters = NULL;
	
	plhs[0] = createIndicesArray();
	plhs[1] = createSectorDataArray();
    plhs[2] = createDensArray();
	plhs[3] = createCoordsArray();
	plhs[4] = createSectorCentersArray();
	
    dataIndicesMatlab = (IndType*)mxGetData(plhs[0]);

	if (dataIndicesMatlab == NULL)
		mexErrMsgTxt("Could not create output mxArray.\n");

	dataSectorDataCount = (IndType*)mxGetData(plhs[1]);
	dataSortedDens   = (DType*)mxGetData(plhs[2]);
	dataSortedCoords = (DType*)mxGetData(plhs[3]);
	sectorCenters = (IndType*)mxGetData(plhs[4]);

	return GriddingOperatorFactory::createGriddingOperator(kSpaceTraj,densCompData,sensData,kernelWidth,sectorWidth,osf,imgDims);
}

