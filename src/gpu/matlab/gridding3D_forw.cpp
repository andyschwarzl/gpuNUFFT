#include "matlab_helper.h"

#include "mex.h"
#include "matrix.h"

#include <complex>
#include <vector>

//GRIDDING 3D
#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h> 
#include <cublas.h>

#include <stdio.h>
#include <string>
#include <iostream>

#include <string.h>

#include "gridding_operator_matlabfactory.hpp"

/** 
 * MEX file cleanup to reset CUDA Device 
**/
void cleanUp() 
{
	cudaDeviceReset();
}

/*
  MATLAB Wrapper for NUFFT Operation

	From MATLAB doc:
	Arguments
	nlhs Number of expected output mxArrays
	plhs Array of pointers to the expected output mxArrays
	nrhs Number of input mxArrays
	prhs Array of pointers to the input mxArrays. Do not modify any prhs values in your MEX-file. Changing the data in these read-only mxArrays can produce undesired side effects.
*/
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	if (MATLAB_DEBUG)
		mexPrintf("Starting Forward GRIDDING 3D Function...\n");
	
	// get cuda context associated to MATLAB 
	int cuDevice = 0;
	cudaGetDevice(&cuDevice);
	cudaSetDevice(cuDevice);

	mexAtExit(cleanUp);

	// fetch data from MATLAB
	int pcount = 0;  //Parametercounter
    
	// Input: Image data
	DType2* imdata = NULL;
	int im_count;
	int n_coils;
	readMatlabInputArray<DType2>(prhs, pcount++, 2,"imdata",&imdata, &im_count,3,&n_coils);
	
	// Data indices
	GriddingND::Array<IndType> dataIndicesArray = readAndCreateArray<IndType>(prhs,pcount++,0,"data-indices");
	
	// Coords
	GriddingND::Array<DType> kSpaceTraj = readAndCreateArray<DType>(prhs, pcount++, 0,"coords");

	// SectorData Count
	GriddingND::Array<IndType> sectorDataCountArray = readAndCreateArray<IndType>(prhs,pcount++,0,"sector-data-count");

  // Sector Processing Order
  GriddingND::Array<IndType> sectorProcessingOrderArray = readAndCreateArray<IndType>(prhs,pcount++,0,"sector-processing-order");
  
	// Sector centers
	GriddingND::Array<IndType> sectorCentersArray = readAndCreateArray<IndType>(prhs,pcount++,0,"sector-centers");

	if (MATLAB_DEBUG)
    mexPrintf("1st sector center: [%d,%d]\n", sectorCentersArray.data[0],sectorCentersArray.data[1]);

	//TODO Sens array	
	GriddingND::Array<DType2>  sensArray;
	sensArray.data = NULL;

	//Parameters
    const mxArray *matParams = prhs[pcount++];
	
	if (!mxIsStruct (matParams))
         mexErrMsgTxt ("expects struct containing parameters!");

	GriddingND::Dimensions imgDims = getDimensionsFromParamField(matParams,"img_dims");
	DType osr = getParamField<DType>(matParams,"osr"); 
	int kernel_width = getParamField<int>(matParams,"kernel_width");
	int sector_width = getParamField<int>(matParams,"sector_width");
	int data_entries = getParamField<int>(matParams,"trajectory_length");
  int interpolation_type = getParamField<int>(matParams,"interpolation_type");
	
	GriddingND::Array<DType2> imdataArray;
	imdataArray.data = imdata;
	imdataArray.dim = imgDims;
	imdataArray.dim.channels = n_coils;

	if (MATLAB_DEBUG)
	{
		mexPrintf("data indices count: %d\n",dataIndicesArray.count());
		mexPrintf("coords count: %d\n",kSpaceTraj.count());
		mexPrintf("sector data count: %d\n",sectorDataCountArray.count());
		mexPrintf("centers count: %d\n",sectorCentersArray.count());

		mexPrintf("passed Params, IM_WIDTH: [%d,%d,%d], IM_COUNT: %d, OSR: %f, KERNEL_WIDTH: %d, SECTOR_WIDTH: %d, DATA_ENTRIES: %d, n_coils: %d, interpolation type: %d\n",imgDims.width,imgDims.height,imgDims.depth,im_count,osr,kernel_width,sector_width,data_entries,n_coils,interpolation_type);
		size_t free_mem = 0;
		size_t total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		mexPrintf("memory usage on device, free: %lu total: %lu\n",free_mem,total_mem);
	}
   
	//Output Image
	CufftType* data;
	const mwSize n_dims = 3;//2 * data_cnt * ncoils, 2 -> Re + Im
	mwSize dims_data[n_dims];
	dims_data[0] =(mwSize)2; // complex 
	dims_data[1] = (mwSize)data_entries;
	dims_data[2] = (mwSize)n_coils;

	plhs[0] = mxCreateNumericArray(n_dims,dims_data,mxSINGLE_CLASS,mxREAL);
	
	data = (CufftType*)mxGetData(plhs[0]);
	if (data == NULL)
     mexErrMsgTxt("Could not create output mxArray.\n");

	GriddingND::Array<CufftType> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;
	dataArray.dim.channels = n_coils;

	try
	{
    GriddingND::GriddingOperatorMatlabFactory griddingFactory(getInterpolationTypeOf(interpolation_type));
		GriddingND::GriddingOperator *griddingOp = griddingFactory.loadPrecomputedGriddingOperator(kSpaceTraj,dataIndicesArray,sectorDataCountArray,sectorProcessingOrderArray,sectorCentersArray,sensArray,kernel_width,sector_width,osr,imgDims);

		griddingOp->performForwardGridding(imdataArray,dataArray);
		
		delete griddingOp;
	}
	catch (...)
	{
		mexPrintf("FAILURE in gridding operation\n");
	}

	cudaThreadSynchronize();
	if (MATLAB_DEBUG)	
	{
		size_t free_mem = 0;
		size_t total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		mexPrintf("memory usage on device afterwards, free: %lu total: %lu\n",free_mem,total_mem);
	}
}













