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
  MATLAB Wrapper for NUFFT^H Operation

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
		mexPrintf("Starting ADJOINT GRIDDING 3D Function...\n");

	// get cuda context associated to MATLAB 
	int cuDevice = 0;
	cudaGetDevice(&cuDevice);
	cudaSetDevice(cuDevice);

	mexAtExit(cleanUp);

	// fetch data from MATLAB
	int pcount = 0;  //param counter
    
	// Input: K-space Data
	DType2* data = NULL;
	int data_count;
	int n_coils;
	readMatlabInputArray<DType2>(prhs, pcount++, 2,"data",&data, &data_count,3,&n_coils);
	
	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_count;
	dataArray.dim.channels = n_coils;

	// Data indices
	GriddingND::Array<IndType> dataIndicesArray = readAndCreateArray<IndType>(prhs,pcount++,0,"data-indices");
	
	// Coords
	GriddingND::Array<DType> kSpaceTraj = readAndCreateArray<DType>(prhs, pcount++, 0,"coords");

	// SectorData Count
	GriddingND::Array<IndType> sectorDataCountArray = readAndCreateArray<IndType>(prhs,pcount++,0,"sector-data-count");

	// Sector centers
	GriddingND::Array<IndType> sectorCentersArray = readAndCreateArray<IndType>(prhs,pcount++,0,"sector-centers");

	// Density compensation
	GriddingND::Array<DType> density_compArray = readAndCreateArray<DType>(prhs, pcount++, 0,"density-comp");

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
	int traj_length = getParamField<int>(matParams,"trajectory_length");
		
	if (MATLAB_DEBUG)
	{
		mexPrintf("data indices count: %d\n",dataIndicesArray.count());
		mexPrintf("coords count: %d\n",kSpaceTraj.count());
		mexPrintf("sector data count: %d\n",sectorDataCountArray.count());
		mexPrintf("centers count: %d\n",sectorCentersArray.count());
		mexPrintf("dens count: %d\n",density_compArray.count());
		
		mexPrintf("passed Params, IM_WIDTH: [%d,%d,%d], OSR: %f, KERNEL_WIDTH: %d, SECTOR_WIDTH: %d dens_count: %d traj_len: %d n coils: %d\n",imgDims.width,imgDims.height,imgDims.depth,osr,kernel_width,sector_width,density_compArray.count(),traj_length,n_coils);
		size_t free_mem = 0;
		size_t total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		mexPrintf("memory usage on device, free: %lu total: %lu\n",free_mem,total_mem);
	}
	
	// Allocate Output: Image
	CufftType* imdata = NULL;
	const mwSize n_dims = 5;//2 * w * h * d * ncoils, 2 -> Re + Im
	mwSize dims_im[n_dims];
	dims_im[0] = (mwSize)2; /* complex */
	dims_im[1] = (mwSize)imgDims.width;
	dims_im[2] = (mwSize)imgDims.height;
	dims_im[3] = (mwSize)MAX(1,imgDims.depth);
	dims_im[4] = (mwSize)n_coils;

	plhs[0] = mxCreateNumericArray(n_dims,dims_im,mxSINGLE_CLASS,mxREAL);
	
  imdata = (CufftType*)mxGetData(plhs[0]);
	if (imdata == NULL)
     mexErrMsgTxt("Could not create output mxArray.\n");

	GriddingND::Array<DType2> imdataArray;
	imdataArray.data = imdata;
	imdataArray.dim = imgDims;
	imdataArray.dim.channels = n_coils;

	if (MATLAB_DEBUG)
  {
    mexPrintf(" imdataArray dims %d,%d,%d,%d\n",imdataArray.dim.width,imdataArray.dim.height,imdataArray.dim.depth,imdataArray.dim.channels);
	  mexPrintf(" data count: %d \n",data_count);
  }

	try
	{
    GriddingND::GriddingOperatorMatlabFactory griddingFactory(GriddingND::InterpolationType::TEXTURE_LOOKUP);
		GriddingND::GriddingOperator *griddingOp = griddingFactory.loadPrecomputedGriddingOperator(kSpaceTraj,dataIndicesArray,sectorDataCountArray,sectorCentersArray,density_compArray,sensArray,kernel_width,sector_width,osr,imgDims);

		griddingOp->performGriddingAdj(dataArray,imdataArray);

		delete griddingOp;
	}
	catch(...)
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
