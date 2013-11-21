#include "matlab_helper.h"

#include "mex.h"
#include "matrix.h"

#include <complex>
#include <vector>

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
  MATLAB Wrapper for Precomputation of GriddingOperator

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
		mexPrintf("Starting GRIDDING 3D Precomputation...\n");

	// get cuda context associated to MATLAB 
	// 
	int cuDevice = 0;
	cudaGetDevice(&cuDevice);
	cudaSetDevice(cuDevice);//check if really necessary

	mexAtExit(cleanUp);

	// fetching data arrays from MATLAB
	int pcount = 0; //Parametercounter

	//Coords
	DType* coords = NULL;
	int coord_count;
	readMatlabInputArray<DType>(prhs, pcount++, 0,"coords",&coords, &coord_count);

	//Density compensation
	DType* density_comp = NULL;
	int density_count;
	readMatlabInputArray<DType>(prhs, pcount++, 0,"density-comp",&density_comp, &density_count);
	
	//Sensitivity data
	DType* sens = NULL;
	int sens_count;
	readMatlabInputArray<DType>(prhs, pcount++, 0,"sens",&sens, &sens_count);
	
	//Parameters
    const mxArray *matParams = prhs[pcount++];
	
	if (!mxIsStruct (matParams))
         mexErrMsgTxt ("expects struct containing parameters!");

	int im_width = getParamField<int>(matParams,"im_width");
	DType osr = getParamField<DType>(matParams,"osr"); 
	int kernel_width = getParamField<int>(matParams,"kernel_width");
	int sector_width = getParamField<int>(matParams,"sector_width");
	int traj_length = getParamField<int>(matParams,"trajectory_length");
		
	if (MATLAB_DEBUG)
	{
		mexPrintf("passed Params, IM_WIDTH: %d, OSR: %f, KERNEL_WIDTH: %d, SECTOR_WIDTH: %d dens_count: %d sens_count: %d traj_len: %d\n",im_width,osr,kernel_width,sector_width,density_count,sens_count,traj_length);
		size_t free_mem = 0;
		size_t total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		mexPrintf("memory usage on device, free: %lu total: %lu\n",free_mem,total_mem);
	}
 
	// Output 
	// dataIndices
	// dataSectorMapping
	// sectorCenters
	// 

    GriddingND::Array<DType> kSpaceTraj;
    kSpaceTraj.data = coords;
    kSpaceTraj.dim.length = traj_length;
	
	GriddingND::Array<DType> density_compArray;
	density_compArray.data = density_comp;
	density_compArray.dim.length = density_count;

	GriddingND::Array<DType> sensArray;
	sensArray.data = NULL;
	sensArray.dim.length = 0;

	GriddingND::Dimensions imgDims;
	imgDims.width = im_width;
	imgDims.height = im_width;
	imgDims.depth = im_width;

	try
	{
		GriddingND::GriddingOperatorMatlabFactory factory = GriddingND::GriddingOperatorMatlabFactory::getInstance();

		GriddingND::GriddingOperator *griddingOp;
		griddingOp = factory.createGriddingOperator(kSpaceTraj,density_compArray,sensArray,kernel_width,sector_width,osr,imgDims,plhs);
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
