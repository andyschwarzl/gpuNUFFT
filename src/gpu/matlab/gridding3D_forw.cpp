#include "matlab_helper.h"

#include "mex.h"
#include "matrix.h"

#include <complex>
#include <vector>

//GRIDDING 3D
#include "gridding_gpu.hpp"

#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h> 
#include <cublas.h>

#include <stdio.h>
#include <string>
#include <iostream>

#include <string.h>

#include "gridding_operator_factory.hpp"

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
  // 
  int cuDevice = 0;
  cudaGetDevice(&cuDevice);
  cudaSetDevice(cuDevice);//check if really necessary

  mexAtExit(cleanUp);
	//TODO check input params count first!
	// if(nrhs != 9 ) {
	//printf("\nUsage:\n");
    //return;
//	} else if(nlhs>1) {
//	printf("Too many output arguments\n");
 //   return;
//	}

  // fetching data from MATLAB

	int pcount = 0;  //Parametercounter
    
	//Data
	DType2* imdata = NULL;
	int im_count;
	int n_coils;
	readMatlabInputArray<DType2>(prhs, pcount++, 2,"imdata",&imdata, &im_count,3,&n_coils);
	
	//Coords
	DType* coords = NULL;
	int coord_count;
	readMatlabInputArray<DType>(prhs, pcount++, 0,"coords",&coords, &coord_count);

	//Sectors
	int* sectors = NULL;
	int sector_count;
	readMatlabInputArray<int>(prhs, pcount++, 1,"sectors",&sectors, &sector_count);

	//Sector centers
	int* sector_centers = NULL;
	readMatlabInputArray<int>(prhs, pcount++, 3,"sectors-centers",&sector_centers, &sector_count);

	//Parameters
    const mxArray *matParams = prhs[pcount++];
	
	if (!mxIsStruct (matParams))
         mexErrMsgTxt ("expects struct containing parameters!");

	int im_width = getParamField<int>(matParams,"im_width");
	DType osr = getParamField<DType>(matParams,"osr"); 
	int kernel_width = getParamField<int>(matParams,"kernel_width");
	int sector_width = getParamField<int>(matParams,"sector_width");
	mwSize data_entries = getParamField<mwSize>(matParams,"trajectory_length");

	if (MATLAB_DEBUG)
	{
		mexPrintf("passed Params, IM_WIDTH: %d, IM_COUNT: %d, OSR: %f, KERNEL_WIDTH: %d, SECTOR_WIDTH: %d, DATA_ENTRIES: %d, n_coils: %d\n",im_width,im_count,osr,kernel_width,sector_width,data_entries,n_coils);
		size_t free_mem = 0;
		size_t total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		mexPrintf("memory usage on device, free: %lu total: %lu\n",free_mem,total_mem);
	}
   
	long kernel_count = calculateGrid3KernelSize(osr, kernel_width/2.0f);
	DType* kernel = (DType*) calloc(kernel_count,sizeof(DType));
	loadGrid3Kernel(kernel,kernel_count,kernel_width,osr);
	
	//calc grid width -> oversampling
	int grid_width = (unsigned long)(im_width * osr);
	if (MATLAB_DEBUG)
		mexPrintf("grid width (incl. osr) = %d\n",grid_width);
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

    GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.width  = im_width;
    kSpaceData.dim.height = im_width;
    kSpaceData.dim.depth  = im_width;

	GriddingND::Array<CufftType> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;
	dataArray.dim.channels = n_coils;
	
    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance()->createGriddingOperator(kSpaceData,kernel_width,sector_width,osr);
    //GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernel_width,sector_width,osr);
	griddingOp->setSectorCount(sector_count);
	griddingOp->setOsf(osr);

	griddingOp->setSectors((size_t*)sectors);
	griddingOp->setSectorCenters((size_t*)sector_centers);

	griddingOp->performForwardGridding(imdata,dataArray);

	//gridding3D_gpu(&data,data_entries,n_coils,coords,imdata,im_count,grid_width,kernel,kernel_count,kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,CONVOLUTION);
    
	cudaThreadSynchronize();	
	free(kernel);
	if (MATLAB_DEBUG)	
	{
		size_t free_mem = 0;
		size_t total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		mexPrintf("memory usage on device afterwards, free: %lu total: %lu\n",free_mem,total_mem);
	}
}













