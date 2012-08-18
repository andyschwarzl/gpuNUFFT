#include "matlab_helper.h"

#include "mex.h"
#include "matrix.h"
//#include <math.h>
#include <complex>
#include <vector>

//GRIDDING 3D
#include "gridding_gpu.hpp"

//#include "fftw3.h"
#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h> 
#include <cublas.h>

#include <stdio.h>
#include <string>
#include <iostream>

#include <string.h>

/*
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
  
    int cuDevice = 0;
	cudaGetDevice(&cuDevice);
	// Create context
	//CUcontext cuContext;
	//cuCtxCreate(&cuContext, 0, cuDevice);
	

//TODO check input params count first!
	/*  if(nrhs != 9 ) {
	printf("\nUsage:\n");
    return;
	} else if(nlhs>1) {
	printf("Too many output arguments\n");
    return;
	}*/

    //////////////////////////////////// fetching data from MATLAB

	int pcount = 0;  //Parametercounter
    
	//Data
	DType* data = NULL;
	int data_count;
	int n_coils;
	readMatlabInputArray<DType>(prhs, pcount++, 2,"data",&data, &data_count,3,&n_coils);
	
	//Coords
	DType* coords = NULL;
	int coord_count;
	readMatlabInputArray<DType>(prhs, pcount++, 3,"coords",&coords, &coord_count);

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
    //'im_width', 'osr', 'kernel_width', 'sector_width'
	int im_width = getParamField<int>(matParams,"im_width");
	DType osr = getParamField<DType>(matParams,"osr"); 
	int kernel_width = getParamField<int>(matParams,"kernel_width");
	int sector_width = getParamField<int>(matParams,"sector_width");
		
	if (MATLAB_DEBUG)
		mexPrintf("passed Params, IM_WIDTH: %d, OSR: %f, KERNEL_WIDTH: %d, SECTOR_WIDTH: %d\n",im_width,osr,kernel_width,sector_width);
  
	//if (MATLAB_DEBUG)
	{
		size_t free_mem = 0;
		size_t total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		mexPrintf("memory usage on device, free: %lu total: %lu\n",free_mem,total_mem);
	}
 
	long kernel_count = calculateGrid3KernelSize(osr, kernel_width/2.0f);
	DType* kernel = (DType*) calloc(kernel_count,sizeof(float));
	loadGrid3Kernel(kernel,kernel_count,kernel_width,osr);
	
	//calc grid width -> oversampling
	int grid_width = (unsigned long)(im_width * osr);
	
	//Output Image
	CufftType* imdata;
	const int n_dims = 5;//2 * w * h * d * ncoils, 2 -> Re + Im
	unsigned long dims_im[n_dims];
	dims_im[0] = 2; /* complex */
	dims_im[1] = im_width;
	dims_im[2] = im_width;
	dims_im[3] = im_width;
	dims_im[4] = (unsigned long)(n_coils);

	long im_count = dims_im[1]*dims_im[2]*dims_im[3];
	
	//plhs[0] = mxCreateNumericArray(n_dims,(const mwSize*)dims_im,mxGetClassID(prhs[0]),mxREAL);
	plhs[0] = mxCreateNumericArray(n_dims,(const mwSize*)dims_im,mxSINGLE_CLASS,mxREAL);
	
    imdata = (CufftType*)mxGetData(plhs[0]);
	if (imdata == NULL)
     mexErrMsgTxt("Could not create output mxArray.\n");

	gridding3D_gpu_adj(data,data_count,n_coils,coords,&imdata,im_count,grid_width,kernel,kernel_count,kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,DEAPODIZATION);//CONVOLUTION);
    cudaThreadSynchronize();
	free(kernel);

	{
		size_t free_mem = 0;
		size_t total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		mexPrintf("memory usage on device afterwards, free: %lu total: %lu\n",free_mem,total_mem);
	}
//	cudaDeviceReset();
     //CUcontext  pctx ;
     //cuCtxPopCurrent(&pctx);	
	//mexPrintf("%s\n", cudaGetErrorString(cudaGetLastError()));
}













