#include "mex.h"
#include "matrix.h"
//#include <math.h>
#include <complex>
#include <vector>

#ifdef __unix__ 
# include <unistd.h>
#elif defined _WIN32 
# include <windows.h>
#endif

//#include "fftw3.h"
#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h> 
#include <cublas.h>

#include <stdio.h>
#include <string>
#include <iostream>

#include <string.h>

//GRIDDING 3D
#include "gridding_gpu.hpp"

#ifdef __unix__ 
	#include <sys/time.h>
#elif defined _WIN32 
	#include <time.h>
#endif

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

#define MAX_BLOCK_SZ 512

template <typename TType>
void readMatlabInputArray(const mxArray *prhs[], int input_index, int highest_varying_dim, const char* name,TType** data, int* data_entries)
{
	int nd = mxGetNumberOfDimensions(prhs[input_index]); /* get coordinate dimensions */
	mexPrintf("number of dims %d\n",nd);
	const int *dims = (int*)mxGetDimensions(prhs[input_index]);
    
	if (nd == 2)
	{
		if(dims[0] != highest_varying_dim)//total: highest_varying_dim x N = 2
		{
			mexPrintf("dimensions of '%s' input array don't fit. Need to be %d x N but are %d x %d\n",name, highest_varying_dim, dims[0],dims[1]);
			mexErrMsgTxt ("Error occured!\n");
		}
	}
	else
	{
		mexPrintf("dimensions of '%s' input array don't fit. Need to be %d x N but are ",name, highest_varying_dim);
			for (int i=0; i<nd-1; i++)
				mexPrintf(" %d x ", dims[i]);
			if (nd > 1) mexPrintf(" %d\n", dims[nd-1]);
			else mexPrintf(" 0\n");
		mexErrMsgTxt ("Error occured!\n");
	}

	for (int i=0; i<nd; i++)
		mexPrintf("%s dimensions: %d ",name, dims[i]);
	mexPrintf("\n");
	
	*data_entries = dims[1];

	const mxArray *matlabData;
    matlabData = prhs[input_index];
	if (mxIsInt32(matlabData))
	{
		*data = ( TType*) mxGetData(matlabData);
	}
	else
	{
		*data = ( TType*) mxGetPr(matlabData);
	}
}

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
	mexPrintf("Starting GRIDDING 3D Function...\n");

	//TODO check input params count first!
  /*  if(nrhs != 9 ) {
	printf("\nUsage:\n");
    return;
	} else if(nlhs>1) {
	printf("Too many output arguments\n");
    return;
	}*/

    //////////////////////////////////// fetching data from MATLAB

	int pcnt = 0;  
    
	//Data
	DType* data = NULL;
	int data_cnt;
	readMatlabInputArray<DType>(prhs, pcnt++, 2,"data",&data, &data_cnt);
	for (int i = 0; i < 2*data_cnt; i++)//re, im
		mexPrintf("data: %f, ",data[i]);
	mexPrintf("\n");

	//Coords
	DType* coords = NULL;
	int coord_cnt;
	readMatlabInputArray<DType>(prhs, pcnt++, 3,"coords",&coords, &coord_cnt);
	for (int i = 0; i < 3*coord_cnt; i++)//x,y,z
		mexPrintf("coords: %f, ",coords[i]);
	mexPrintf("\n");

	//Sectors
	int* sectors = NULL;
	int sector_cnt;
	readMatlabInputArray<int>(prhs, pcnt++, 1,"sectors",&sectors, &sector_cnt);

	for (int i = 0; i < sector_cnt; i++)
	{
		mexPrintf("sectors: %d, ",sectors[i]);
	}
	mexPrintf("\n");

	//Sector centers
	int* sector_centers = NULL;
	readMatlabInputArray<int>(prhs, pcnt++, 3,"sectors-centers",&sector_centers, &sector_cnt);
	
	for (int i = 0; i < 3*sector_cnt; i++)//no int array as input array?
	{
		mexPrintf("sector-centers: %d, ",sector_centers[i]);
	}
	mexPrintf("\n");

	//Parameters
	const mxArray *Params;
    Params = prhs[pcnt++]; //8... Parameter   
    float *params = (float*) mxGetData(Params);
	//Image
	int im_width = (int)params[0];

	//oversampling ratio
	DType osr = (DType)params[1];

	//kernel width
	int kernel_width = (int)params[2];
	
	//sectors of data, count and start indices
	int sector_width = (int)params[3];


   /**************** Init Cuda *****************/
    
    cudaError_t rv; 
    CUdevice dev; 
    
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
    {
    //   CUcontext  pctx ;
    //   cuCtxPopCurrent(&pctx);	      
    }   
   
	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);
	DType* kern = (DType*) calloc(kernel_entries,sizeof(float));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

	//Output Grid
    DType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

	plhs[0] = mxCreateNumericArray(4,(const mwSize*)dims_g,mxGetClassID(prhs[0]),mxREAL);
    gdata = (DType*) mxGetData(plhs[0]);

	
	gridding3D_gpu(data,data_cnt,coords,gdata,grid_size,kern,kernel_entries,sectors,sector_cnt,sector_centers,sector_width, kernel_width, kernel_entries,dims_g[1]);

	free(kern);

    CUcontext  pctx ;
    cuCtxPopCurrent(&pctx);	
}













