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
	*data = ( TType*) mxGetPr(matlabData);
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
	DType* sectors = NULL;
	int sector_cnt;
	readMatlabInputArray<DType>(prhs, pcnt++, 1,"sectors",&sectors, &sector_cnt);
	
	int* sectors_int = (int*) calloc(sector_cnt,sizeof(int));

	for (int i = 0; i < sector_cnt; i++)//no int array as input array?
	{
		sectors_int[i] = (int)sectors[i];
		mexPrintf("sectors: %d, ",sectors_int[i]);
	}
	mexPrintf("\n");

	//Sector centers
	DType* sector_centers = NULL;
	readMatlabInputArray<DType>(prhs, pcnt++, 3,"sectors-centers",&sector_centers, &sector_cnt);
	
	int* sector_centers_int = (int*) calloc(3*sector_cnt,sizeof(int));

	for (int i = 0; i < 3*sector_cnt; i++)//no int array as input array?
	{
		sector_centers_int[i] = (int)sector_centers[i];
		mexPrintf("sector-centers: %d, ",sector_centers_int[i]);
	}
	mexPrintf("\n");


   /**************** Init Cuda *****************/
    
    cudaError_t rv; 
    CUdevice dev; 
    
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
    {
    //   CUcontext  pctx ;
    //   cuCtxPopCurrent(&pctx);	      
    }   
   
	//oversampling ratio
	DType osr = DEFAULT_OVERSAMPLING_RATIO;
	//kernel width
	int kernel_width = 3;

	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);
	DType* kern = (DType*) calloc(kernel_entries,sizeof(float));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

	//Image
	int im_width = 10;

	//Output Grid
    DType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    //gdata = (float*) calloc(grid_size,sizeof(float));
	plhs[0] = mxCreateNumericArray(4,(const mwSize*)dims_g,mxGetClassID(prhs[0]),mxREAL);
    gdata = (DType*) mxGetData(plhs[0]);

	//sectors of data, count and start indices
	int sector_width = 5;
	
//	int* sector_centers = (int*) calloc(3*sector_cnt,sizeof(int));

	gridding3D_gpu(data,data_cnt,coords,gdata,grid_size,kern,kernel_entries,sectors_int,sector_cnt,sector_centers_int,sector_width, kernel_width, kernel_entries,dims_g[1]);

	//free(data);
	//free(coords);
	//free(gdata);
	free(kern);
	free(sectors_int);
	free(sector_centers_int);

    CUcontext  pctx ;
    cuCtxPopCurrent(&pctx);	
}













