#include "mex.h"
#include "matrix.h"
//#include <math.h>
#include <complex>
#include <vector>

#ifdef __unix__ 
#include <unistd.h>
#elif defined _WIN32 
# include <windows.h>
#endif

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

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

template <typename TType>
void readMatlabInputArray(const mxArray *prhs[], int input_index, int highest_varying_dim, const char* name,TType** data, int* data_entries)
{
	int nd = mxGetNumberOfDimensions(prhs[input_index]); /* get coordinate dimensions */
	
	if (MATLAB_DEBUG)
		mexPrintf("number of dims %d\n",nd);

	const mwSize *dims = mxGetDimensions(prhs[input_index]);
    
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

	if (MATLAB_DEBUG)
	{
		mexPrintf("%s dimensions: ",name);
		for (int i=0; i<nd; i++)
			mexPrintf(" %d ",dims[i]);
		mexPrintf("\n");
	}
	
	*data_entries = dims[1];

	const mxArray *matlabData;
    matlabData = prhs[input_index];
	bool is_int = false;

	if (mxIsInt32(matlabData) || mxIsUint32(matlabData))
	{
		is_int = true;
		*data = ( TType*) mxGetData(matlabData);		
	}
	else
	{
		*data = ( TType*) mxGetPr(matlabData);
	}
	if (MATLAB_DEBUG)
	{
		for (int i = 0; i < MIN((highest_varying_dim * (*data_entries)),100); i++)//re, im
			if (is_int)
				mexPrintf("%s: %d, ",name,(*data)[i]);
			else
				mexPrintf("%s: %f, ",name,(*data)[i]);

		mexPrintf("\n");
	}
}

template <typename TName>
inline TName getParamField(const mxArray* params, const char* fieldname)
{
	const mxArray* data = mxGetField(params, 0, fieldname);
	if (mxIsInt32(data))
	{
		return (TName)(((TName*)mxGetData(data))[0]); 
	}
	else
	{
		return (TName)(((TName*)mxGetPr(data))[0]); 
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
	if (MATLAB_DEBUG)
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

	int pcnt = 0;  //Parametercounter
    
	//Data
	DType* data = NULL;
	int data_cnt;
	readMatlabInputArray<DType>(prhs, pcnt++, 2,"data",&data, &data_cnt);
	
	//Coords
	DType* coords = NULL;
	int coord_cnt;
	readMatlabInputArray<DType>(prhs, pcnt++, 3,"coords",&coords, &coord_cnt);

	//Sectors
	int* sectors = NULL;
	int sector_cnt;
	readMatlabInputArray<int>(prhs, pcnt++, 1,"sectors",&sectors, &sector_cnt);

	//Sector centers
	int* sector_centers = NULL;
	readMatlabInputArray<int>(prhs, pcnt++, 3,"sectors-centers",&sector_centers, &sector_cnt);

	//Parameters
    mwIndex j;
    const mxArray *matParams = prhs[pcnt++];
	
	if (!mxIsStruct (matParams))
         mexErrMsgTxt ("expects struct containing parameters!");
    //'im_width', 'osr', 'kernel_width', 'sector_width'
	int im_width = getParamField<int>(matParams,"im_width");
	DType osr = getParamField<DType>(matParams,"osr"); 
	int kernel_width = getParamField<int>(matParams,"kernel_width");
	int sector_width = getParamField<int>(matParams,"sector_width");
		
	if (MATLAB_DEBUG)
		mexPrintf("passed Params, IM_WIDTH: %d, OSR: %f, KERNEL_WIDTH: %d, SECTOR_WIDTH: %d\n",im_width,osr,kernel_width,sector_width);

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
	CufftType* gdata;
	const int n_dims = 4;
	unsigned long dims_g[n_dims];
	dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
	dims_g[2] = (unsigned long)(im_width * osr);
	dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[1]*dims_g[2]*dims_g[3];

	plhs[0] = mxCreateNumericArray(n_dims,(const mwSize*)dims_g,mxGetClassID(prhs[0]),mxREAL);
    gdata = (CufftType*)mxGetData(plhs[0]);
	if (gdata == NULL)
     mexErrMsgTxt("Could not create output mxArray.\n");

	gridding3D_gpu(data,data_cnt,coords,gdata,grid_size,kern,kernel_entries,sectors,sector_cnt,sector_centers,sector_width, kernel_width, kernel_entries,dims_g[1],FFT);
	
	free(kern);
  CUcontext  pctx ;
  cuCtxPopCurrent(&pctx);	
}













