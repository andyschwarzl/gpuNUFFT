#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <complex>
#include <vector>

#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h> 
#include <cublas.h>
#include "cuda_utils.hpp"
#include <stdio.h>
#include <iostream>
#include "config.hpp"
#include "cufft_config.hpp"

#define MAX_BLOCK_SZ 512

#include "tikreg_gridding_kernels.cu"

/**
 * Forward Gridding using sparse Matrix
 * 
 * Extracted from FREIBURG Code 
 * 
*/
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	//check count of passed arguments
    if(nrhs != 11 ) 
	{
		printf("\nUsage:\n");
		return;
	} else if(nlhs>1) 
	{
		printf("Too many output arguments\n");
		return;
	}

    // fetching data from MATLAB
    int pcnt = 0;  
    const mxArray *ImageData;
    ImageData = prhs[pcnt++];//0...Image Daten       
    CufftType *img = ( CufftType *) mxGetData(ImageData);

    const mxArray *ImageDim;
    ImageDim = prhs[pcnt++];//1...Image Dimensions
    DType *image_dims = (DType*) mxGetData(ImageDim);
	const mwSize *dims_imagedim = mxGetDimensions(ImageDim);
	if (MATLAB_DEBUG)
		mexPrintf("Test %d, %d\n",(int)image_dims[0],(int)image_dims[1]);
	
	const mxArray *Sn;
    Sn = prhs[pcnt++];//1...SN Map
    DType *sn = ( DType *) mxGetData(Sn);
    
	int numsens;
	const mxArray *NumSens;
	NumSens = prhs[pcnt++];//2...Anzahl Spulen
	DType *num_sens = (DType*) mxGetData(NumSens);
	numsens = (int) num_sens[0];
	if (MATLAB_DEBUG)
		mexPrintf("Number of Coils: %d\n",numsens);

    const mwSize dims_sz[] = {2, (int)image_dims[0], (int)image_dims[1], (int)image_dims[2],numsens };//2x64x64x44
    int w = (int)dims_sz[1];//64
    int h = (int)dims_sz[2];//64
    int d = (int)dims_sz[3];//44
    int totsz = w*h*d;//64x64x44 = 180224
	
    const mxArray *Ipk_index;
    Ipk_index = prhs[pcnt++];//2...Index (Y)       
    const mwSize *dims_ipk = mxGetDimensions(Ipk_index);
    DType *ipk_index = (DType*) mxGetData(Ipk_index);

    const mxArray *Ipk_we;
    Ipk_we = prhs[pcnt++]; //3...Weight (Y)      
    CufftType *ipk_we = (CufftType*) mxGetData(Ipk_we);
  
    int numP = dims_ipk[0];//125
    int numK = dims_ipk[1];//11685
	
	//output dimensions
	const mwSize numdim =3;
	const mwSize dims_k[] = {2, numK, numsens};
    
	int the_index[numP*numK];
    for(int i = 0; i < numP*numK; i++)
        the_index[i] = (int)(ipk_index[i]-1);

    const mxArray *Dims_pad;
    Dims_pad = prhs[pcnt++];//4...Dimension Kd - Bild 64x64x44
    DType *dims_pad_d = (DType*) mxGetData(Dims_pad);
    int w_pad = (int)dims_pad_d[0];
    int h_pad = (int)dims_pad_d[1];
    int d_pad = (int)dims_pad_d[2];
    int totsz_pad  = w_pad*h_pad*d_pad;
 
    const mxArray *BPidx;
    BPidx = prhs[pcnt++];  //5...Backprojection VXIdx  - Bildinhalt   
    int numVox= mxGetM(BPidx); //Anzahl Zeilen
    int * bpidx = (int*) mxGetData(BPidx);
    
    //const mxArray *BPmidx;
    //BPmidx = prhs[pcnt++]; //6...Backprojection MidX 
    pcnt++;
    //const mxArray *BPweight;
    //BPweight = prhs[pcnt++]; //7...Backprojection Weight  
	pcnt++;
    const mxArray *Params;
    Params = prhs[pcnt++]; //8... Parameter   
    DType *params = (DType*) mxGetData(Params);
    DType lambda = params[1]; //Regularisierungsparam
    int device_num = (int) params[2]; //Device
    int VERBOSE = (int) params[4]; //Verbose-Mode
	  VERBOSE = MATLAB_DEBUG;
    if (VERBOSE == 1)  
        mexPrintf("gpuDevice: %i  lambda^2: %f\n",device_num,lambda);

   /**************** Init Cuda *****************/
    //CUdevice dev; 
		if (MATLAB_DEBUG)
			mexPrintf("start 1...\n");
    
		CUdevice dev; 
		if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
    {
			if (MATLAB_DEBUG)
				mexPrintf("dev:%i\n",dev);
		}   
  
    // MALLOCs    
    CufftType *tmp1,*tmp2, *_r, *_ipk_we;
	CufftType *_img;
	DType* _sn;
	
    int *_the_index;
	    	
    cufftHandle            plan;
	//output erzeugen
	plhs[0]             =  mxCreateNumericArray(numdim,dims_k,mxSINGLE_CLASS,mxREAL);
     
  CufftType *res = (CufftType*) mxGetData(plhs[0]);

	//allocateAndCopyToDeviceMem<std::complex<DType>>(&_img,img, totsz*numsens);
  cudaMalloc( (void **) &_r,sizeof(CufftType)*numK*numsens);
	cudaMalloc( (void **) &_img,sizeof(CufftType)*totsz*numsens);
  cudaMalloc( (void **) &tmp1,sizeof(CufftType)*totsz_pad);
  cudaMalloc( (void **) &tmp2,sizeof(CufftType)*totsz_pad);
  cudaMalloc( (void **) &_sn,sizeof(DType)*totsz);
	cudaMalloc( (void **) &_r,sizeof(CufftType)*numK*numsens);
	cudaMalloc( (void **) &_ipk_we,sizeof(CufftType)*numP*numK);
	cudaMalloc( (void **) &_the_index,sizeof(int)*numP*numK);

  cudaMemset( tmp1,0,sizeof(CufftType)*totsz_pad);
	cudaMemset( tmp2,0,sizeof(CufftType)*totsz_pad);
	cudaMemset( _img,0,sizeof(CufftType)*totsz*numsens);

  //   ************** copy data on device **********************
  cudaMemcpy( _ipk_we, ipk_we, sizeof(CufftType)*numP*numK, cudaMemcpyHostToDevice);
	cudaMemcpy( _img, img, sizeof(CufftType)*numsens*totsz, cudaMemcpyHostToDevice);
	cudaMemcpy( _the_index, the_index, sizeof(int)*numP*numK, cudaMemcpyHostToDevice);
	cudaMemcpy( _sn, sn, sizeof(DType)*totsz, cudaMemcpyHostToDevice);
  cudaMemcpy( ipk_we, _ipk_we, sizeof(CufftType)*numP*numK, cudaMemcpyDeviceToHost);
  cudaMemcpy( the_index, _the_index, sizeof(int)*numP*numK, cudaMemcpyDeviceToHost);

  cudaThreadSynchronize();
    
  if (VERBOSE == 1) 
    mexPrintf("numP: %i  numK: %i whd %i %i %i pad %i %i %i numsens: %i\n",numP,numK,w,h,d,w_pad,h_pad,d_pad,numsens);
          
  if (VERBOSE == 1) 
    mexPrintf("num active Vox: %i\n",numVox);    
  
    if (MATLAB_DEBUG)
		mexPrintf("trying to create cufft plan with %d\n",CufftTransformType);
	int err;
	if (err=cufftPlan3d(&plan, d_pad, h_pad, w_pad, CufftTransformType) != CUFFT_SUCCESS)
	{
		mexPrintf("create cufft plan has failed with err %i \n",err);
		mexPrintf("%s\n", cudaGetErrorString(cudaGetLastError()));
		//return;
	}
    // thread managements 
    int vx_block = 128;
    dim3 dimBlock_vx(vx_block,1);
    dim3 dimGrid_vx (numVox/vx_block + 1,1);
 
    dim3 dimBlock_dw(d,1);//d=64
    dim3 dimGrid_dw (w,h);//w=64,h=64

    dim3 dimBlock_sq(d,1);
    dim3 dimGrid_sq (w*h,1);
    
	// for sensing 
    int sens_block = 256;
    dim3 dimBlock_se(sens_block,1);
    dim3 dimGrid_se (numK/sens_block + 1,1);

    // we need this because first fft fails
    int _res = pt2CufftExec(plan, tmp1, tmp2, CUFFT_FORWARD);
	
    if (VERBOSE == 1)
      mexPrintf("first fft call ret: %i\n", _res);
	
	cudaMemset( tmp2,0,sizeof(CufftType)*totsz_pad);
    cudaMemset(_r,0, sizeof(CufftType)*numK*numsens);            
	
	if (VERBOSE == 1)
		mexPrintf("start forward gridding...\n");
	// do sens -- b=A x
    for (int i = 0; i < numsens; i++)
    { 
		//reset data for next coil
        cudaMemset(tmp1,0, sizeof(CufftType)*totsz_pad);
		
		//Multiplikation mit SN Matrix
        sn_mult<<<dimGrid_dw,dimBlock_dw>>>(tmp1,_img + i*totsz_pad, _sn, w, h, d, w_pad, h_pad, d_pad);     
		
		//FT in k-space
        if (err=pt2CufftExec(plan, tmp1, tmp2, CUFFT_FORWARD) != CUFFT_SUCCESS)
        {
			mexPrintf("cufft has failed with err %i \n",err);
			mexPrintf("%s\n", cudaGetErrorString(cudaGetLastError()));
            //return;
        }
			dosenswithoffset<<<dimGrid_se,dimBlock_se>>>(_r,tmp2,_ipk_we,_the_index,numP,numK,i*numK,numK*numsens);
		
		//add to result -> without sense 
		//addcoiltores<<<dimGrid_se,dimBlock_se>>>(_r,tmp1, numK*numsens,i*numK);
     }
  
    cudaMemcpy( res, _r, sizeof(CufftType)*numK*numsens,cudaMemcpyDeviceToHost);    

    cudaFree(tmp1);
    cudaFree(tmp2);
	
    cudaFree(_r); 
    cudaFree(_img);
	  cudaFree(_sn);
	
    cudaFree(_ipk_we);
    cudaFree(_the_index);
	    
    cufftDestroy(plan);

		//delete the_index;
   // leads to segfaults
   // CUcontext  pctx ;
   // cuCtxPopCurrent(&pctx);	
}













