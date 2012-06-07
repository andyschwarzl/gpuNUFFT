#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <complex>
#include <vector>

#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h> 
#include <cublas.h>

#include <stdio.h>
#include <iostream>

#define MAX_BLOCK_SZ 512

#include "tikreg_gridding_kernels.cu"

/**
 * Adjoint Gridding using sparse Matrix
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
    const mxArray *Measurement;
    Measurement = prhs[pcnt++];//0...Daten       
    std::complex<float> *meas = ( std::complex<float> *) mxGetData(Measurement);

    const mxArray *ImageDim;
    ImageDim = prhs[pcnt++];//1...Image Dimensions
    float *image_dims = (float*) mxGetData(ImageDim);
	const mwSize *dims_imagedim = mxGetDimensions(ImageDim);
	mexPrintf("Test %d, %d\n",dims_imagedim[0],dims_imagedim[1]);
		
	const mxArray *Sn;
    Sn = prhs[pcnt++];//1...SN Map
    float *sn = ( float *) mxGetData(Sn);
    
	int numsens;
	const mxArray *NumSens;
	NumSens = prhs[pcnt++];//2...Anzahl Spulen
	float *num_sens = (float*) mxGetData(NumSens);
	numsens = (int) num_sens[0];
	mexPrintf("Number of Coils: %d\n",numsens);
	
    const int numdim =5;
    const int dims_sz[] = {2, (int)image_dims[0], (int)image_dims[1], (int)image_dims[2],numsens };//2x64x64x44
    int w = (int)dims_sz[1];//64
    int h = (int)dims_sz[2];//64
    int d = (int)dims_sz[3];//44
    int totsz = w*h*d;//64x64x44 = 180224
	
    const mxArray *Ipk_index;
    Ipk_index = prhs[pcnt++];//2...Index (Y)       
    const mwSize *dims_ipk = mxGetDimensions(Ipk_index);
    float *ipk_index = (float*) mxGetData(Ipk_index);

    const mxArray *Ipk_we;
    Ipk_we = prhs[pcnt++]; //3...Weight (Y)      
    std::complex<float> *ipk_we = (std::complex<float>*) mxGetData(Ipk_we);
  
    int numP = dims_ipk[0];//125
    int numK = dims_ipk[1];//11685
    
	int* the_index= new int[numP*numK];
    for(int i = 0; i < numP*numK; i++)
        the_index[i] = (int)(ipk_index[i]-1);

    const mxArray *Dims_pad;
    Dims_pad = prhs[pcnt++];//4...Dimension Kd - Bild 64x64x44
    float *dims_pad_d = (float*) mxGetData(Dims_pad);
    int w_pad = (int)dims_pad_d[0];
    int h_pad = (int)dims_pad_d[1];
    int d_pad = (int)dims_pad_d[2];
    int totsz_pad  = w_pad*h_pad*d_pad;
 
    const mxArray *BPidx;
    BPidx = prhs[pcnt++];  //5...Backprojection VXIdx  - Bildinhalt   
    int numVox= mxGetM(BPidx); //Anzahl Zeilen
    int * bpidx = (int*) mxGetData(BPidx);
    
    const mxArray *BPmidx;
    BPmidx = prhs[pcnt++]; //6...Backprojection MidX 
   
    const mxArray *BPweight;
    BPweight = prhs[pcnt++]; //7...Backprojection Weight  

    const mxArray *Params;
    Params = prhs[pcnt++]; //8... Parameter   
    float *params = (float*) mxGetData(Params);
    float lambda = params[1]; //Regularisierungsparam
    int device_num = (int) params[2]; //Device
    int VERBOSE = (int) params[4]; //Verbose-Mode
    
    if (VERBOSE == 1)  
        mexPrintf("gpuDevice: %i  lambda^2: %f\n",device_num,lambda);

    /**************** Init Cuda *****************/
    CUdevice dev; 
    
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
    {
		//   CUcontext  pctx ;
		//   cuCtxPopCurrent(&pctx);	      
    }   
    mexPrintf("dev:%i\n",dev);
       
    // MALLOCs
    cufftComplex *tmp1,*tmp2, *_r , *_meas, *_ipk_we;
	float* _sn;
	
    int *_the_index;
    cufftHandle            plan;
    
	//output erzeugen
	plhs[0]             =  mxCreateNumericArray(numdim,(const mwSize*)dims_sz,mxGetClassID(Measurement),mxREAL);
     
    std::complex<float> *res = (std::complex<float> *) mxGetData(plhs[0]);
   	
	cudaMalloc( (void **) &_meas,sizeof(cufftComplex)*numsens*numK);
    cudaMalloc( (void **) &tmp1,sizeof(cufftComplex)*totsz_pad);
    cudaMalloc( (void **) &tmp2,sizeof(cufftComplex)*totsz_pad);

	cudaMalloc( (void **) &_sn,sizeof(float)*totsz);
    cudaMalloc( (void **) &_r,sizeof(cufftComplex)*totsz*numsens);
	
    cudaMalloc( (void **) &_ipk_we,sizeof(cufftComplex)*numP*numK);
    cudaMalloc( (void **) &_the_index,sizeof(int)*numP*numK);
	 
    cudaThreadSynchronize();
   
    cudaMemset( tmp1,0,sizeof(cufftComplex)*totsz_pad);
    cudaMemset( tmp2,0,sizeof(cufftComplex)*totsz_pad);
    cudaMemset(  _r,0,sizeof(cufftComplex)*totsz*numsens); 
     cudaThreadSynchronize();

     /************** copy data on device **********************/

     cudaMemcpy( _meas, meas, sizeof(cufftComplex)*numsens*numK, cudaMemcpyHostToDevice);
     cudaMemcpy( _ipk_we, ipk_we, sizeof(cufftComplex)*numP*numK, cudaMemcpyHostToDevice);
     cudaMemcpy( _the_index, the_index, sizeof(int)*numP*numK, cudaMemcpyHostToDevice);
   
     cudaMemcpy( ipk_we, _ipk_we, sizeof(cufftComplex)*numP*numK, cudaMemcpyDeviceToHost);
     cudaMemcpy( the_index, _the_index, sizeof(int)*numP*numK, cudaMemcpyDeviceToHost);
     cudaMemcpy( _sn, sn, sizeof(float)*totsz, cudaMemcpyHostToDevice);

     cudaThreadSynchronize();
    
    if (VERBOSE == 1) 
        mexPrintf("numP: %i  numK: %i whd %i %i %i pad %i %i %i numsens: %i\n",numP,numK,w,h,d,w_pad,h_pad,d_pad,numsens);

    /************** copy bpidx on device **********************/
    int *_bpmidx;
    cufftComplex *_bpweight;
    int *bpsize = (int*) malloc(sizeof(int)*numVox);
    int *bponset  = (int*) malloc(sizeof(int)*(numVox+1));
    int *_bpsize, *_bponset, *_bpidx;
    bponset[0] = 0;
    for (int j = 0; j < numVox;j++)
    {
        mxArray *Midx = mxGetCell(BPmidx,j);
        bpsize[j] = mxGetM(Midx);
        bponset[j+1] = bponset[j] + bpsize[j];
    }
    
    int *tmp_bpmidx;
    cufftComplex *tmp_bpweight;
    tmp_bpmidx = (int*) malloc(sizeof(int)*bponset[numVox]);
    tmp_bpweight = (cufftComplex*) malloc(sizeof(cufftComplex)*bponset[numVox]);
    if (tmp_bpmidx == 0)
    {
        mexPrintf("out of mem (host)\n");
        return;
    }
    if (tmp_bpweight == 0)
    {
        mexPrintf("out of mem (host)\n");
        return;
    }
    
    for (int j = 0; j < numVox;j++)
    {
        mxArray *Midx = mxGetCell(BPmidx,j);
        mxArray *Weight = mxGetCell(BPweight,j);
        int *midx = (int*)  mxGetData(Midx);
        cufftComplex *bpwei = (cufftComplex*) mxGetData(Weight);
        memcpy(tmp_bpmidx + bponset[j] , midx, sizeof(int)* bpsize[j]);
        memcpy(tmp_bpweight + bponset[j] , bpwei, sizeof(cufftComplex)* bpsize[j]);    
    }
    
    cudaMalloc( (void **) &_bpmidx,sizeof(int)* bponset[numVox]);
    cudaMalloc( (void **) &_bpweight,sizeof(cufftComplex)* bponset[numVox]);
      
    cudaMemcpy(_bpmidx,tmp_bpmidx,sizeof(int)*bponset[numVox], cudaMemcpyHostToDevice);
    cudaMemcpy(_bpweight,tmp_bpweight,sizeof(cufftComplex)*bponset[numVox], cudaMemcpyHostToDevice);
 
    free(tmp_bpmidx);
    free(tmp_bpweight);

    cudaMalloc( (void **) &_bpsize,sizeof(int)* numVox);   
    cudaMalloc( (void **) &_bpidx,sizeof(int)* numVox);
    cudaMalloc( (void **) &_bponset,sizeof(int)* numVox+1);    
    cudaMemcpy(_bpsize,bpsize,sizeof(int)* numVox, cudaMemcpyHostToDevice);
    cudaMemcpy(_bpidx,bpidx,sizeof(int)* numVox, cudaMemcpyHostToDevice);
    cudaMemcpy(_bponset,bponset,sizeof(int)* numVox+1, cudaMemcpyHostToDevice);
            
    if (VERBOSE == 1) {
        mexPrintf("num active Vox: %i\n",numVox);
		mexPrintf("creating cufft plan with %d %d %d dimensions\n",d_pad,h_pad,w_pad);
	}
    
	int err;	
	if (err=cufftPlan3d(&plan, d_pad, h_pad, w_pad, CUFFT_C2C) != CUFFT_SUCCESS)
	{
		mexPrintf("create cufft plan has failed with err %i \n",err);
		mexPrintf("%s\n", cudaGetErrorString(cudaGetLastError()));
		return;
	}
    // thread managements 
    int vx_block = 128;
    dim3 dimBlock_vx(vx_block,1);
    dim3 dimGrid_vx (numVox/vx_block + 1,1);
 
    dim3 dimBlock_dw(d,1);//d=64
    dim3 dimGrid_dw (w,h);//w=64,h=64

    dim3 dimBlock_sq(d,1);
    dim3 dimGrid_sq (w*h,1);

    // we need this because first fft fails
    int _res = cufftExecC2C(plan, tmp1, tmp2, CUFFT_FORWARD);
	
    if (VERBOSE == 1)
      mexPrintf("first fft call ret: %i\n", _res);
	
	cudaMemset( tmp2,0,sizeof(cufftComplex)*totsz_pad);
    cudaMemset(_r,0, sizeof(cufftComplex)*totsz*numsens);            
	
    // backproject measurement -- x=A'b
    for (int i = 0; i < numsens; i++)
    { 
		//reset data for next coil
        cudaMemset(tmp1,0, sizeof(cufftComplex)*totsz_pad);
		//convolution with Gridding Kernel and Sampling (precomputed)
        backprojVX<<<dimGrid_vx,dimBlock_vx>>>(_bpidx,_bponset,_bpweight,_bpmidx,_bpsize,_meas + i*numK, tmp1,numVox);
        //Inverse FFT
		if (err=cufftExecC2C(plan, tmp1, tmp2, CUFFT_INVERSE) != CUFFT_SUCCESS)
        {
            mexPrintf("cufft has failed with err %i \n",err);
			mexPrintf("%s\n", cudaGetErrorString(cudaGetLastError()));
            return;
        }
		//Multiplikation mit SN Matrix
        sn_mult<<<dimGrid_dw,dimBlock_dw>>>(tmp2,tmp2, _sn, w, h, d, w_pad, h_pad, d_pad);
		
		//add -> without sense 
		addcoiltores<<<dimGrid_dw,dimBlock_dw>>>(_r,tmp2, totsz*numsens,i*totsz);
     }
  
     cudaMemcpy( res, _r, sizeof(cufftComplex)*totsz*numsens,cudaMemcpyDeviceToHost);    

    cudaFree(tmp1);
    cudaFree(tmp2);
	
	cudaFree(_sn);
    cudaFree(_r); 
    cudaFree(_meas);
	
    cudaFree(_ipk_we);
    cudaFree(_the_index);
	
    cudaFree(_bpmidx);
    cudaFree(_bpweight);
    cudaFree(_bpsize);
    cudaFree(_bpidx);
    cudaFree(_bponset);    
    
    cufftDestroy(plan);
    free(bpsize);
    free(bponset);
 
	mexPrintf("%s\n", cudaGetErrorString(cudaGetLastError()));

     CUcontext  pctx ;
     cuCtxPopCurrent(&pctx);	
}













