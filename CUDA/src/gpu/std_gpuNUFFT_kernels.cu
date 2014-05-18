#ifndef STD_GPUNUFFT_KERNELS_CU
#define STD_GPUNUFFT_KERNELS_CU

#include <string>

#include "gpuNUFFT_kernels.hpp"
#include "cuda_utils.hpp"
#include "precomp_utils.hpp"
#include "cuda_utils.cuh"

// Method to initialize CONSTANT memory symbols. Needs to reside in *.cu file 
// to work properly
//
//
void initConstSymbol(const char* symbol, const void* src, IndType size)
{
  if (std::string("GI").compare(symbol)==0)
    HANDLE_ERROR(cudaMemcpyToSymbol(GI, src,size));

  if (std::string("KERNEL").compare(symbol)==0)
    HANDLE_ERROR(cudaMemcpyToSymbol(KERNEL, src,size));
}

void bindTo1DTexture(const char* symbol, void* devicePtr, IndType count)
{
  if (std::string("texDATA").compare(symbol)==0)
  {
    HANDLE_ERROR (cudaBindTexture(NULL,texDATA, devicePtr,count*sizeof(DType2)));
  }
  else if (std::string("texGDATA").compare(symbol)==0)
  {
    HANDLE_ERROR (cudaBindTexture(NULL,texGDATA, devicePtr,count*sizeof(CufftType)));
  }
}


void initTexture(const char* symbol, cudaArray** devicePtr, gpuNUFFT::Array<DType> hostTexture)
{
  if (std::string("texKERNEL").compare(symbol)==0)
  {
    HANDLE_ERROR (cudaMallocArray (devicePtr, &texKERNEL.channelDesc, hostTexture.dim.width, 1));
    HANDLE_ERROR (cudaBindTextureToArray (texKERNEL, *devicePtr));
    HANDLE_ERROR(cudaMemcpyToArray(*devicePtr, 0, 0, hostTexture.data, sizeof(DType)*hostTexture.count(), cudaMemcpyHostToDevice));
    
    texKERNEL.filterMode = cudaFilterModePoint;
    texKERNEL.normalized = true;
    texKERNEL.addressMode[0] = cudaAddressModeClamp;
  }
  if (std::string("texKERNEL2D").compare(symbol)==0)
  {
    HANDLE_ERROR (cudaMallocArray (devicePtr, &texKERNEL2D.channelDesc, hostTexture.dim.width, hostTexture.dim.height));
    HANDLE_ERROR (cudaBindTextureToArray (texKERNEL2D, *devicePtr));
    HANDLE_ERROR(cudaMemcpyToArray(*devicePtr, 0, 0, hostTexture.data, sizeof(DType)*hostTexture.count(), cudaMemcpyHostToDevice));
    
    texKERNEL2D.filterMode = cudaFilterModeLinear;
    texKERNEL2D.normalized = true;
    texKERNEL2D.addressMode[0] = cudaAddressModeClamp;
    texKERNEL2D.addressMode[1] = cudaAddressModeClamp;    
  }
  else if (std::string("texKERNEL3D").compare(symbol)==0)
  {
    cudaExtent volumesize=make_cudaExtent(hostTexture.dim.width, hostTexture.dim.height, hostTexture.dim.depth); 
    cudaMalloc3DArray(devicePtr,&texKERNEL3D.channelDesc,volumesize); 

    cudaMemcpy3DParms copyparams = {0};
    copyparams.extent=volumesize; 
    copyparams.dstArray=*devicePtr; 
    copyparams.kind=cudaMemcpyHostToDevice; 
    copyparams.srcPtr= make_cudaPitchedPtr((void*)hostTexture.data,sizeof(DType)*hostTexture.dim.width,hostTexture.dim.height,hostTexture.dim.depth); 

    HANDLE_ERROR(cudaMemcpy3D(&copyparams)); 
    HANDLE_ERROR (cudaBindTextureToArray (texKERNEL3D, *devicePtr));
  
    texKERNEL3D.filterMode = cudaFilterModeLinear;
    texKERNEL3D.normalized = true;
    texKERNEL3D.addressMode[0] = cudaAddressModeClamp;
    texKERNEL3D.addressMode[1] = cudaAddressModeClamp;
    texKERNEL3D.addressMode[2] = cudaAddressModeClamp;
  }
}

void unbindTexture(const char* symbol)
{
  if (std::string("texKERNEL").compare(symbol)==0)
  {
    HANDLE_ERROR(cudaUnbindTexture(texKERNEL));    
  }
  else if (std::string("texKERNEL2D").compare(symbol)==0)
  {
    HANDLE_ERROR(cudaUnbindTexture(texKERNEL2D));    
  }
  else if (std::string("texKERNEL3D").compare(symbol)==0)
  {
    HANDLE_ERROR(cudaUnbindTexture(texKERNEL3D));    
  }
  else if (std::string("texDATA").compare(symbol)==0)
  {
    HANDLE_ERROR(cudaUnbindTexture(texDATA));    
  }
}


void freeTexture(const char* symbol,cudaArray* devicePtr)
{
  HANDLE_ERROR(cudaFreeArray(devicePtr));
  unbindTexture(symbol);
}

__global__ void fftScaleKernel(CufftType* data, DType scaling, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    CufftType data_p = data[t]; 
    data_p.x = data_p.x * scaling;
    data_p.y = data_p.y * scaling;
    data[t] = data_p;
    t = t+ blockDim.x*gridDim.x;
  }
}

void performFFTScaling(CufftType* data,int N, gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim(N,THREAD_BLOCK_SIZE));
  DType scaling_factor = (DType)1.0 / (DType) sqrt((DType)gi_host->im_width_dim);

  fftScaleKernel<<<grid_dim,block_dim>>>(data,scaling_factor,N);
}

__global__ void sensMulKernel(CufftType* imdata, DType2* sens, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    CufftType data_p = imdata[t]; 
    imdata[t].x = data_p.x * sens[t].x - data_p.y * sens[t].y; //Re
    imdata[t].y = data_p.x * sens[t].y + data_p.y * sens[t].x; //Im
    t = t+ blockDim.x*gridDim.x;
  }
}

__global__ void conjSensMulKernel(CufftType* imdata, DType2* sens, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    CufftType data_p = imdata[t]; 
    imdata[t].x = data_p.x * sens[t].x + data_p.y * sens[t].y; //Re
    imdata[t].y = data_p.y * sens[t].x - data_p.x * sens[t].y; //Im
    t = t+ blockDim.x*gridDim.x;
  }
}

void performSensMul(CufftType* imdata_d,
  DType2* sens_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host,
  bool conjugate)
{
  if (DEBUG)
    printf("running deapodization with precomputed values\n");

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);
  if (conjugate)
    conjSensMulKernel<<<grid_dim,block_dim>>>(imdata_d,sens_d,gi_host->im_width_dim);
  else
    sensMulKernel<<<grid_dim,block_dim>>>(imdata_d,sens_d,gi_host->im_width_dim);
}


__global__ void densityCompensationKernel(DType2* data, DType* density_comp, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    DType2 data_p = data[t]; 
    data_p.x = data_p.x * density_comp[t];
    data_p.y = data_p.y * density_comp[t];
    data[t] = data_p;
    t = t+ blockDim.x*gridDim.x;
  }
}

void performDensityCompensation(DType2* data, DType* density_comp, gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim(gi_host->data_count,THREAD_BLOCK_SIZE));
  densityCompensationKernel<<<grid_dim,block_dim>>>(data,density_comp,gi_host->data_count);
}

__global__ void deapodizationKernel(CufftType* gdata, DType beta, DType norm_val, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  int x, y, z;
  DType deapo;
  while (t < N) 
  { 
    getCoordsFromIndex(t, &x, &y, &z, GI.imgDims.x,GI.imgDims.y,GI.imgDims.z);

    deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
    //check if deapodization value is valid number
    if (!isnan(deapo))
    {
      CufftType gdata_p = gdata[t]; 
      gdata_p.x = gdata_p.x / deapo;//Re
      gdata_p.y = gdata_p.y / deapo;//Im
      gdata[t] = gdata_p;
    }
    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void deapodizationKernel2D(CufftType* gdata, DType beta, DType norm_val, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  int x, y;
  DType deapo;
  while (t < N) 
  { 
    getCoordsFromIndex2D(t, &x, &y, GI.imgDims.x,GI.imgDims.y);

    deapo = calculateDeapodizationAt2D(x,y,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
    //check if deapodization value is valid number
    if (!isnan(deapo))
    {
      CufftType gdata_p = gdata[t]; 
      gdata_p.x = gdata_p.x / deapo;//Re
      gdata_p.y = gdata_p.y / deapo;//Im
      gdata[t] = gdata_p;
    }
    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void precomputeDeapodizationKernel(DType* deapo_d, DType beta, DType norm_val, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;
  int x, y, z;
  DType deapo;
  while (t < N) 
  { 
    getCoordsFromIndex(t, &x, &y, &z, GI.imgDims.x,GI.imgDims.y,GI.imgDims.z);

    deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
    //check if deapodization value is valid number
    if (!isnan(deapo))// == deapo)
    {
      deapo_d[t] = (DType)1.0 / deapo;
    }
    else
      deapo_d[t] = (DType)1.0;

    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void precomputeDeapodizationKernel2D(DType* deapo_d, DType beta, DType norm_val, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;
  int x, y;
  DType deapo;
  while (t < N) 
  { 
    getCoordsFromIndex2D(t, &x, &y, GI.imgDims.x,GI.imgDims.y);

    deapo = calculateDeapodizationAt2D(x,y,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
    //check if deapodization value is valid number
    if (!isnan(deapo))// == deapo)
    {
      deapo_d[t] = (DType)1.0 / deapo;
    }
    else
      deapo_d[t] = (DType)1.0;

    t = t + blockDim.x*gridDim.x;
  }
}


__global__ void cropKernel(CufftType* gdata,CufftType* imdata, IndType3 offset, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;
  int x, y, z, grid_ind;
  while (t < N) 
  {
    getCoordsFromIndex(t, &x, &y, &z, GI.imgDims.x,GI.imgDims.y,GI.imgDims.z);
    grid_ind = computeXYZ2Lin(offset.x+x,offset.y+y,offset.z+z,GI.gridDims);
    imdata[t] = gdata[grid_ind];
    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void cropKernel2D(CufftType* gdata,CufftType* imdata, IndType3 offset, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;
  int x, y, grid_ind;
  while (t < N) 
  {
    getCoordsFromIndex2D(t, &x, &y, GI.imgDims.x,GI.imgDims.y);
    grid_ind = computeXY2Lin(offset.x+x,offset.y+y,GI.gridDims);
    imdata[t] = gdata[grid_ind];
    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void fftShiftKernel(CufftType* gdata, IndType3 offset, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;
  int x, y, z, x_opp, y_opp, z_opp, ind_opp;
  while (t < N) 
  { 
    getCoordsFromIndex(t, &x, &y, &z, GI.gridDims.x,GI.gridDims.y,GI.gridDims.z);
    //calculate "opposite" coord pair
    x_opp = (x + offset.x) % GI.gridDims.x;
    y_opp = (y + offset.y) % GI.gridDims.y;
    z_opp = (z + offset.z) % GI.gridDims.z;
    ind_opp = computeXYZ2Lin(x_opp,y_opp,z_opp,GI.gridDims);
    //swap points
    CufftType temp = gdata[t];
    gdata[t] = gdata[ind_opp];
    gdata[ind_opp] = temp;

    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void fftShiftKernel2D(CufftType* gdata, IndType3 offset, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;
  int x, y, x_opp, y_opp, ind_opp;
  while (t < N) 
  { 
    getCoordsFromIndex2D(t, &x, &y, GI.gridDims.x,GI.gridDims.y);
    //calculate "opposite" coord pair
    x_opp = (x + offset.x) % GI.gridDims.x;
    y_opp = (y + offset.y) % GI.gridDims.y;
    ind_opp = computeXY2Lin(x_opp,y_opp,GI.gridDims);
    //swap points
    CufftType temp = gdata[t];
    gdata[t] = gdata[ind_opp];
    gdata[ind_opp] = temp;

    t = t + blockDim.x*gridDim.x;
  }
}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performDeapodization(CufftType* imdata_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

  //Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
  DType norm_val_x = calculateDeapodizationValue(0,gi_host->grid_width_inv.x,gi_host->kernel_width,beta);
  DType norm_val_y = calculateDeapodizationValue(0,gi_host->grid_width_inv.y,gi_host->kernel_width,beta);
  DType norm_val_z = calculateDeapodizationValue(0,gi_host->grid_width_inv.z,gi_host->kernel_width,beta);
  DType norm_val;

  if (gi_host->is2Dprocessing)
    norm_val = norm_val_x * norm_val_y;
  else
    norm_val = norm_val_x * norm_val_y * norm_val_z;

  if (DEBUG)
    printf("running deapodization with norm_val %.2f\n",norm_val);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);
  if (gi_host->is2Dprocessing)
    deapodizationKernel2D<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
  else
    deapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
}

__global__ void precomputedDeapodizationKernel(CufftType* imdata, DType* deapo, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    CufftType data_p = imdata[t]; 
    data_p.x = data_p.x * deapo[t];
    data_p.y = data_p.y * deapo[t];
    imdata[t] = data_p;
    t = t+ blockDim.x*gridDim.x;
  }
}

void performDeapodization(CufftType* imdata_d,
  DType* deapo_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  if (DEBUG)
    printf("running deapodization with precomputed values\n");

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);
  precomputedDeapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,deapo_d,gi_host->im_width_dim);
}

void precomputeDeapodization(DType* deapo_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

  //Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
  DType norm_val_x = calculateDeapodizationValue(0,gi_host->grid_width_inv.x,gi_host->kernel_width,beta);
  DType norm_val_y = calculateDeapodizationValue(0,gi_host->grid_width_inv.y,gi_host->kernel_width,beta);
  DType norm_val_z = calculateDeapodizationValue(0,gi_host->grid_width_inv.z,gi_host->kernel_width,beta);
  DType norm_val;
  if (gi_host->is2Dprocessing)
    norm_val = norm_val_x * norm_val_y;
  else
    norm_val = norm_val_x * norm_val_y * norm_val_z;

  if (DEBUG)
    printf("running deapodization precomputation with norm_val %.2f\n",norm_val);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);
  if (gi_host->is2Dprocessing)
    precomputeDeapodizationKernel2D<<<grid_dim,block_dim>>>(deapo_d,beta,norm_val,gi_host->im_width_dim);
  else
    precomputeDeapodizationKernel<<<grid_dim,block_dim>>>(deapo_d,beta,norm_val,gi_host->im_width_dim);
}

void performCrop(CufftType* gdata_d,
  CufftType* imdata_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  IndType3 ind_off;
  ind_off.x = (IndType)(gi_host->imgDims.x * ((DType)gi_host->osr - 1.0f)/(DType)2);
  ind_off.y = (IndType)(gi_host->imgDims.y * ((DType)gi_host->osr - 1.0f)/(DType)2);
  ind_off.z = (IndType)(gi_host->imgDims.z * ((DType)gi_host->osr - 1.0f)/(DType)2);
  if (DEBUG)
    printf("start cropping image with offset %d\n",ind_off);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);

  if (gi_host->is2Dprocessing)
    cropKernel2D<<<grid_dim,block_dim>>>(gdata_d,imdata_d,ind_off,gi_host->im_width_dim);
  else
    cropKernel<<<grid_dim,block_dim>>>(gdata_d,imdata_d,ind_off,gi_host->im_width_dim);
}

void performFFTShift(CufftType* gdata_d,
  gpuNUFFT::FFTShiftDir shift_dir,
  gpuNUFFT::Dimensions gridDims,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  IndType3 offset;
  int t_width = 0;
  if (shift_dir == gpuNUFFT::FORWARD)
  {
    offset.x = (int)ceil((DType)(gridDims.width / (DType)2.0));
    offset.y = (int)ceil((DType)(gridDims.height / (DType)2.0));
    offset.z = (int)ceil((DType)(gridDims.depth / (DType)2.0));
    if (gridDims.width % 2)
    {
      t_width = (int)offset.x - 1;
    }
    else 
      t_width = (int)offset.x;
  }
  else
  {
    offset.x = (int)floor((DType)(gridDims.width / (DType)2.0));
    offset.y = (int)floor((DType)(gridDims.height / (DType)2.0));
    offset.z = (int)floor((DType)(gridDims.depth / (DType)2.0));
    t_width = (int)offset.x;
  }
  dim3 grid_dim;
  if (gi_host->is2Dprocessing)
    grid_dim = dim3(getOptimalGridDim((long)(gridDims.width*t_width),THREAD_BLOCK_SIZE));
  else
    grid_dim = dim3(getOptimalGridDim((long)(gridDims.width*gridDims.width),THREAD_BLOCK_SIZE));

  dim3 block_dim(THREAD_BLOCK_SIZE);

  if (gi_host->is2Dprocessing)
    fftShiftKernel2D<<<grid_dim,block_dim>>>(gdata_d,offset,(int)gridDims.width*t_width);
  else
    fftShiftKernel<<<grid_dim,block_dim>>>(gdata_d,offset,(int)gridDims.count()/2);
}

__global__ void forwardDeapodizationKernel(DType2* imdata, DType beta, DType norm_val, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  int x, y, z;
  DType deapo;
  while (t < N) 
  { 
    getCoordsFromIndex(t, &x, &y, &z, GI.imgDims.x);

    deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
    //check if deapodization value is valid number
    if (!isnan(deapo))// == deapo)
    {
      imdata[t].x = imdata[t].x / deapo;//Re
      imdata[t].y = imdata[t].y / deapo ; //Im
    }
    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void forwardDeapodizationKernel2D(DType2* imdata, DType beta, DType norm_val, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  int x, y;
  DType deapo;
  while (t < N) 
  { 
    getCoordsFromIndex2D(t, &x, &y, GI.imgDims.x);

    deapo = calculateDeapodizationAt2D(x,y,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
    //check if deapodization value is valid number
    if (!isnan(deapo))// == deapo)
    {
      imdata[t].x = imdata[t].x / deapo;//Re
      imdata[t].y = imdata[t].y / deapo ; //Im
    }
    t = t + blockDim.x*gridDim.x;
  }
}


__global__ void paddingKernel(DType2* imdata,CufftType* gdata, IndType3 offset,int N)
{	
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  int x, y, z,grid_ind;
  while (t < N) 
  { 
    getCoordsFromIndex(t, &x, &y, &z, GI.imgDims.x,GI.imgDims.y,GI.imgDims.z);
    grid_ind =  computeXYZ2Lin(offset.x + x,offset.y + y,offset.z +z,GI.gridDims);

    gdata[grid_ind].x =  imdata[t].x;
    gdata[grid_ind].y = imdata[t].y;
    t = t+ blockDim.x*gridDim.x;
  }
}

__global__ void paddingKernel2D(DType2* imdata,CufftType* gdata, IndType3 offset,int N)
{	
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  int x, y,grid_ind;
  while (t < N) 
  { 
    getCoordsFromIndex2D(t, &x, &y,GI.imgDims.x,GI.imgDims.y);
    grid_ind =  computeXY2Lin(offset.x + x,offset.y + y,GI.gridDims);

    gdata[grid_ind].x =  imdata[t].x;
    gdata[grid_ind].y = imdata[t].y;
    t = t+ blockDim.x*gridDim.x;
  }
}


//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performForwardDeapodization(DType2* imdata_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);

  //Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
  DType norm_val_x = calculateDeapodizationValue(0,gi_host->grid_width_inv.x,gi_host->kernel_width,beta);
  DType norm_val_y = calculateDeapodizationValue(0,gi_host->grid_width_inv.y,gi_host->kernel_width,beta);
  DType norm_val_z = calculateDeapodizationValue(0,gi_host->grid_width_inv.z,gi_host->kernel_width,beta);
  DType norm_val;
  if (gi_host->is2Dprocessing)
    norm_val = norm_val_x * norm_val_y;
  else
    norm_val = norm_val_x * norm_val_y * norm_val_z;

  if (gi_host->is2Dprocessing)
    forwardDeapodizationKernel2D<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
  else
    forwardDeapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
}

void performForwardDeapodization(DType2* imdata_d,
  DType* deapo_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  if (DEBUG)
    printf("running forward deapodization with precomputed values\n");

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);

  precomputedDeapodizationKernel<<<grid_dim,block_dim>>>((CufftType*)imdata_d,deapo_d,gi_host->im_width_dim);
}

void performPadding(DType2* imdata_d,
  CufftType* gdata_d,					
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  IndType3 ind_off;
  ind_off.x = (IndType)(gi_host->imgDims.x * ((DType)gi_host->osr -1.0f)/(DType)2);
  ind_off.y = (IndType)(gi_host->imgDims.y * ((DType)gi_host->osr -1.0f)/(DType)2);
  ind_off.z = (IndType)(gi_host->imgDims.z * ((DType)gi_host->osr -1.0f)/(DType)2);
  if (DEBUG)
    printf("start padding image with offset (%d,%d,%d)\n",ind_off.x,ind_off.y,ind_off.z);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);
  if (gi_host->is2Dprocessing)
    paddingKernel2D<<<grid_dim,block_dim>>>(imdata_d,gdata_d,ind_off,gi_host->im_width_dim);
  else
    paddingKernel<<<grid_dim,block_dim>>>(imdata_d,gdata_d,ind_off,gi_host->im_width_dim);
}

#endif //STD_GPUNUFFT_KERNELS_CU