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
void initConstSymbol(const char* symbol, const void* src, IndType size, cudaStream_t stream)
{
  if (std::string("GI").compare(symbol)==0)
    HANDLE_ERROR(cudaMemcpyToSymbolAsync(GI, src, size, 0, cudaMemcpyHostToDevice, stream));

  if (std::string("KERNEL").compare(symbol)==0)
    HANDLE_ERROR(cudaMemcpyToSymbolAsync(KERNEL, src, size, 0, cudaMemcpyHostToDevice, stream));
}

void bindTo1DTexture(const char* symbol, void* devicePtr, IndType count)
{
  if (std::string("texDATA").compare(symbol)==0)
  {
    HANDLE_ERROR (cudaBindTexture(NULL,texDATA, devicePtr,(unsigned long)count*sizeof(float2)));
  }
  else if (std::string("texGDATA").compare(symbol)==0)
  {
    HANDLE_ERROR (cudaBindTexture(NULL,texGDATA, devicePtr,(unsigned long)count*sizeof(cufftComplex)));
  }
}


void initTexture(const char* symbol, cudaArray** devicePtr, gpuNUFFT::Array<DType> hostTexture)
{
  if (std::string("texKERNEL").compare(symbol)==0)
  {
    HANDLE_ERROR(cudaMallocArray (devicePtr, &texKERNEL.channelDesc, hostTexture.dim.width, 1));
    HANDLE_ERROR(cudaBindTextureToArray(texKERNEL, *devicePtr));
    HANDLE_ERROR(cudaMemcpyToArray(*devicePtr, 0, 0, hostTexture.data, sizeof(float)*hostTexture.count(), cudaMemcpyHostToDevice));
    
    texKERNEL.filterMode = cudaFilterModePoint;
    texKERNEL.normalized = true;
    texKERNEL.addressMode[0] = cudaAddressModeClamp;
  }
  else if (std::string("texKERNEL2D").compare(symbol)==0)
  {
    HANDLE_ERROR(cudaMallocArray (devicePtr, &texKERNEL2D.channelDesc, hostTexture.dim.width, hostTexture.dim.height));

    HANDLE_ERROR(cudaBindTextureToArray(texKERNEL2D, *devicePtr));
    HANDLE_ERROR(cudaMemcpyToArray(*devicePtr, 0, 0, hostTexture.data, sizeof(float)*hostTexture.count(), cudaMemcpyHostToDevice));
    
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
    copyparams.srcPtr= make_cudaPitchedPtr((void*)hostTexture.data,sizeof(float)*hostTexture.dim.width,hostTexture.dim.height,hostTexture.dim.depth); 

    HANDLE_ERROR(cudaMemcpy3D(&copyparams)); 
    HANDLE_ERROR(cudaBindTextureToArray(texKERNEL3D, *devicePtr));
  
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
  else if (std::string("texGDATA").compare(symbol)==0)
  {
    HANDLE_ERROR(cudaUnbindTexture(texGDATA));    
  }
}


void freeTexture(const char* symbol, cudaArray* devicePtr)
{
  unbindTexture(symbol);
  HANDLE_ERROR(cudaFreeArray(devicePtr));  
}

__global__ void fftScaleKernel(CufftType* data, DType scaling, long int N)
{
  long int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      CufftType data_p = data[t + c*N]; 
      data_p.x = data_p.x * scaling;
      data_p.y = data_p.y * scaling;
      data[t + c*N] = data_p;
    }
    t = t+ blockDim.x*gridDim.x;

  }
}

void performFFTScaling(CufftType* data,long int N, gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  dim3 block_dim(64, 1, 8);
  //dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim(N,THREAD_BLOCK_SIZE));
  DType scaling_factor = (DType)1.0 / (DType) sqrt((DType)gi_host->gridDims_count);

  fftScaleKernel<<<grid_dim,block_dim>>>(data,scaling_factor,N);
}

__global__ void sensMulKernel(CufftType* imdata, DType2* sens, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      CufftType data_p = imdata[t + c*N]; 
      imdata[t + c*N].x = data_p.x * sens[t + c*N].x - data_p.y * sens[t + c*N].y; //Re
      imdata[t + c*N].y = data_p.x * sens[t + c*N].y + data_p.y * sens[t + c*N].x; //Im
    }
    t = t+ blockDim.x*gridDim.x;
  }
}

__global__ void conjSensMulKernel(CufftType* imdata, DType2* sens, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      CufftType data_p = imdata[t + c*N]; 
      imdata[t + c*N].x = data_p.x * sens[t + c*N].x + data_p.y * sens[t + c*N].y; //Re
      imdata[t + c*N].y = data_p.y * sens[t + c*N].x - data_p.x * sens[t + c*N].y; //Im
    }
    t = t+ blockDim.x*gridDim.x;
  }
}

void performSensMul(CufftType* imdata_d,
  DType2* sens_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host,
  bool conjugate)
{
  if (DEBUG)
    printf("perform sensitivity multiplication \n");

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  //dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 block_dim(64, 1, 8);
  if (conjugate)
    conjSensMulKernel<<<grid_dim,block_dim>>>(imdata_d,sens_d,gi_host->im_width_dim);
  else
    sensMulKernel<<<grid_dim,block_dim>>>(imdata_d,sens_d,gi_host->im_width_dim);
}

__global__ void sensSumKernel(CufftType* imdata, DType2* imdata_sum, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    for (int c = 0; c < GI.n_coils_cc; c ++)
    {
      CufftType data_p = imdata[t + c*N]; 
      imdata_sum[t].x += data_p.x; // Re
      imdata_sum[t].y += data_p.y; // Im
    }
    t = t + blockDim.x*gridDim.x;
  }
}

void performSensSum(CufftType* imdata_d,
  CufftType* imdata_sum_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  if (DEBUG)
    printf("perform sens coil summation\n");

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);

  sensSumKernel<<<grid_dim,block_dim>>>(imdata_d,imdata_sum_d,gi_host->im_width_dim);
}

__global__ void densityCompensationKernel(DType2* data, DType* density_comp, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      DType2 data_p = data[t + c*N]; 
      data_p.x = data_p.x * density_comp[t];
      data_p.y = data_p.y * density_comp[t];
      data[t + c*N] = data_p;
    }
    t = t+ blockDim.x*gridDim.x;
  }
}

void performDensityCompensation(DType2* data, DType* density_comp, gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  dim3 block_dim(64, 1, 8);
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
      for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
      {
        CufftType gdata_p = gdata[t + c*N]; 
        gdata_p.x = gdata_p.x / deapo;//Re
        gdata_p.y = gdata_p.y / deapo;//Im
        gdata[t + c*N] = gdata_p;
      }
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
      for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
      {
        CufftType gdata_p = gdata[t + c*N]; 
        gdata_p.x = gdata_p.x / deapo;//Re
        gdata_p.y = gdata_p.y / deapo;//Im
        gdata[t + c*N] = gdata_p;
      }
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

    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      imdata[t + c*N] = gdata[grid_ind + c*GI.gridDims_count];
    }
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
    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      imdata[t + c*N] = gdata[grid_ind + c*GI.gridDims_count];
    }
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
    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      CufftType temp = gdata[t + c*GI.gridDims_count];
      gdata[t + c*GI.gridDims_count] = gdata[ind_opp + c*GI.gridDims_count];
      gdata[ind_opp + c*GI.gridDims_count] = temp;
    }

    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void fftShiftKernel(CufftType* gdata, CufftType* outdata, IndType3 offset, int N)
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

    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      outdata[t + c*GI.gridDims_count] = gdata[ind_opp + c*GI.gridDims_count];
    }

    t = t + blockDim.x*gridDim.x;
  }
}

__global__ void fftShiftKernel2D(CufftType* gdata, CufftType* outdata, IndType3 offset, int N)
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

    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      outdata[t + c*GI.gridDims_count] = gdata[ind_opp + c*GI.gridDims_count];
    }

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
    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      CufftType temp = gdata[t + c*GI.gridDims_count];
      gdata[t + c*GI.gridDims_count] = gdata[ind_opp + c*GI.gridDims_count];
      gdata[ind_opp + c*GI.gridDims_count] = temp;
    }

    t = t + blockDim.x*gridDim.x;
  }
}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performDeapodization(CufftType* imdata_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  DType beta = (DType)BETA(gi_host->kernel_width, gi_host->osr);

  //Calculate normalization value
  DType norm_val = I0_BETA(gi_host->kernel_width, gi_host->osr) / (DType)gi_host->kernel_width;

  if (gi_host->is2Dprocessing)
    norm_val = norm_val * norm_val;
  else
    norm_val = norm_val * norm_val * norm_val;

  if (DEBUG)
    printf("running deapodization with norm_val %.2f\n",norm_val);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);
  if (gi_host->is2Dprocessing)
  {
    dim3 block_dim(64, 1, 8);
    deapodizationKernel2D<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
  }
  else
    deapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
}

__global__ void precomputedDeapodizationKernel(CufftType* imdata, DType* deapo, int N)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < N) 
  {
    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      CufftType data_p = imdata[t + c*N]; 
      data_p.x = data_p.x * deapo[t];
      data_p.y = data_p.y * deapo[t];
      imdata[t + c*N] = data_p;
    }
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
  //dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 block_dim(64, 1, 8);
  precomputedDeapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,deapo_d,gi_host->im_width_dim);
}

void precomputeDeapodization(DType* deapo_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  DType beta = (DType)BETA(gi_host->kernel_width, gi_host->osr);

  //Calculate normalization value
  DType norm_val = I0_BETA(gi_host->kernel_width, gi_host->osr) / (DType)gi_host->kernel_width;
  if (gi_host->is2Dprocessing)
    norm_val = norm_val * norm_val;
  else
    norm_val = norm_val * norm_val * norm_val;

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
    printf("start cropping image with offset %u\n",ind_off.x);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);

  if (gi_host->is2Dprocessing)
  {
    dim3 block_dim(64, 1, 8);
    cropKernel2D<<<grid_dim,block_dim>>>(gdata_d,imdata_d,ind_off,gi_host->im_width_dim);
  }
  else
    cropKernel<<<grid_dim,block_dim>>>(gdata_d,imdata_d,ind_off,gi_host->im_width_dim);
}

void performInPlaceFFTShift(CufftType* gdata_d,
  gpuNUFFT::FFTShiftDir shift_dir,
  gpuNUFFT::Dimensions gridDims,
  IndType3 offset,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  int t_width = (int)offset.x;
  if (shift_dir == gpuNUFFT::FORWARD)
  {
    if (gridDims.width % 2)
    {
      t_width = (int)offset.x - 1;
    }
  }
  dim3 grid_dim;
  if (gi_host->is2Dprocessing)
    grid_dim = dim3(getOptimalGridDim((long)(gridDims.width*t_width),THREAD_BLOCK_SIZE));
  else
    grid_dim = dim3(getOptimalGridDim((long)(gridDims.width*gridDims.height),THREAD_BLOCK_SIZE));

  dim3 block_dim(THREAD_BLOCK_SIZE);

  if (gi_host->is2Dprocessing)
  {
    dim3 block_dim(64, 1, 8);
    fftShiftKernel2D<<<grid_dim,block_dim>>>(gdata_d,offset,(int)gridDims.height * t_width);
  }
  else
    fftShiftKernel<<<grid_dim,block_dim>>>(gdata_d,offset,(int)gridDims.count()/2);
}

void performOutOfPlaceFFTShift(CufftType* gdata_d,
  gpuNUFFT::FFTShiftDir shift_dir,
  gpuNUFFT::Dimensions gridDims,
  IndType3 offset,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  dim3 grid_dim;
  if (gi_host->is2Dprocessing)
    grid_dim = dim3(getOptimalGridDim((long)(gridDims.width*gridDims.height),THREAD_BLOCK_SIZE));
  else
    grid_dim = dim3(getOptimalGridDim((long)(gridDims.width*gridDims.height*gridDims.depth),THREAD_BLOCK_SIZE));

  dim3 block_dim(THREAD_BLOCK_SIZE);

  CufftType *copy_gdata_d;
  allocateDeviceMem(&copy_gdata_d,gridDims.count());
  cudaMemcpy(copy_gdata_d,gdata_d,gridDims.count()*sizeof(CufftType),cudaMemcpyDeviceToDevice);

  if (gi_host->is2Dprocessing)
  {
    dim3 block_dim(64, 1, 8);
    fftShiftKernel2D<<<grid_dim,block_dim>>>(copy_gdata_d,gdata_d,offset,(int)gridDims.count());
  }
  else
    fftShiftKernel<<<grid_dim,block_dim>>>(copy_gdata_d,gdata_d,offset,(int)gridDims.count());

  cudaFree(copy_gdata_d);
}

void performFFTShift(CufftType* gdata_d,
  gpuNUFFT::FFTShiftDir shift_dir,
  gpuNUFFT::Dimensions gridDims,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  IndType3 offset;
  if (shift_dir == gpuNUFFT::FORWARD)
  {
    offset.x = (int)ceil((DType)(gridDims.width / (DType)2.0));
    offset.y = (int)ceil((DType)(gridDims.height / (DType)2.0));
    offset.z = (int)ceil((DType)(gridDims.depth / (DType)2.0));
  }
  else
  {
    offset.x = (int)floor((DType)(gridDims.width / (DType)2.0));
    offset.y = (int)floor((DType)(gridDims.height / (DType)2.0));
    offset.z = (int)floor((DType)(gridDims.depth / (DType)2.0));
  }
  
  if (gridDims.width % 2 || 
      gridDims.height % 2 ||
      gridDims.depth % 2)
    performOutOfPlaceFFTShift(gdata_d,shift_dir,gridDims,offset,gi_host);
  else
    performInPlaceFFTShift(gdata_d,shift_dir,gridDims,offset,gi_host);
}

__global__ void forwardDeapodizationKernel(DType2* imdata, DType beta, DType norm_val, int N)
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
      imdata[t].x = imdata[t].x / deapo;//Re
      imdata[t].y = imdata[t].y / deapo; //Im
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
      imdata[t].y = imdata[t].y / deapo; //Im
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

    for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
    {
      gdata[grid_ind + c*GI.gridDims_count].x =  imdata[t + c*N].x;
      gdata[grid_ind + c*GI.gridDims_count].y = imdata[t + c*N].y;
    }
    t = t+ blockDim.x*gridDim.x;
  }
}


//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performForwardDeapodization(DType2* imdata_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host)
{
  DType beta = (DType)BETA(gi_host->kernel_width, gi_host->osr);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);

  //Calculate normalization value
  DType norm_val = I0_BETA(gi_host->kernel_width, gi_host->osr) / (DType)gi_host->kernel_width;

  if (gi_host->is2Dprocessing)
    norm_val = norm_val * norm_val;
  else
    norm_val = norm_val * norm_val * norm_val;

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
  //dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 block_dim(64, 1, 8);

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
    printf("start padding image with offset (%u,%u,%u)\n",ind_off.x,ind_off.y,ind_off.z);

  dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
  dim3 block_dim(THREAD_BLOCK_SIZE);
  if (gi_host->is2Dprocessing)
  {
    dim3 block_dim(64, 1, 8);
    paddingKernel2D<<<grid_dim,block_dim>>>(imdata_d,gdata_d,ind_off,gi_host->im_width_dim);
  }
  else
    paddingKernel<<<grid_dim,block_dim>>>(imdata_d,gdata_d,ind_off,gi_host->im_width_dim);
}

#endif //STD_GPUNUFFT_KERNELS_CU
