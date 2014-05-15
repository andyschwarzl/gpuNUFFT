#ifndef TEXTURE_GRIDDING_KERNELS_CU
#define TEXTURE_GRIDDING_KERNELS_CU

#include "gpuNUFFT_kernels.hpp"
#include "../std_gpuNUFFT_kernels.cu"
#include "cuda_utils.cuh"
#include "precomp_utils.hpp"

// ----------------------------------------------------------------------------
// convolutionKernel: NUFFT^H kernel
//
// Performs the gpuNUFFT step by convolution of sample points with 
// interpolation function and resampling onto grid. 
//
// parameters:
//  * data           : complex input sample points
//  * crds           : coordinates of data points (x,y,z)
//  * gdata          : output grid data 
//  * sectors        : mapping of sample indices according to each sector
//  * sector_centers : coordinates (x,y,z) of sector centers
//  * temp_gdata     : temporary grid data
//  * N              : number of threads
__device__ void textureConvolutionFunction(DType2* sdata, int sec, int sec_cnt, int sec_max, int sec_offset, DType2* data, DType* crds, CufftType* gdata, IndType* sectors, IndType* sector_centers,DType2* temp_gdata)
{
  int ind, k, i, j;
  __shared__ int max_dim, imin, imax,jmin,jmax,kmin,kmax;

  DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

  __shared__ IndType3 center;
  center.x = sector_centers[sec * 3];
  center.y = sector_centers[sec * 3 + 1];
  center.z = sector_centers[sec * 3 + 2];//+ GI.aniso_z_shift;

  //Grid Points over threads
  int data_cnt;
  data_cnt = sectors[sec] + sec_offset;

  max_dim =  GI.sector_pad_max;		

  while (data_cnt < sec_max)
  {
    __shared__ DType3 data_point; //datapoint shared in every thread
    data_point.x = crds[data_cnt];
    data_point.y = crds[data_cnt +GI.data_count];
    data_point.z = crds[data_cnt +2*GI.data_count];
    // set the boundaries of final dataset for gpuNUFFT this point
    ix = static_cast<DType>((data_point.x + 0.5) * (GI.gridDims.x) - center.x + GI.sector_offset);
    set_minmax(&ix, &imin, &imax, max_dim, GI.kernel_radius);
    jy = static_cast<DType>((data_point.y + 0.5) * (GI.gridDims.y) - center.y + GI.sector_offset);
    set_minmax(&jy, &jmin, &jmax, max_dim, GI.kernel_radius);
    // take resolution in x(y) direction to keep isotropic voxel size
    //kz = static_cast<DType>((data_point.z + 0.5 - GI.aniso_z_shift) * (GI.gridDims.x) - center.z + GI.sector_offset);
    kz = static_cast<DType>((data_point.z + 0.5) * (GI.gridDims.z) - center.z + GI.sector_offset);
    set_minmax(&kz, &kmin, &kmax, max_dim, GI.kernel_radius);

    // grid this point onto the neighboring cartesian points
    for (k=threadIdx.z;k<=kmax; k += blockDim.z)
    {
      if (k<=kmax && k>=kmin)
      {
        //kz = static_cast<DType>((k + center.z - GI.sector_offset)) / static_cast<DType>((GI.gridDims.x)) - 0.5f + GI.aniso_z_shift;
        kz = static_cast<DType>((k + center.z - GI.sector_offset)) / static_cast<DType>((GI.gridDims.z)) - 0.5f;
        // scale distance in z direction with x,y dimension
        dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
        dz_sqr *= dz_sqr;

        j=threadIdx.y;
        if (j<=jmax && j>=jmin)
        {
          jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.gridDims.y)) - 0.5f;
          dy_sqr = jy - data_point.y;
          dy_sqr *= dy_sqr;
          i=threadIdx.x;

          if (i<=imax && i>=imin)
          {
            ix = static_cast<DType>(i + center.x - GI.sector_offset) / static_cast<DType>((GI.gridDims.x)) - 0.5f;
            dx_sqr = ix - data_point.x;
            dx_sqr *= dx_sqr;

            //get kernel value from texture
            val = computeTextureLookup(dx_sqr*GI.radiusSquared_inv,dy_sqr*GI.radiusSquared_inv,dz_sqr*GI.radiusSquared_inv);

            ind = getIndex(i,j,k,GI.sector_pad_width);

            // multiply data by current kernel val 
            // grid complex or scalar 
            //sdata[ind].x += val * data[data_cnt].x;
            //sdata[ind].y += val * data[data_cnt].y;
            sdata[ind].x += val * tex1Dfetch(texDATA,data_cnt).x;
            sdata[ind].y += val * tex1Dfetch(texDATA,data_cnt).y;
          } // x
        } // y
      } // z
    }//for loop over z entries
    __syncthreads();	
    data_cnt++;
  } //grid points per sector
  __syncthreads();	
  //write shared data to temporary output grid
  int sector_ind_offset = sec_cnt * GI.sector_dim;

  for (k=threadIdx.z;k<GI.sector_pad_width; k += blockDim.z)
  {
    i=threadIdx.x;
    j=threadIdx.y;

    int s_ind = getIndex(i,j,k,GI.sector_pad_width) ;//index in shared grid
    ind = sector_ind_offset + s_ind;//index in temp output grid

    temp_gdata[ind].x = sdata[s_ind].x;//Re
    temp_gdata[ind].y = sdata[s_ind].y;//Im
    __syncthreads();
    sdata[s_ind].x = (DType)0.0;
    sdata[s_ind].y = (DType)0.0;
    __syncthreads();
  }
}

__global__ void textureConvolutionKernel(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  DType2* temp_gdata,
  int N
  )
{
  extern __shared__ DType2 sdata[];//externally managed shared memory

  int sec;
  sec = blockIdx.x;
  //init shared memory
  for (int z=threadIdx.z;z<GI.sector_pad_width; z += blockDim.z)
  {
    int y=threadIdx.y;
    int x=threadIdx.x;
    int s_ind = getIndex(x,y,z,GI.sector_pad_width) ;
    sdata[s_ind].x = 0.0f;//Re
    sdata[s_ind].y = 0.0f;//Im
  }
  __syncthreads();
  //start convolution
  while (sec < N)
  {
    convolutionFunction(sdata,sec,sec,sectors[sec+1],0,data,crds,gdata,sectors,sector_centers,temp_gdata);
    __syncthreads();
    sec = sec + gridDim.x;
  }//sec < sector_count

}

__device__ void textureConvolutionFunction2D(DType2* sdata, int sec, int sec_cnt, int sec_max, int sec_offset, DType2* data, DType* crds, CufftType* gdata, IndType* sectors, IndType* sector_centers, DType2* temp_gdata)
{
    int ind, i, j;
  __shared__ int max_dim, imin, imax,jmin,jmax;

  DType dx_sqr, dy_sqr, val, ix, jy;

  __shared__ IndType2 center;
  center.x = sector_centers[sec * 2];
  center.y = sector_centers[sec * 2 + 1];

  //Grid Points over threads
  int data_cnt;
  data_cnt = sectors[sec] + sec_offset;

  max_dim =  GI.sector_pad_max;		
  while (data_cnt < sec_max)
  {
    __shared__ DType2 data_point; //datapoint shared in every thread
    data_point.x = crds[data_cnt];
    data_point.y = crds[data_cnt +GI.data_count];
    // set the boundaries of final dataset for gpuNUFFT this point
    ix = (data_point.x + 0.5f) * (GI.gridDims.x) - center.x + GI.sector_offset;
    set_minmax(&ix, &imin, &imax, max_dim, GI.kernel_radius);
    jy = (data_point.y + 0.5f) * (GI.gridDims.y) - center.y + GI.sector_offset;
    set_minmax(&jy, &jmin, &jmax, max_dim, GI.kernel_radius);

    // grid this point onto the neighboring cartesian points
    j=threadIdx.y;
    if (j<=jmax && j>=jmin)
    {
      jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.gridDims.y)) - 0.5f;   
      dy_sqr = jy - data_point.y;
      dy_sqr *= dy_sqr;

      i=threadIdx.x;

      if (i<=imax && i>=imin)
      {
        ix = static_cast<DType>(i + center.x - GI.sector_offset) / static_cast<DType>((GI.gridDims.x)) - 0.5f;
        dx_sqr = ix - data_point.x;
        dx_sqr *= dx_sqr;
        //get kernel value from texture
        val = computeTextureLookup(dx_sqr*GI.radiusSquared_inv,dy_sqr*GI.radiusSquared_inv);

        ind = getIndex2D(i,j,GI.sector_pad_width);

        // multiply data by current kernel val 
        // grid complex or scalar 
        //sdata[ind].x += val * data[data_cnt].x;
        //sdata[ind].y += val * data[data_cnt].y;
        sdata[ind].x += val * tex1Dfetch(texDATA,data_cnt).x;
        sdata[ind].y += val * tex1Dfetch(texDATA,data_cnt).y;
      } // x 	 
    } // y 
    __syncthreads();	
    data_cnt++;
  } //grid points per sector
  __syncthreads();	

  //write shared data to temporary output grid
  int sector_ind_offset = sec_cnt * GI.sector_dim;

  i=threadIdx.x;
  j=threadIdx.y;

  int s_ind = getIndex2D(i,j,GI.sector_pad_width) ;//index in shared grid
  ind = sector_ind_offset + s_ind;//index in temp output grid

  temp_gdata[ind].x = sdata[s_ind].x;//Re
  temp_gdata[ind].y = sdata[s_ind].y;//Im

  __syncthreads();
  sdata[s_ind].x = (DType)0.0;
  sdata[s_ind].y = (DType)0.0;
}

__global__ void textureConvolutionKernel2D(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  DType2* temp_gdata,
  int N
  )
{
  extern __shared__ DType2 sdata[];//externally managed shared memory

  int sec;
  sec = blockIdx.x;
  //init shared memory
  int y=threadIdx.y;
  int x=threadIdx.x;
  int s_ind = getIndex2D(x,y,GI.sector_pad_width) ;
  sdata[s_ind].x = 0.0f;//Re
  sdata[s_ind].y = 0.0f;//Im
  __syncthreads();
  //start convolution
  while (sec < N)
  {
    convolutionFunction2D(sdata,sec,sec,sectors[sec+1],0,data,crds,gdata,sectors,sector_centers,temp_gdata);
    __syncthreads();
    sec = sec + gridDim.x;
  }//sec < sector_count
}

void performTextureConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType* kernel_d, 
  IndType* sectors_d, 
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  )
{
  DType2* temp_gdata_d;
  int temp_grid_count = gi_host->sector_count * gi_host->sector_dim;
  if (DEBUG)
    printf("allocate temp grid data of size %d...\n",temp_grid_count);
  allocateDeviceMem<DType2>(&temp_gdata_d,temp_grid_count);

  long shared_mem_size = gi_host->sector_dim*sizeof(DType2);

  //TODO third dimension > 1?
  dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,1);
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,(gi_host->sector_pad_width)*(gi_host->sector_pad_width)*(1)));
  if (DEBUG)
    printf("convolution requires %d bytes of shared memory!\n",shared_mem_size);

  if (gi_host->is2Dprocessing)
    textureConvolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,temp_gdata_d,gi_host->sector_count);
  else
    textureConvolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,temp_gdata_d,gi_host->sector_count);

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error at adj thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));
  //compose total output from local blocks 
  composeOutput(temp_gdata_d,gdata_d,sector_centers_d,gi_host,NULL,gi_host->sector_count);

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error at adj thread synchronization 3: %s\n",cudaGetErrorString(cudaGetLastError()));

  freeDeviceMem((void*)temp_gdata_d);
}


// ----------------------------------------------------------------------------
// forwardConvolutionKernel: NUFFT kernel
//
// Performs the inverse gpuNUFFT step by convolution of grid points with 
// interpolation function and resampling onto trajectory. 
//
// parameters:
//  * data           : complex output sample points
//  * crds           : coordinates of data points (x,y,z)
//  * gdata          : input grid data 
//  * sectors        : mapping of sample indices according to each sector
//  * sector_centers : coordinates (x,y,z) of sector centers
//  * N              : number of threads
__global__ void textureForwardConvolutionKernel(CufftType* data, 
  DType*     crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared_out_data[];//externally managed shared memory

  __shared__ int sec;
  sec = blockIdx.x;

  //init shared memory
  shared_out_data[threadIdx.x].x = 0.0f;//Re
  shared_out_data[threadIdx.x].y = 0.0f;//Im
  __syncthreads();
  //start convolution
  while (sec < N)
  {
    int ind, imin, imax, jmin, jmax,kmin,kmax, k, i, j;
    DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

    __shared__ IndType3 center;
    center.x = sector_centers[sec * 3];
    center.y = sector_centers[sec * 3 + 1];
    center.z = sector_centers[sec * 3 + 2];

    //Grid Points over Threads
    int data_cnt = sectors[sec] + threadIdx.x;

    __shared__ int sector_ind_offset;
    sector_ind_offset = computeXYZ2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.gridDims);

    while (data_cnt < sectors[sec+1])
    {
      DType3 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt + GI.data_count];
      data_point.z = crds[data_cnt + 2*GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = (data_point.x + 0.5f) * (GI.gridDims.x) - center.x + GI.sector_offset;
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = (data_point.y + 0.5f) * (GI.gridDims.x) - center.y + GI.sector_offset;
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
      kz = (data_point.z + 0.5f) * (GI.gridDims.z) - center.z + GI.sector_offset;
      set_minmax(&kz, &kmin, &kmax, GI.sector_pad_max, GI.kernel_radius);

      // convolve neighboring cartesian points to this data point
      k = kmin;
      while (k<=kmax && k>=kmin)
      {
        kz = static_cast<DType>((k + center.z - GI.sector_offset)) / static_cast<DType>((GI.gridDims.z)) - 0.5f;//(k - center_z) *width_inv;
        dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
        dz_sqr *= dz_sqr;

        j=jmin;
        while (j<=jmax && j>=jmin)
        {
          jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.gridDims.y)) - 0.5f;   //(j - center_y) *width_inv;
          dy_sqr = jy - data_point.y;
          dy_sqr *= dy_sqr;

          i=imin;								
          while (i<=imax && i>=imin)
          {
            ix = static_cast<DType>(i + center.x - GI.sector_offset) / static_cast<DType>((GI.gridDims.x)) - 0.5f;// (i - center_x) *width_inv;
            dx_sqr = ix - data_point.x;
            dx_sqr *= dx_sqr;
            // get kernel value from texture
            val = computeTextureLookup(dx_sqr*GI.radiusSquared_inv,dy_sqr*GI.radiusSquared_inv,dz_sqr*GI.radiusSquared_inv);

            // multiply data by current kernel val 
            // grid complex or scalar 
            if (isOutlier(i,j,k,center.x,center.y,center.z,GI.gridDims,GI.sector_offset))
              //calculate opposite index
              ind = computeXYZ2Lin(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
              calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
              calculateOppositeIndex(k,center.z,GI.gridDims.z,GI.sector_offset),
              GI.gridDims);
            else
              ind = (sector_ind_offset + computeXYZ2Lin(i,j,k,GI.gridDims));

            //shared_out_data[threadIdx.x].x += gdata[ind].x * val; 
            //shared_out_data[threadIdx.x].y += gdata[ind].y * val;	
            shared_out_data[threadIdx.x].x += tex1Dfetch(texGDATA,ind).x * val; 
            shared_out_data[threadIdx.x].y += tex1Dfetch(texGDATA,ind).y * val;	
            
            i++;
          } // x loop
          j++;
        } // y loop
        k++;
      } // z loop
      data[data_cnt].x = shared_out_data[threadIdx.x].x;
      data[data_cnt].y = shared_out_data[threadIdx.x].y;

      data_cnt = data_cnt + blockDim.x;

      shared_out_data[threadIdx.x].x = (DType)0.0;//Re
      shared_out_data[threadIdx.x].y = (DType)0.0;//Im
    } //data points per sector
    __syncthreads();
    sec = sec + gridDim.x;
  } //sector check
}

__global__ void textureForwardConvolutionKernel2D(CufftType* data, 
  DType*     crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared_out_data[];//externally managed shared memory

  __shared__ int sec;
  sec = blockIdx.x;

  //init shared memory
  shared_out_data[threadIdx.x].x = 0.0f;//Re
  shared_out_data[threadIdx.x].y = 0.0f;//Im
  __syncthreads();
  //start convolution
  while (sec < N)
  {
    int ind, imin, imax, jmin, jmax, i, j;
    DType dx_sqr, dy_sqr, val, ix, jy;

    __shared__ IndType2 center;
    center.x = sector_centers[sec * 2];
    center.y = sector_centers[sec * 2 + 1];

    //Grid Points over Threads
    int data_cnt = sectors[sec] + threadIdx.x;

    __shared__ int sector_ind_offset;
    sector_ind_offset = computeXY2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,GI.gridDims);

    while (data_cnt < sectors[sec+1])
    {
      DType2 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt + GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = (data_point.x + 0.5f) * (GI.gridDims.x) - center.x + GI.sector_offset;
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = (data_point.y + 0.5f) * (GI.gridDims.x) - center.y + GI.sector_offset;
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);

      // convolve neighboring cartesian points to this data point
      j=jmin;
      while (j<=jmax && j>=jmin)
      {
        jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.gridDims.x)) - 0.5f;   //(j - center_y) *width_inv;
        dy_sqr = jy - data_point.y;
        dy_sqr *= dy_sqr;
        i=imin;								
        while (i<=imax && i>=imin)
        {
          ix = static_cast<DType>(i + center.x - GI.sector_offset) / static_cast<DType>((GI.gridDims.y)) - 0.5f;// (i - center_x) *width_inv;
          dx_sqr = ix - data_point.x;
          dx_sqr *= dx_sqr;
          // get kernel value
          // calc as separable filter
          val = computeTextureLookup(dx_sqr*GI.radiusSquared_inv,dy_sqr*GI.radiusSquared_inv);

          // multiply data by current kernel val 
          // grid complex or scalar 
          if (isOutlier2D(i,j,center.x,center.y,GI.gridDims.x,GI.sector_offset))
            //calculate opposite index
            ind = getIndex2D(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
            calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
            GI.gridDims.x);
          else
            ind = (sector_ind_offset + getIndex2D(i,j,GI.gridDims.x));

          //shared_out_data[threadIdx.x].x += gdata[ind].x * val; 
          //shared_out_data[threadIdx.x].y += gdata[ind].y * val;
          shared_out_data[threadIdx.x].x += tex1Dfetch(texGDATA,ind).x * val; 
          shared_out_data[threadIdx.x].y += tex1Dfetch(texGDATA,ind).y * val;	
          i++;
        } // x loop
        j++;
      } // y loop
      data[data_cnt].x = shared_out_data[threadIdx.x].x;
      data[data_cnt].y = shared_out_data[threadIdx.x].y;

      data_cnt = data_cnt + blockDim.x;

      shared_out_data[threadIdx.x].x = (DType)0.0;//Re
      shared_out_data[threadIdx.x].y = (DType)0.0;//Im
    } //data points per sector
    __syncthreads();
    sec = sec + gridDim.x;
  } //sector check
}

void performTextureForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo*	gi_host
  )
{
  int thread_size = THREAD_BLOCK_SIZE;
  long shared_mem_size = thread_size * sizeof(CufftType);//empiric

  dim3 block_dim(thread_size);
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,thread_size));

  if (DEBUG)
    printf("convolution requires %d bytes of shared memory!\n",shared_mem_size);
  if (gi_host->is2Dprocessing)
    textureForwardConvolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
  else
    textureForwardConvolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
}

#endif //TEXTURE_GRIDDING_KERNELS_CU