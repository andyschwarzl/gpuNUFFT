#ifndef GPUNUFFT_KERNELS_CU
#define GPUNUFFT_KERNELS_CU

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
__device__ void convolutionFunction(DType2* sdata, int sec, int sec_cnt, int sec_max, int sec_offset, DType2* data, DType* crds, CufftType* gdata, IndType* sectors, IndType* sector_centers,DType2* temp_gdata)
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
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, max_dim, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, max_dim, GI.kernel_radius);
      // take resolution in x(y) direction to keep isotropic voxel size
      kz = mapKSpaceToGrid(data_point.z,GI.gridDims.z,center.z,GI.sector_offset);
      set_minmax(&kz, &kmin, &kmax, max_dim, GI.kernel_radius);
      
      // grid this point onto the neighboring cartesian points
      for (k=threadIdx.z;k<=kmax; k += blockDim.z)
      {
        if (k<=kmax && k>=kmin)
        {
          kz = mapGridToKSpace(k,GI.gridDims.z,center.z,GI.sector_offset);
          // scale distance in z direction with x,y dimension
          dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
          dz_sqr *= dz_sqr;
          if (dz_sqr < GI.radiusSquared)
          {
            j=threadIdx.y;
            if (j<=jmax && j>=jmin)
            {
              jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
              dy_sqr = (jy - data_point.y)*GI.aniso_y_scale;
              dy_sqr *= dy_sqr;
              if (dy_sqr < GI.radiusSquared)	
              {
                i=threadIdx.x;

                if (i<=imax && i>=imin)
                {
                  ix = mapGridToKSpace(i,GI.gridDims.x,center.x,GI.sector_offset);
                  dx_sqr = (ix - data_point.x)*GI.aniso_x_scale;
                  dx_sqr *= dx_sqr;
                  if (dx_sqr < GI.radiusSquared)	
                  {
                    //get kernel value
                    //Calculate Separable Filters 
                    val = KERNEL[(int) round(dz_sqr * GI.dist_multiplier)] *
                      KERNEL[(int) round(dy_sqr * GI.dist_multiplier)] *
                      KERNEL[(int) round(dx_sqr * GI.dist_multiplier)];
                    ind = getIndex(i,j,k,GI.sector_pad_width);

                    // multiply data by current kernel val 
                    // grid complex or scalar 
                    sdata[ind].x += val * data[data_cnt].x;
                    sdata[ind].y += val * data[data_cnt].y;
                  }  // kernel bounds check x, spherical support 
                } // x 	 
              }// kernel bounds check y, spherical support 
            } // y 
          } //kernel bounds check z 
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

__global__ void convolutionKernel(DType2* data, 
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

__global__ void balancedConvolutionKernel(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors,
  IndType2* sector_processing_order,
  IndType* sector_centers,
  DType2* temp_gdata,
  int N
  )
{
  extern __shared__ DType2 sdata[];//externally managed shared memory

  int sec_cnt = blockIdx.x;
  int sec;
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
  while (sec_cnt < N)
  {
    sec = sector_processing_order[sec_cnt].x;
    convolutionFunction(sdata,sec,sec_cnt,min(sectors[sec+1],sectors[sec]+sector_processing_order[sec_cnt].y+MAXIMUM_PAYLOAD),sector_processing_order[sec_cnt].y,data,crds,gdata,sectors,sector_centers,temp_gdata);
    __syncthreads();
    sec_cnt = sec_cnt + gridDim.x;
  }//sec < sector_count
}

__device__ void convolutionFunction2D(DType2* sdata, int sec, int sec_cnt, int sec_max, int sec_offset, DType2* data, DType* crds, CufftType* gdata, IndType* sectors, IndType* sector_centers, DType2* temp_gdata)
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
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, max_dim, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, max_dim, GI.kernel_radius);
      // grid this point onto the neighboring cartesian points
      j=threadIdx.y;
      if (j<=jmax && j>=jmin)
      {
        jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
        dy_sqr = (jy - data_point.y)*GI.aniso_y_scale;
        dy_sqr *= dy_sqr;
        if (dy_sqr < GI.radiusSquared)	
        {
          i=threadIdx.x;

          if (i<=imax && i>=imin)
          {
            ix = mapGridToKSpace(i,GI.gridDims.x,center.x,GI.sector_offset);
            dx_sqr = (ix - data_point.x)*GI.aniso_x_scale;
            dx_sqr *= dx_sqr;
            if (dx_sqr <= GI.radiusSquared)	
            {
              //get kernel value
              //Calculate Separable Filters 
              val = KERNEL[(int) round(dy_sqr * GI.dist_multiplier)] *
                KERNEL[(int) round(dx_sqr * GI.dist_multiplier)];
              ind = getIndex2D(i,j,GI.sector_pad_width);

              // multiply data by current kernel val 
              // grid complex or scalar 
              int c = 0;
              while (c < GI.n_coils_cc)
              {
                sdata[ind + c*GI.sector_dim].x += val * data[data_cnt + c*GI.data_count].x;
                sdata[ind + c*GI.sector_dim].y += val * data[data_cnt + c*GI.data_count].y;
                c++;
              }
            } // kernel bounds check x, spherical support 
          } // x 	 
        } // kernel bounds check y, spherical support 
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

    int c = 0;
    while (c < GI.n_coils_cc)
    {
      temp_gdata[ind + c*GI.sectorsToProcess * GI.sector_dim].x = sdata[s_ind + c*GI.sector_dim].x;//Re
      temp_gdata[ind + c*GI.sectorsToProcess * GI.sector_dim].y = sdata[s_ind + c*GI.sector_dim].y;//Im

      sdata[s_ind + c*GI.sector_dim].x = (DType)0.0;
      sdata[s_ind + c*GI.sector_dim].y = (DType)0.0;
      c++;
    }
    __syncthreads();
}

__global__ void convolutionKernel2D(DType2* data, 
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
  int c=0;
  while (c<GI.n_coils_cc)
  {
    sdata[s_ind + c*GI.sector_dim].x = 0.0f;//Re
    sdata[s_ind + c*GI.sector_dim].y = 0.0f;//Im
    c++;
  }
  __syncthreads();
  //start convolution
  while (sec < N)
  {
    convolutionFunction2D(sdata,sec,sec,sectors[sec+1],0,data,crds,gdata,sectors,sector_centers,temp_gdata);
    __syncthreads();
    sec = sec + gridDim.x;
  }//sec < sector_count
}

__global__ void balancedConvolutionKernel2D(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType2* sector_processing_order,
  IndType* sector_centers,
  DType2* temp_gdata,
  int N
  )
{
  extern __shared__ DType2 sdata[];//externally managed shared memory
  
  int sec_cnt = blockIdx.x;
  int sec;

  //init shared memory
  int y=threadIdx.y;
  int x=threadIdx.x;
  int s_ind = getIndex2D(x,y,GI.sector_pad_width) ;

  int c=0;
  while (c<GI.n_coils_cc)
  {
    sdata[s_ind + c*GI.sector_dim].x = 0.0f;//Re
    sdata[s_ind + c*GI.sector_dim].y = 0.0f;//Im
    c++;
  }

  __syncthreads();
  //start convolution
  while (sec_cnt < N)
  {
    sec = sector_processing_order[sec_cnt].x;
    convolutionFunction2D(sdata,sec,sec_cnt,min(sectors[sec+1],sectors[sec]+sector_processing_order[sec_cnt].y+MAXIMUM_PAYLOAD),sector_processing_order[sec_cnt].y,data,crds,gdata,sectors,sector_centers,temp_gdata);
    __syncthreads();
    sec_cnt = sec_cnt + gridDim.x;
  }//sec < sector_count
}


__global__ void composeOutputKernel(DType2* temp_gdata, CufftType* gdata, IndType* sector_centers, IndType2* sector_processing_order, int N)
{
  int sec;
  for (int sec_cnt = 0; sec_cnt < N; sec_cnt++)
  {
    if (sector_processing_order != NULL)
      sec = sector_processing_order[sec_cnt].x;
    else
      sec = sec_cnt;
    __syncthreads();
    IndType3 center;
    center.x = sector_centers[sec * 3];
    center.y = sector_centers[sec * 3 + 1];
    center.z = sector_centers[sec * 3 + 2];
    int sector_ind_offset;

    sector_ind_offset = computeXYZ2Lin((int)center.x - GI.sector_offset,(int)center.y - GI.sector_offset,(int)center.z - GI.sector_offset,GI.gridDims);

     int sector_grid_offset;
    sector_grid_offset = sec_cnt * GI.sector_dim;
    //write data from temp grid to overall output grid
    for (int z=threadIdx.z;z<GI.sector_pad_width; z += blockDim.z)
    {
      int x=threadIdx.x;
      int y=threadIdx.y;
      int s_ind = (sector_grid_offset + getIndex(x,y,z,GI.sector_pad_width));

      int ind;
      if (isOutlier(x,y,z,center.x,center.y,center.z,GI.gridDims,GI.sector_offset))
      {
        //calculate opposite index
        ind = computeXYZ2Lin(calculateOppositeIndex(x,center.x,GI.gridDims.x,GI.sector_offset),
            calculateOppositeIndex(y,center.y,GI.gridDims.y,GI.sector_offset),
            calculateOppositeIndex(z,center.z,GI.gridDims.z,GI.sector_offset),
            GI.gridDims);

        gdata[ind].x += temp_gdata[s_ind].x;//Re
        gdata[ind].y += temp_gdata[s_ind].y;//Im
      }
      else
      {
        ind = (sector_ind_offset + computeXYZ2Lin(x,y,z,GI.gridDims));
        gdata[ind].x += temp_gdata[s_ind].x;//Re
        gdata[ind].y += temp_gdata[s_ind].y;//Im
      }
    }
  }
}

__global__ void composeOutputKernel2D(DType2* temp_gdata, CufftType* gdata, IndType* sector_centers, IndType2* sector_processing_order, int N)
{
  int sec;
  for (int sec_cnt = 0; sec_cnt < N; sec_cnt++)
  {
    if (sector_processing_order != NULL)
      sec = sector_processing_order[sec_cnt].x;
    else
      sec = sec_cnt;

    __syncthreads();
    __shared__ IndType2 center;
    center.x = sector_centers[sec * 2];
    center.y = sector_centers[sec * 2 + 1];
    __shared__ int sector_ind_offset;
    sector_ind_offset = computeXY2Lin((int)center.x - GI.sector_offset,(int)center.y - GI.sector_offset,GI.gridDims);
    __shared__ int sector_grid_offset;
    sector_grid_offset = sec_cnt * GI.sector_dim;
    //write data from temp grid to overall output grid
    int x=threadIdx.x;
    int y=threadIdx.y;
    int s_ind = (sector_grid_offset + getIndex2D(x,y,GI.sector_pad_width));

    int ind;
    if (isOutlier2D(x,y,center.x,center.y,GI.gridDims,GI.sector_offset))
      //calculate opposite index
      ind = computeXY2Lin(calculateOppositeIndex(x,center.x,GI.gridDims.x,GI.sector_offset),
      calculateOppositeIndex(y,center.y,GI.gridDims.y,GI.sector_offset),
      GI.gridDims);
    else
      ind = (sector_ind_offset + getIndex2D(x,y,GI.gridDims.x));

    int c=0;
    while (c < GI.n_coils_cc)
    {
      gdata[ind + c*GI.gridDims_count].x += temp_gdata[s_ind + c * GI.sectorsToProcess * GI.sector_dim].x;//Re
      gdata[ind + c*GI.gridDims_count].y += temp_gdata[s_ind + c * GI.sectorsToProcess * GI.sector_dim].y;//Im			
      c++;
    }
  }
}


//very slow way of composing the output, should only be used on compute capabilties lower than 2.0
void composeOutput(DType2* temp_gdata_d, CufftType* gdata_d, IndType* sector_centers_d, gpuNUFFT::GpuNUFFTInfo* gi_host, IndType2* sector_processing_order_d, int N)
{
  dim3 grid_dim(1);
  dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,1);
  if (gi_host->is2Dprocessing)
    composeOutputKernel2D<<<grid_dim,block_dim>>>(temp_gdata_d,gdata_d,sector_centers_d,sector_processing_order_d,N);
  else
    composeOutputKernel<<<grid_dim,block_dim>>>(temp_gdata_d,gdata_d,sector_centers_d,sector_processing_order_d,N);
}

void performConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType* kernel_d, 
  IndType* sectors_d, 
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  )
{
  DType2* temp_gdata_d;
  int temp_grid_count = gi_host->sector_count * gi_host->sector_dim * gi_host->n_coils_cc;
  if (DEBUG)
    printf("allocate temp grid data of size %d...\n",temp_grid_count);
  allocateDeviceMem<DType2>(&temp_gdata_d,temp_grid_count);

  long shared_mem_size = gi_host->sector_dim*sizeof(DType2) * gi_host->n_coils_cc;

  dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,1);
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,(gi_host->sector_pad_width)*(gi_host->sector_pad_width)*(1)));
  if (DEBUG)
    printf("convolution requires %ld bytes of shared memory!\n",shared_mem_size);

  if (gi_host->is2Dprocessing)
    convolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,temp_gdata_d,gi_host->sector_count);
  else
    convolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,temp_gdata_d,gi_host->sector_count);

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error at adj thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));
  //compose total output from local blocks 
  composeOutput(temp_gdata_d,gdata_d,sector_centers_d,gi_host,NULL,gi_host->sector_count);
  
  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error at adj thread synchronization 3: %s\n",cudaGetErrorString(cudaGetLastError()));
  
  freeDeviceMem((void*)temp_gdata_d);
}


void performConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType* kernel_d, 
  IndType* sectors_d,
  IndType2* sector_processing_order_d,
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  )
{
  DType2* temp_gdata_d;
  int temp_grid_count = gi_host->sectorsToProcess * gi_host->sector_dim * gi_host->n_coils_cc;
  if (DEBUG)
    printf("allocate temp grid data of size %d...\n",temp_grid_count);
  allocateDeviceMem<DType2>(&temp_gdata_d,temp_grid_count);

  long shared_mem_size = gi_host->sector_dim * sizeof(DType2) * gi_host->n_coils_cc;

  dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,1);
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,(gi_host->sector_pad_width)*(gi_host->sector_pad_width)*(1)));
  if (DEBUG)
    printf("convolution requires %ld bytes of shared memory!\n",shared_mem_size);

  if (gi_host->is2Dprocessing)
    balancedConvolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_processing_order_d,sector_centers_d,temp_gdata_d,gi_host->sectorsToProcess);
  else
    balancedConvolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_processing_order_d,sector_centers_d,temp_gdata_d,gi_host->sectorsToProcess);

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error at adj thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));
  //compose total output from local blocks 
  composeOutput(temp_gdata_d,gdata_d,sector_centers_d,gi_host,sector_processing_order_d,gi_host->sectorsToProcess);
  
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
__global__ void forwardConvolutionKernel(CufftType* data, 
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
  shared_out_data[threadIdx.x].x = (DType)0.0;//Re
  shared_out_data[threadIdx.x].y = (DType)0.0;//Im
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
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
      kz = mapKSpaceToGrid(data_point.z,GI.gridDims.z,center.z,GI.sector_offset);
      set_minmax(&kz, &kmin, &kmax, GI.sector_pad_max, GI.kernel_radius);

      // convolve neighboring cartesian points to this data point
      k = kmin;
      while (k<=kmax && k>=kmin)
      {
        kz = mapGridToKSpace(k,GI.gridDims.z,center.z,GI.sector_offset);
        dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
        dz_sqr *= dz_sqr;

        if (dz_sqr < GI.radiusSquared)
        {
          j=jmin;
          while (j<=jmax && j>=jmin)
          {
            jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
            dy_sqr = (jy - data_point.y)*GI.aniso_y_scale;
            dy_sqr *= dy_sqr;
            if (dy_sqr < GI.radiusSquared)	
            {
              i=imin;								
              while (i<=imax && i>=imin)
              {
                ix = mapGridToKSpace(i,GI.gridDims.x,center.x,GI.sector_offset);
                dx_sqr = (ix - data_point.x)*GI.aniso_x_scale;
                dx_sqr *= dx_sqr;
                if (dx_sqr < GI.radiusSquared)	
                {
                  // get kernel value
                  // calc as separable filter
                  val = KERNEL[(int) round(dz_sqr * GI.dist_multiplier)] *
                    KERNEL[(int) round(dy_sqr * GI.dist_multiplier)] *
                    KERNEL[(int) round(dx_sqr * GI.dist_multiplier)];

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

                  shared_out_data[threadIdx.x].x += gdata[ind].x * val; 
                  shared_out_data[threadIdx.x].y += gdata[ind].y * val;									
                }// kernel bounds check x, spherical support 
                i++;
              } // x loop
            } // kernel bounds check y, spherical support  
            j++;
          } // y loop
        } //kernel bounds check z 
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

__global__ void forwardConvolutionKernel2(CufftType* data, 
  DType*     crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared_data[];//externally managed shared memory
  CufftType* shared_out_data =(CufftType*) &shared_data[0];  
  CufftType* gdata_cache =(CufftType*) &shared_data[blockDim.x]; 

  int sec;
  sec = blockIdx.x;

  //init shared memory
  shared_out_data[threadIdx.x].x = (DType)0.0;//Re
  shared_out_data[threadIdx.x].y = (DType)0.0;//Im

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

    __shared__ int sector_ind_offset;
    sector_ind_offset = computeXYZ2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.gridDims);

    // init sector cache 
    // preload sector grid data into cache
    for (int ind=threadIdx.x; ind<GI.sector_dim; ind+=blockDim.x)
    {
      int grid_index;
      getCoordsFromIndex(ind,&i,&j,&k,GI.sector_pad_width);

      if (isOutlier(i,j,k,center.x,center.y,center.z,GI.gridDims,GI.sector_offset))
        //calculate opposite index
        grid_index = computeXYZ2Lin(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
        calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
        calculateOppositeIndex(k,center.z,GI.gridDims.z,GI.sector_offset),
        GI.gridDims);
      else
        grid_index = (sector_ind_offset + computeXYZ2Lin(i,j,k,GI.gridDims));

      gdata_cache[ind].x = gdata[grid_index].x;
      gdata_cache[ind].y = gdata[grid_index].y;
    }

    __syncthreads();

    //Grid Points over Threads
    int data_cnt = sectors[sec] + threadIdx.x;

    while (data_cnt < sectors[sec+1])
    {
      DType3 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt + GI.data_count];
      data_point.z = crds[data_cnt + 2*GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
      kz = mapKSpaceToGrid(data_point.z,GI.gridDims.z,center.z,GI.sector_offset);
      set_minmax(&kz, &kmin, &kmax, GI.sector_pad_max, GI.kernel_radius);

      // convolve neighboring cartesian points to this data point
      k = kmin;
      while (k<=kmax && k>=kmin)
      {
        kz = mapGridToKSpace(k,GI.gridDims.z,center.z,GI.sector_offset);
        dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
        dz_sqr *= dz_sqr;

        if (dz_sqr < GI.radiusSquared)
        {
          j=jmin;
          while (j<=jmax && j>=jmin)
          {
            jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
            dy_sqr = (jy - data_point.y)*GI.aniso_y_scale;
            dy_sqr *= dy_sqr;
            if (dy_sqr < GI.radiusSquared)	
            {
              i=imin;								
              while (i<=imax && i>=imin)
              {
                ix = mapGridToKSpace(i,GI.gridDims.x,center.x,GI.sector_offset);
                dx_sqr = (ix - data_point.x)*GI.aniso_x_scale;
                dx_sqr *= dx_sqr;
                if (dx_sqr < GI.radiusSquared)	
                {
                  // get kernel value
                  // calc as separable filter
                  val = KERNEL[(int) round(dz_sqr * GI.dist_multiplier)] *
                    KERNEL[(int) round(dy_sqr * GI.dist_multiplier)] *
                    KERNEL[(int) round(dx_sqr * GI.dist_multiplier)];

                  ind = getIndex(i,j,k,GI.sector_pad_width);

                  shared_out_data[threadIdx.x].x += gdata_cache[ind].x * val; 
                  shared_out_data[threadIdx.x].y += gdata_cache[ind].y * val;									
                }// kernel bounds check x, spherical support 
                i++;
              } // x loop
            } // kernel bounds check y, spherical support  
            j++;
          } // y loop
        } //kernel bounds check z 
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

__global__ void forwardConvolutionKernel2D(CufftType* data, 
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
  shared_out_data[threadIdx.x].x = (DType)0.0;//Re
  shared_out_data[threadIdx.x].y = (DType)0.0;//Im
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
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);

      // convolve neighboring cartesian points to this data point
      j=jmin;
      while (j<=jmax && j>=jmin)
      {
        jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
        dy_sqr = (jy - data_point.y)*GI.aniso_y_scale;
        dy_sqr *= dy_sqr;
        if (dy_sqr < GI.radiusSquared)	
        {
          i=imin;								
          while (i<=imax && i>=imin)
          {
            ix = mapGridToKSpace(i,GI.gridDims.x,center.x,GI.sector_offset);
            dx_sqr = (ix - data_point.x)*GI.aniso_x_scale;
            dx_sqr *= dx_sqr;
            if (dx_sqr < GI.radiusSquared)	
            {
              // get kernel value
              // calc as separable filter
              val = KERNEL[(int) round(dy_sqr * GI.dist_multiplier)] *
                KERNEL[(int) round(dx_sqr * GI.dist_multiplier)];

              // multiply data by current kernel val 
              // grid complex or scalar 
              if (isOutlier2D(i,j,center.x,center.y,GI.gridDims,GI.sector_offset))
                //calculate opposite index
                ind = getIndex2D(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
                calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
                GI.gridDims.x);
              else
                ind = (sector_ind_offset + getIndex2D(i,j,GI.gridDims.x));

              shared_out_data[threadIdx.x].x += gdata[ind].x * val; 
              shared_out_data[threadIdx.x].y += gdata[ind].y * val;									
            }// kernel bounds check x, spherical support 
            i++;
          } // x loop
        } // kernel bounds check y, spherical support  
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

__global__ void forwardConvolutionKernel22D(CufftType* data, 
  DType*     crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared_data[];//externally managed shared memory
  CufftType* shared_out_data =(CufftType*) &shared_data[0];  
  CufftType* gdata_cache =(CufftType*) &shared_data[blockDim.x * GI.n_coils_cc];

  __shared__ int sec;
  sec = blockIdx.x;

  //init shared memory
  for (int c = 0; c < GI.n_coils_cc; c++)
  { 
    shared_out_data[threadIdx.x + c*blockDim.x].x = 0.0f;//Re
    shared_out_data[threadIdx.x + c*blockDim.x].y = 0.0f;//Im
  }
  __syncthreads();
  //start convolution
  while (sec < N)
  {
    int ind, imin, imax, jmin, jmax, i, j;
    DType dx_sqr, dy_sqr, val, ix, jy;

    __shared__ IndType2 center;
    center.x = sector_centers[sec * 2];
    center.y = sector_centers[sec * 2 + 1];

    __shared__ int sector_ind_offset;
    sector_ind_offset = computeXY2Lin((int)center.x - (int)GI.sector_offset,(int)center.y - (int)GI.sector_offset,GI.gridDims);

     // init sector cache 
    // preload sector grid data into cache
    for (int ind=threadIdx.x; ind<GI.sector_dim; ind+=blockDim.x)
    {
      int grid_index;
      getCoordsFromIndex2D(ind,&i,&j,GI.sector_pad_width);

      // multiply data by current kernel val 
      // grid complex or scalar 
      if (isOutlier2D(i,j,center.x,center.y,GI.gridDims,GI.sector_offset))
        //calculate opposite index
        grid_index = getIndex2D(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
        calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
        GI.gridDims.x);
      else
        grid_index = (sector_ind_offset + getIndex2D(i,j,GI.gridDims.x));

      for (int c=0; c < GI.n_coils_cc; c++)
      {
        gdata_cache[ind + c*GI.sector_dim].x = gdata[grid_index + c*GI.gridDims_count].x;
        gdata_cache[ind + c*GI.sector_dim].y = gdata[grid_index + c*GI.gridDims_count].y;
      }
    }
    __syncthreads();

    //Grid Points over Threads
    int data_cnt = sectors[sec] + threadIdx.x;

    while (data_cnt < sectors[sec+1])
    {
      DType2 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt + GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
      
      // convolve neighboring cartesian points to this data point
      j=jmin;
      while (j<=jmax && j>=jmin)
      {
        jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
        dy_sqr = (jy - data_point.y)*GI.aniso_y_scale;
        dy_sqr *= dy_sqr;
        if (dy_sqr < GI.radiusSquared)	
        {
          i=imin;								
          while (i<=imax && i>=imin)
          {
            ix = mapGridToKSpace(i,GI.gridDims.x,center.x,GI.sector_offset);
            dx_sqr = (ix - data_point.x)*GI.aniso_x_scale;
            dx_sqr *= dx_sqr;
            if (dx_sqr < GI.radiusSquared)	
            {
              // get kernel value
              // calc as separable filter
              val = KERNEL[(int) round(dy_sqr * GI.dist_multiplier)] *
                KERNEL[(int) round(dx_sqr * GI.dist_multiplier)];

              ind = getIndex2D(i,j,GI.sector_pad_width);

              for (int c=0; c < GI.n_coils_cc; c++)
              {
                shared_out_data[threadIdx.x + c*blockDim.x].x += gdata_cache[ind + c*GI.sector_dim].x * val; 
                shared_out_data[threadIdx.x + c*blockDim.x].y += gdata_cache[ind + c*GI.sector_dim].y * val;
              }
            }// kernel bounds check x, spherical support 
            i++;
          } // x loop
        } // kernel bounds check y, spherical support  
        j++;
      } // y loop
      for (int c=0; c < GI.n_coils_cc; c++)
      {
        data[data_cnt + c*GI.data_count].x += shared_out_data[threadIdx.x + c*blockDim.x].x;
        data[data_cnt + c*GI.data_count].y += shared_out_data[threadIdx.x + c*blockDim.x].y;
        shared_out_data[threadIdx.x + c*blockDim.x].x = (DType)0.0;//Re
        shared_out_data[threadIdx.x + c*blockDim.x].y = (DType)0.0;//Im
      }

      data_cnt = data_cnt + blockDim.x;

    } //data points per sector
    __syncthreads();
    sec = sec + gridDim.x;
  } //sector check
}

void performForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo*	gi_host
  )
{
  bool useCache = true;
  if (!useCache)
  {
    int thread_size = THREAD_BLOCK_SIZE;
    long shared_mem_size = thread_size * sizeof(CufftType);//empiric

    dim3 block_dim(thread_size);
    dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,1));

    if (DEBUG)
      printf("forward convolution requires %ld bytes of shared memory!\n",shared_mem_size);
    if (gi_host->is2Dprocessing)
      forwardConvolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
    else
      forwardConvolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
  }
  else
  {
    int thread_size = THREAD_BLOCK_SIZE;
    long shared_mem_size = (thread_size + gi_host->sector_dim) * gi_host->n_coils_cc * sizeof(CufftType);

    dim3 block_dim(thread_size);
    dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,1));

    if (DEBUG)
      printf("cached forward convolution requires %ld bytes of shared memory!\n",shared_mem_size);
    if (gi_host->is2Dprocessing)
      forwardConvolutionKernel22D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
    else
      forwardConvolutionKernel2<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
  }
}

void performForwardConvolution( CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d,   
  IndType2* sector_processing_order_d,
  IndType*		sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo*	gi_host
  )
{
  // balanced version in coarse gridding due to lack of atomic operations not possible
  performForwardConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,gi_host);
}

#endif //GPUNUFFT_KERNELS_CU
