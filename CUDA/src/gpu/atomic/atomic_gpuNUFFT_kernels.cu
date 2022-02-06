#ifndef ATOMIC_GPUNUFFT_KERNELS_H
#define ATOMIC_GPUNUFFT_KERNELS_H
#include "gpuNUFFT_kernels.hpp"
#include "../std_gpuNUFFT_kernels.cu"
#include "cuda_utils.cuh"

// convolve every data point on grid position -> controlled by threadIdx.x .y and .z 
// shared data holds grid values as software managed cache
__global__ void convolutionKernel3( DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N,
  int CACHE_SIZE
  )
{
  extern __shared__ DType shared_data[];//externally managed shared memory
  DType2* data_cache =(DType2*)&shared_data[0];  
  DType3* coord_cache =(DType3*) &shared_data[2*CACHE_SIZE]; 
  __shared__ int sec;
  sec = blockIdx.x;
  //start convolution
  while (sec < N)
  {
    int ind, k, i, j;
    int imin, imax, jmin, jmax, kmin, kmax;

    DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

    __shared__ IndType3 center;
    center.x = sector_centers[sec * 3];
    center.y = sector_centers[sec * 3 + 1];
    center.z = sector_centers[sec * 3 + 2];

    //Grid Points over threads, start position of data points of this sector
    __shared__ int data_off;
    data_off = sectors[sec];
    int data_max = sectors[sec+1];
    //init shared memory data cache
    int c_ind = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y* threadIdx.z;

    //load data points into shared mem
    while (c_ind < CACHE_SIZE && (data_off + c_ind) < data_max)
    {
      data_cache[c_ind] = data[data_off + c_ind];
      coord_cache[c_ind].x = crds[c_ind + data_off];
      coord_cache[c_ind].y = crds[c_ind + data_off+GI.data_count];
      coord_cache[c_ind].z = crds[c_ind + data_off+2*GI.data_count];
      c_ind += blockDim.x * blockDim.y*blockDim.z;
    }
    __syncthreads();

    __shared__ int sector_ind_offset;
    sector_ind_offset = computeXYZ2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.gridDims);

    c_ind = 0;
    __shared__ int reload_count;
    reload_count = 0;

    while (data_off+c_ind < data_max)
    {
      if (c_ind >= (reload_count+1)*CACHE_SIZE)
      {
        __syncthreads();
        /*	int reload_ind = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y* threadIdx.z;
        //load next data points into shared mem
        while (reload_ind < CACHE_SIZE && (data_off + c_ind + reload_ind) < data_max)
        {
        data_cache[reload_ind] = data[data_off + c_ind + reload_ind];
        coord_cache[reload_ind].x = crds[c_ind + data_off + reload_ind];
        coord_cache[reload_ind].y = crds[c_ind + data_off+ reload_ind + GI.data_count];
        coord_cache[reload_ind].z = crds[c_ind + data_off+ reload_ind + 2*GI.data_count];
        reload_ind += blockDim.x * blockDim.y * blockDim.z;
        }*/
        reload_count++;
      }
      __syncthreads();
      //			DType3 data_point; //datapoint shared in every thread
      DType3 data_point = coord_cache[c_ind - reload_count*CACHE_SIZE];
      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
      kz = mapKSpaceToGrid(data_point.z,GI.gridDims.z,center.z,GI.sector_offset);
      set_minmax(&kz, &kmin, &kmax, GI.sector_pad_max, GI.kernel_radius);

      // grid this point onto the neighboring cartesian points
      for (k=threadIdx.z;k<=kmax; k += blockDim.z)
      {
        j=threadIdx.y;
        i=threadIdx.x;
        if ((k<=kmax && k>=kmin)
          && (j<=jmax && j>=jmin)
          && (i<=imax && i>=imin))
        {
          kz = mapGridToKSpace(k,GI.gridDims.z,center.z,GI.sector_offset);
          dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
          dz_sqr *= dz_sqr;
          if (dz_sqr < GI.radiusSquared)
          {
            jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
            dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
            dy_sqr *= dy_sqr;
            if (dy_sqr < GI.radiusSquared)	
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
                if (isOutlier(i,j,k,center.x,center.y,center.z,GI.gridDims,GI.sector_offset))
                  //calculate opposite index
                  ind = computeXYZ2Lin(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
                  calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
                  calculateOppositeIndex(k,center.z,GI.gridDims.z,GI.sector_offset),
                  GI.gridDims);
                else
                  ind = sector_ind_offset + computeXYZ2Lin(i,j,k,GI.gridDims);//index in output grid

                atomicAdd(&(gdata[ind].x),val * data_cache[c_ind-reload_count*CACHE_SIZE].x);//Re
                atomicAdd(&(gdata[ind].y),val * data_cache[c_ind-reload_count*CACHE_SIZE].y);//Im
              } // kernel bounds check x, spherical support 
            } // kernel bounds check y, spherical support 
          } //kernel bounds check z 
        } //x,y,z bounds check
      }//for loop over z entries
      c_ind++;
    } //grid points per sector
    __syncthreads();	
    sec = sec + gridDim.x;
  }//sec < sector_count
}

__device__ void convolutionFunction4(int sec, int sec_max, int sec_offset, DType2* data, DType* crds, CufftType* gdata, IndType* sectors, IndType* sector_centers)
{
    int ind, k, i, j;
    __shared__ int max_dim, imin, imax, jmin, jmax, kmin, kmax;

    DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

    __shared__ IndType3 center;
    center.x = sector_centers[sec * 3];
    center.y = sector_centers[sec * 3 + 1];
    center.z = sector_centers[sec * 3 + 2];

    __shared__ int sector_ind_offset;
    sector_ind_offset = computeXYZ2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.gridDims);

    __syncthreads();	
    //Grid Points over threads
    int data_cnt;
    data_cnt = sectors[sec]+sec_offset;

    max_dim =  GI.sector_pad_max;		
    
    int s_ind = getIndex(threadIdx.x,threadIdx.y,threadIdx.z,GI.sector_pad_width);

    while (data_cnt < sec_max)
    {
      __syncthreads();	
      __shared__ DType3 data_point; //datapoint shared in every thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt +GI.data_count];
      data_point.z = crds[data_cnt +2*GI.data_count];

      __shared__ DType2 s_data;
      s_data.x = data[data_cnt].x;
      s_data.y = data[data_cnt].y;

      __syncthreads();	
      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, max_dim, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, max_dim, GI.kernel_radius);
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
              dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
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
                    
                    //each thread writes one position from shared mem to global mem
                    if (isOutlier(i,j,k,center.x,center.y,center.z,GI.gridDims,GI.sector_offset))
                      //calculate opposite index
                      ind = computeXYZ2Lin(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
                          calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
                          calculateOppositeIndex(k,center.z,GI.gridDims.z,GI.sector_offset),
                          GI.gridDims);
                    else
                      ind = sector_ind_offset + computeXYZ2Lin(i,j,k,GI.gridDims);//index in output grid

                    atomicAdd(&(gdata[ind].x),val * s_data.x);//Re
                    atomicAdd(&(gdata[ind].y),val * s_data.y);//Im
                  }  // kernel bounds check x, spherical support 
                } // x 	 
              }// kernel bounds check y, spherical support 
            } // y 
          } //kernel bounds check z 
        } // z
      }//k, for loop over z entries
      __syncthreads();	
      data_cnt++;
    } //grid points per sector
}

__global__ void convolutionKernel4(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N
  )
{
  int sec;
  sec = blockIdx.x;

  //start convolution
  while (sec < N)
  {
	  convolutionFunction4(sec,sectors[sec+1],0,data,crds,gdata,sectors,sector_centers);
    __syncthreads();	
    sec = sec + gridDim.x;
  }//sec < sector_count 
}

__global__ void balancedConvolutionKernel4(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType2* sector_processing_order,
  IndType* sector_centers,
  int N
  )
{
  int sec;
  int sec_cnt = blockIdx.x;

  //start convolution
  while (sec_cnt < N)
  {
    sec = sector_processing_order[sec_cnt].x;
    
    convolutionFunction4(sec,min(sectors[sec+1],sectors[sec]+sector_processing_order[sec_cnt].y+MAXIMUM_PAYLOAD),sector_processing_order[sec_cnt].y,data,crds,gdata,sectors,sector_centers);
    __syncthreads();	
    sec_cnt = sec_cnt + gridDim.x;
  }//sec < sector_count
}

// ----------------------------------------------------------------------------
// convolutionKernel: NUFFT^H kernel
//
// Performs the gpuNUFFT step by convolution of sample points with 
// interpolation function and resampling onto grid. Basic concept based on Zwart
// et al. 
//
// parameters:
//  * data           : complex input sample points
//  * crds           : coordinates of data points (x,y,z)
//  * gdata          : output grid data 
//  * sectors        : mapping of sample indices according to each sector
//  * sector_centers : coordinates (x,y,z) of sector centers
//  * temp_gdata     : temporary grid data
//  * N              : number of threads
__device__ void convolutionFunction2(int* sec, int sec_max, int sec_offset, DType2* sdata, DType2* data, DType* crds, CufftType* gdata, IndType* sectors, IndType* sector_centers)
{
    //init shared memory
    for (int s_ind=threadIdx.x;s_ind<GI.sector_dim; s_ind+= blockDim.x)
    {
      sdata[s_ind].x = 0.0f;//Re
      sdata[s_ind].y = 0.0f;//Im
    }
    __syncthreads();

    //start convolution
    int ind, x, y, z;
    int imin, imax, jmin, jmax, kmin, kmax;

    DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

    __shared__ IndType3 center;
    center.x = sector_centers[sec[threadIdx.x] * 3];
    center.y = sector_centers[sec[threadIdx.x] * 3 + 1];
    center.z = sector_centers[sec[threadIdx.x] * 3 + 2];

    //Grid Points over Threads
    int data_cnt = sectors[sec[threadIdx.x]] + threadIdx.x + sec_offset;
    //loop over all data points of the current sector, and check if grid position lies inside 
    //affected region, if so, add data point weighted to grid position value
    while (data_cnt < sec_max)
    {
      DType3 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt +GI.data_count];
      data_point.z = crds[data_cnt +2*GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
      kz = mapKSpaceToGrid(data_point.z,GI.gridDims.z,center.z,GI.sector_offset);
      set_minmax(&kz, &kmin, &kmax, GI.sector_pad_max, GI.kernel_radius);

      // grid this point onto its cartesian points neighbors
      for (int k = kmin; k <= kmax; k++)
      {
        kz = mapGridToKSpace(k,GI.gridDims.z,center.z,GI.sector_offset);
        dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
        dz_sqr *= dz_sqr;
        if (dz_sqr < GI.radiusSquared)
        {
          for (int j = jmin; j <= jmax; j++)
          {
            jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
            dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
            dy_sqr *= dy_sqr;
            if (dy_sqr < GI.radiusSquared)	
            {
              for (int i = imin; i <= imax; i++)
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
                  atomicAdd(&(sdata[ind].x),val * data[data_cnt].x);
                  atomicAdd(&(sdata[ind].y),val * data[data_cnt].y);
                } // kernel bounds check x, spherical support 
              } // x 	 
            } // kernel bounds check y, spherical support 
          } // y 
        } //kernel bounds check z 
      } // z
      data_cnt = data_cnt + blockDim.x;
    } //grid points per sector

    //write shared data to output grid
    __syncthreads();

    __shared__ int sector_ind_offset;
    sector_ind_offset  = computeXYZ2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.gridDims);

    //each thread writes one position from shared mem to global mem
    for (int s_ind=threadIdx.x;s_ind<GI.sector_dim; s_ind += blockDim.x)
    {
      getCoordsFromIndex(s_ind,&x,&y,&z,GI.sector_pad_width);

      if (isOutlier(x,y,z,center.x,center.y,center.z,GI.gridDims,GI.sector_offset))
        //calculate opposite index
        ind = computeXYZ2Lin(calculateOppositeIndex(x,center.x,GI.gridDims.x,GI.sector_offset),
        calculateOppositeIndex(y,center.y,GI.gridDims.y,GI.sector_offset),
        calculateOppositeIndex(z,center.z,GI.gridDims.z,GI.sector_offset),
        GI.gridDims);
      else
        ind = sector_ind_offset + computeXYZ2Lin(x,y,z,GI.gridDims);//index in output grid

      atomicAdd(&(gdata[ind].x),sdata[s_ind].x);//Re
      atomicAdd(&(gdata[ind].y),sdata[s_ind].y);//Im
    }
}
__global__ void convolutionKernel2(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N
  )
{
  extern __shared__ DType2 sdata[];//externally managed shared memory
  __shared__ int sec[THREAD_BLOCK_SIZE];
  sec[threadIdx.x] = blockIdx.x;
  while (sec[threadIdx.x] < N)
  {    
    __shared__ int data_max;
    data_max = sectors[sec[threadIdx.x]+1];
    convolutionFunction2(sec,data_max,0,sdata,data,crds,gdata,sectors,sector_centers);
    __syncthreads();
    sec[threadIdx.x] = sec[threadIdx.x]+ gridDim.x;
  }//sec < sector_count	
}

__global__ void balancedConvolutionKernel2(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType2* sector_processing_order,
  IndType* sector_centers,
  int N
  )
{
  extern __shared__ DType2 sdata[];//externally managed shared memory
  int sec_cnt = blockIdx.x;
  __shared__ int sec[THREAD_BLOCK_SIZE];
  
  while (sec_cnt < N)
  {
    sec[threadIdx.x] = sector_processing_order[sec_cnt].x;
    __shared__ int data_max;
    data_max = min(sectors[sec[threadIdx.x]+1],sectors[sec[threadIdx.x]] + sector_processing_order[sec_cnt].y+MAXIMUM_PAYLOAD);
    convolutionFunction2(sec,data_max,sector_processing_order[sec_cnt].y,sdata,data,crds,gdata,sectors,sector_centers);
    __syncthreads();
    sec_cnt = sec_cnt + gridDim.x;
  }//sec < sector_count	
}


// ----------------------------------------------------------------------------
// convolutionKernel: NUFFT^H kernel
//
// Performs the gpuNUFFT step by convolution of sample points with 
// interpolation function and resampling onto grid. Basic concept based on Zwart
// et al. 
//
// parameters:
//  * data           : complex input sample points
//  * crds           : coordinates of data points (x,y,z)
//  * gdata          : output grid data 
//  * sectors        : mapping of sample indices according to each sector
//  * sector_centers : coordinates (x,y,z) of sector centers
//  * temp_gdata     : temporary grid data
//  * N              : number of threads
__device__ void convolutionFunction2D(DType2* sdata,int* sec, int sec_max, int sec_offset, DType2* data, DType* crds, CufftType* gdata,IndType* sectors, IndType* sector_centers)
{
  //init shared memory
    for (int s_ind=threadIdx.x;s_ind<GI.sector_dim; s_ind+= blockDim.x)
    {
      for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
      {
        sdata[s_ind + c*GI.sector_dim].x = 0.0f;//Re
        sdata[s_ind + c*GI.sector_dim].y = 0.0f;//Im
      }
    }
    __syncthreads();

    //start convolution
    int ind, x, y;
    int imin, imax,jmin,jmax;

    DType dx_sqr, dy_sqr, val, ix, jy;

    __shared__ IndType2 center;
    center.x = sector_centers[sec[threadIdx.x] * 2];
    center.y = sector_centers[sec[threadIdx.x] * 2 + 1];

    //Grid Points over Threads
    int data_cnt = sectors[sec[threadIdx.x]] + threadIdx.x + sec_offset;

    //loop over all data points of the current sector, and check if grid position lies inside 
    //affected region, if so, add data point weighted to grid position value
    while (data_cnt < sec_max)
    {
      DType2 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt +GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);

      // grid this point onto its cartesian points neighbors
      for(int  j = jmin; j <= jmax; j++)
      {
        jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
        dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
        dy_sqr *= dy_sqr;
        if (dy_sqr < GI.radiusSquared)	
        {
          for(int i = imin; i <= imax; i++)
          {
            ix = mapGridToKSpace(i,GI.gridDims.x,center.x,GI.sector_offset);
            dx_sqr = (ix - data_point.x)*GI.aniso_x_scale;
            dx_sqr *= dx_sqr;
            if (dx_sqr < GI.radiusSquared)	
            {
              //get kernel value
              //Calculate Separable Filters 
              val = KERNEL[(int) round(dy_sqr * GI.dist_multiplier)] *
                KERNEL[(int) round(dx_sqr * GI.dist_multiplier)];
              ind = getIndex2D(i,j,GI.sector_pad_width);

              // multiply data by current kernel val 
              // grid complex or scalar 
              for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
              {
                atomicAdd(&(sdata[ind + c * GI.sector_dim].x),val * data[data_cnt + c * GI.data_count].x);
                atomicAdd(&(sdata[ind + c * GI.sector_dim].y),val * data[data_cnt + c * GI.data_count].y);
              }
            } // kernel bounds check x, spherical support 
          } // x 	 
        } // kernel bounds check y, spherical support 
      } // y 
      data_cnt = data_cnt + blockDim.x;
    } //grid points per sector

    //write shared data to output grid
    __syncthreads();
    //int sector_ind_offset = sec * GI.sector_dim;
    __shared__ int sector_ind_offset;
    sector_ind_offset  = computeXY2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,GI.gridDims);

    //each thread writes one position from shared mem to global mem
    for (int s_ind=threadIdx.x;s_ind<GI.sector_dim; s_ind += blockDim.x)
    {
      getCoordsFromIndex2D(s_ind,&x,&y,GI.sector_pad_width);

      if (isOutlier2D(x,y,center.x,center.y,GI.gridDims,GI.sector_offset))
        //calculate opposite index
        ind = computeXY2Lin(calculateOppositeIndex(x,center.x,GI.gridDims.x,GI.sector_offset),
        calculateOppositeIndex(y,center.y,GI.gridDims.y,GI.sector_offset),
        GI.gridDims);
      else
        ind = sector_ind_offset + computeXY2Lin(x,y,GI.gridDims);//index in output grid

      for (int c = threadIdx.z; c < GI.n_coils_cc; c+= blockDim.z)
      {
        atomicAdd(&(gdata[ind + c * GI.gridDims_count].x),sdata[s_ind + c * GI.sector_dim].x);//Re
        atomicAdd(&(gdata[ind + c * GI.gridDims_count].y),sdata[s_ind + c * GI.sector_dim].y);//Im
        sdata[s_ind + c * GI.sector_dim].x = 0.0f;//Re
        sdata[s_ind + c * GI.sector_dim].y = 0.0f;//Im
      }
    }
    __syncthreads();
}

__global__ void convolutionKernel2D(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N
  )
{
  extern __shared__ DType2 sdata[];//externally managed shared memory
  __shared__ int sec[THREAD_BLOCK_SIZE];
  sec[threadIdx.x] = blockIdx.x;
  while (sec[threadIdx.x] < N)
  {
    __shared__ int data_max;
    data_max = sectors[sec[threadIdx.x]+1];
    convolutionFunction2D(sdata,sec,data_max,0,data,crds,gdata,sectors,sector_centers);
    __syncthreads();
    sec[threadIdx.x] = sec[threadIdx.x]+ gridDim.x;
  }//sec < sector_count	
}

__global__ void balancedConvolutionKernel2D(DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType2* sector_processing_order,
  IndType* sector_centers,
  int N
  )
{
  extern __shared__ DType2 sdata[];//externally managed shared memory
  int sec_cnt = blockIdx.x;
  __shared__ int sec[THREAD_BLOCK_SIZE];
  
  while (sec_cnt < N)
  {
    sec[threadIdx.x] = sector_processing_order[sec_cnt].x; 
    __shared__ int data_max;
    data_max = min(sectors[sec[threadIdx.x]+1],sectors[sec[threadIdx.x]] + sector_processing_order[sec_cnt].y + MAXIMUM_PAYLOAD);
    convolutionFunction2D(sdata,sec,data_max,sector_processing_order[sec_cnt].y,data,crds,gdata,sectors,sector_centers);
    __syncthreads();
    sec_cnt = sec_cnt+ gridDim.x;
  }//sec < sector_count	
}

// ----------------------------------------------------------------------------
// convolutionKernel: NUFFT^H kernel
//
// Performs the gpuNUFFT step by convolution of sample points with 
// interpolation function and resampling onto grid. Basic concept based on Zwart
// et al. 
//
// parameters:
//  * data           : complex input sample points
//  * crds           : coordinates of data points (x,y,z)
//  * gdata          : output grid data 
//  * sectors        : mapping of sample indices according to each sector
//  * sector_centers : coordinates (x,y,z) of sector centers
//  * temp_gdata     : temporary grid data
//  * N              : number of threads
__global__ void convolutionKernel( DType2* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N
  )
{
  int  sec= blockIdx.x;
  //start convolution
  while (sec < N)
  {
    int ind, imin, imax, jmin, jmax, kmin, kmax;

    DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

    __shared__ IndType3 center;
    center.x = sector_centers[sec * 3];
    center.y = sector_centers[sec * 3 + 1];
    center.z = sector_centers[sec * 3 + 2];

    //Grid Points over Threads
    int data_cnt = sectors[sec] + threadIdx.x;
    int data_max = sectors[sec+1];
    __shared__ int sector_ind_offset;
    sector_ind_offset = computeXYZ2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.gridDims);

    while (data_cnt < data_max)
    {
      DType3 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt +GI.data_count];
      data_point.z = crds[data_cnt +2*GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
      kz = mapKSpaceToGrid(data_point.z,GI.gridDims.z,center.z,GI.sector_offset);
      set_minmax(&kz, &kmin, &kmax, GI.sector_pad_max, GI.kernel_radius);

      // convolve neighboring cartesian points to this data point
      for (int k = kmin; k <= kmax; k++)
      {
        kz = mapGridToKSpace(k,GI.gridDims.z,center.z,GI.sector_offset);
        dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
        dz_sqr *= dz_sqr;

        if (dz_sqr < GI.radiusSquared)
        {
          for (int j = jmin; j <= jmax; j++)
          {
            jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
            dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
            dy_sqr *= dy_sqr;
            if (dy_sqr < GI.radiusSquared)	
            {
              for (int i = imin; i <= imax; i++)
              {
                ix = mapGridToKSpace(i,GI.gridDims.x,center.x,GI.sector_offset);
                dx_sqr = (ix - data_point.x)*GI.aniso_x_scale;
                dx_sqr *= dx_sqr;
                if (dx_sqr < GI.radiusSquared)	
                {
                  // get kernel value
                  //Berechnung mit Separable Filters 
                  val = KERNEL[(int) round(dz_sqr * GI.dist_multiplier)] *
                    KERNEL[(int) round(dy_sqr * GI.dist_multiplier)] *
                    KERNEL[(int) round(dx_sqr * GI.dist_multiplier)];

                  if (isOutlier(i,j,k,center.x,center.y,center.z,GI.gridDims,GI.sector_offset))
                    //calculate opposite index
                    ind = computeXYZ2Lin(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
                    calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
                    calculateOppositeIndex(k,center.z,GI.gridDims.z,GI.sector_offset),
                    GI.gridDims);
                  else
                    ind = sector_ind_offset + computeXYZ2Lin(i,j,k,GI.gridDims);//index in output grid

                  atomicAdd(&(gdata[ind].x),val * data[data_cnt].x);//Re
                  atomicAdd(&(gdata[ind].y),val * data[data_cnt].y);//Im
                }// kernel bounds check x, spherical support 
              } // x loop
            } // kernel bounds check y, spherical support  
          } // y loop
        } //kernel bounds check z 
      } // z loop
      data_cnt = data_cnt + blockDim.x;
    } //data points per sector
    __syncthreads();	
    sec = sec + gridDim.x;
  } //sector check
}

void performConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType*			kernel_d, 
  IndType* sectors_d, 
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  )
{
#define CONVKERNEL2

#ifdef CONVKERNEL 	
  long shared_mem_size = (gi_host->sector_dim)*sizeof(DType2);
  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,THREAD_BLOCK_SIZE));
  if (gi_host->is2Dprocessing)
    convolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
  else
    convolutionKernel<<<grid_dim,block_dim>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
#else
#ifdef CONVKERNEL2
  long shared_mem_size = (gi_host->sector_dim) * sizeof(DType2) * gi_host->n_coils_cc;
  int thread_size = THREAD_BLOCK_SIZE;

  dim3 block_dim(thread_size);
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,1));
  if (DEBUG)
  {
    printf("adjoint convolution requires %ld bytes of shared memory!\n",shared_mem_size);
    printf("grid dim %u, block dim %u \n",grid_dim.x, block_dim.x); 
  }
  if (gi_host->is2Dprocessing)
  {
    dim3 block_dim(64, 1, DEFAULT_VALUE(gi_host->n_coils_cc > 4 ? 4 : gi_host->n_coils_cc));
    convolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
  }
  else
  {
    convolutionKernel2<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
  }
#else 
#ifdef CONVKERNEL4
  // TODO tune param z dim
  // defines size of total shared mem used
  dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,3);
  long shared_mem_size = block_dim.x*block_dim.y*block_dim.z*sizeof(DType2);
  
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,1));
  
  if (DEBUG)
  {
    printf("adjoint convolution requires %ld bytes of shared memory!\n",shared_mem_size);
    printf("grid dim (%u,%u,%u), block dim (%u,%u,%u) \n",grid_dim.x,grid_dim.y,grid_dim.z, block_dim.x,block_dim.y,block_dim.z); 
  }
  if (gi_host->is2Dprocessing)
  {
    shared_mem_size = (gi_host->sector_dim)*sizeof(DType2);
    int thread_size =THREAD_BLOCK_SIZE;

    dim3 block_dim(thread_size);
    dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,1));

    convolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
  }
  else
    convolutionKernel4<<<grid_dim,block_dim>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
#else
  long cache_size = 176;
  long shared_mem_size = (2*cache_size + 3*cache_size)*sizeof(DType);
  dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,4);
  dim3 grid_dim(gi_host->sector_count);
  convolutionKernel3<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count,cache_size);
#endif
#endif
#endif
  if (DEBUG)
    printf("...finished with: %s\n", cudaGetErrorString(cudaGetLastError()));
}

void performConvolution( DType2* data_d, 
  DType* crds_d, 
  CufftType* gdata_d,
  DType*			kernel_d, 
  IndType* sectors_d, 
  IndType2* sector_processing_order_d,
  IndType* sector_centers_d,
  gpuNUFFT::GpuNUFFTInfo* gi_host
  )
{
  long shared_mem_size = (gi_host->sector_dim)*sizeof(DType2) * gi_host->n_coils_cc;
  int thread_size =THREAD_BLOCK_SIZE;

  dim3 block_dim(thread_size);
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,1));
  if (DEBUG)
  {
    printf("adjoint convolution requires %ld bytes of shared memory!\n",shared_mem_size);
    printf("grid dim %u, block dim %u \n",grid_dim.x, block_dim.x); 
  }
  if (gi_host->is2Dprocessing)
  {
    dim3 block_dim(64, 1, DEFAULT_VALUE(gi_host->n_coils_cc > 4 ? 4 : gi_host->n_coils_cc));
    balancedConvolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_processing_order_d,sector_centers_d,gi_host->sectorsToProcess);
  }
  else
  {
    balancedConvolutionKernel2<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_processing_order_d,sector_centers_d,gi_host->sectorsToProcess);
    //dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,3);
    //dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,block_dim.x*block_dim.y*block_dim.z));
    //balancedConvolutionKernel4<<<grid_dim,block_dim>>>(data_d,crds_d,gdata_d,sectors_d,sector_processing_order_d,sector_centers_d,gi_host->sectorsToProcess);
  }
  if (DEBUG)
    printf("...finished with: %s\n", cudaGetErrorString(cudaGetLastError()));
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
__global__ void forwardConvolutionKernel( CufftType* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared_out_data[];//externally managed shared memory
  __shared__ int sec[THREAD_BLOCK_SIZE];
  sec[threadIdx.x]= blockIdx.x;
  //init shared memory
  shared_out_data[threadIdx.x].x = 0.0f;//Re
  shared_out_data[threadIdx.x].y = 0.0f;//Im
  __syncthreads();
  //start convolution
  while (sec[threadIdx.x] < N)
  {
    int ind, imin, imax, jmin, jmax, kmin, kmax;
    DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

    __shared__ IndType3 center;
    center.x = sector_centers[sec[threadIdx.x] * 3];
    center.y = sector_centers[sec[threadIdx.x] * 3 + 1];
    center.z = sector_centers[sec[threadIdx.x] * 3 + 2];

    //Grid Points over Threads
    int data_cnt = sectors[sec[threadIdx.x]] + threadIdx.x;
    __shared__ int data_max;
    data_max = sectors[sec[threadIdx.x]+1];	
    __shared__ int sector_ind_offset; 
    sector_ind_offset = computeXYZ2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.gridDims);

    while (data_cnt < data_max)
    {
      DType3 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt +GI.data_count];
      data_point.z = crds[data_cnt +2*GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
      kz = mapKSpaceToGrid(data_point.z,GI.gridDims.z,center.z,GI.sector_offset);
      set_minmax(&kz, &kmin, &kmax, GI.sector_pad_max, GI.kernel_radius);

      // convolve neighboring cartesian points to this data point
      for (int k = kmin; k <= kmax; k++)
      {
        kz = mapGridToKSpace(k,GI.gridDims.z,center.z,GI.sector_offset);
        dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
        dz_sqr *= dz_sqr;

        if (dz_sqr < GI.radiusSquared)
        {
          for (int j = jmin; j <= jmax; j++)
          {
            jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
            dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
            dy_sqr *= dy_sqr;
            if (dy_sqr < GI.radiusSquared)	
            {
              for (int i = imin; i <= imax; i++)
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
              } // x loop
            } // kernel bounds check y, spherical support  
          } // y loop
        } //kernel bounds check z 
      } // z loop
      data[data_cnt].x = shared_out_data[threadIdx.x].x;
      data[data_cnt].y = shared_out_data[threadIdx.x].y;

      data_cnt = data_cnt + blockDim.x;

      shared_out_data[threadIdx.x].x = (DType)0.0;//Re
      shared_out_data[threadIdx.x].y = (DType)0.0;//Im
    } //data points per sector
    __syncthreads();
    sec[threadIdx.x]= sec[threadIdx.x] + gridDim.x;
  } //sector check
}


__device__ void forwardConvolutionFunction2(int* sec, int sec_max, int sec_offset, DType2* sdata, CufftType* gdata_cache, DType2* data, DType* crds, CufftType* gdata, IndType* sectors, IndType* sector_centers)
{
  int ind, imin, imax, jmin, jmax, kmin, kmax, ii, jj, kk;
  DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

  __shared__ IndType3 center;
  center.x = sector_centers[sec[threadIdx.x] * 3];
  center.y = sector_centers[sec[threadIdx.x] * 3 + 1];
  center.z = sector_centers[sec[threadIdx.x] * 3 + 2];

  __shared__ int sector_ind_offset;
  sector_ind_offset = computeXYZ2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.gridDims);

  // init sector cache 
  // preload sector grid data into cache
  for (int ind=threadIdx.x; ind<GI.sector_dim; ind+=blockDim.x)
  {
    int grid_index;
    getCoordsFromIndex(ind,&ii,&jj,&kk,GI.sector_pad_width);

    if (isOutlier(ii,jj,kk,center.x,center.y,center.z,GI.gridDims,GI.sector_offset))
      //calculate opposite index
      grid_index = computeXYZ2Lin(calculateOppositeIndex(ii,center.x,GI.gridDims.x,GI.sector_offset),
      calculateOppositeIndex(jj,center.y,GI.gridDims.y,GI.sector_offset),
      calculateOppositeIndex(kk,center.z,GI.gridDims.z,GI.sector_offset),
      GI.gridDims);
    else
      grid_index = (sector_ind_offset + computeXYZ2Lin(ii,jj,kk,GI.gridDims));

    gdata_cache[ind].x = gdata[grid_index].x;
    gdata_cache[ind].y = gdata[grid_index].y;
  }
    
  __syncthreads();

  //Grid Points over Threads
  int data_cnt = sectors[sec[threadIdx.x]] + threadIdx.x + sec_offset;
   
  while (data_cnt < sec_max)
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
    for (int k = kmin; k <= kmax; k++)
    {
      kz = mapGridToKSpace(k,GI.gridDims.z,center.z,GI.sector_offset);
      dz_sqr = (kz - data_point.z)*GI.aniso_z_scale;
      dz_sqr *= dz_sqr;

      if (dz_sqr < GI.radiusSquared)
      {
        for (int j = jmin; j <= jmax; j++)
        {
          jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
          dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
          dy_sqr *= dy_sqr;
          if (dy_sqr < GI.radiusSquared)	
          {
            for (int i = imin; i <= imax; i++)
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

                sdata[threadIdx.x].x += gdata_cache[ind].x * val; 
                sdata[threadIdx.x].y += gdata_cache[ind].y * val;									
              }// kernel bounds check x, spherical support 
            } // x loop
          } // kernel bounds check y, spherical support  
        } // y loop
      } //kernel bounds check z 
    } // z loop
    atomicAdd(&(data[data_cnt].x),sdata[threadIdx.x].x);
    atomicAdd(&(data[data_cnt].y),sdata[threadIdx.x].y);

    data_cnt = data_cnt + blockDim.x;

    sdata[threadIdx.x].x = (DType)0.0;//Re
    sdata[threadIdx.x].y = (DType)0.0;//Im
  } //data points per sector
}

// cached version of above kernel
__global__ void forwardConvolutionKernel2(CufftType* data, 
  DType*     crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared[];//externally managed shared memory
  CufftType* shared_out_data =(CufftType*) &shared[0];  
  CufftType* gdata_cache =(CufftType*) &shared[blockDim.x]; 

  __shared__ int sec[THREAD_BLOCK_SIZE];
  sec[threadIdx.x]= blockIdx.x;
  
  //init shared memory
  shared_out_data[threadIdx.x].x = 0.0f;//Re
  shared_out_data[threadIdx.x].y = 0.0f;//Im

  __syncthreads();
  //start convolution
  while (sec[threadIdx.x] < N)
  {
    __shared__ int data_max;
    data_max = sectors[sec[threadIdx.x]+1];	

    forwardConvolutionFunction2(sec,data_max,0,shared_out_data,gdata_cache,data,crds,gdata,sectors,sector_centers);
    __syncthreads();
    sec[threadIdx.x]= sec[threadIdx.x] + gridDim.x;
  } //sector check
}

__global__ void balancedForwardConvolutionKernel2(CufftType* data, 
  DType*     crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType2* sector_processing_order,
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared[];//externally managed shared memory
  CufftType* shared_out_data =(CufftType*) &shared[0];  
  CufftType* gdata_cache =(CufftType*) &shared[blockDim.x]; 
  
  int sec_cnt = blockIdx.x;
  __shared__ int sec[THREAD_BLOCK_SIZE];

  //init shared memory
  shared_out_data[threadIdx.x].x = 0.0f;//Re
  shared_out_data[threadIdx.x].y = 0.0f;//Im

  __syncthreads();
  //start convolution
  while (sec_cnt < N)
  {
    sec[threadIdx.x] = sector_processing_order[sec_cnt].x;
    __shared__ int data_max;
    data_max = min(sectors[sec[threadIdx.x]+1],sectors[sec[threadIdx.x]] + sector_processing_order[sec_cnt].y + MAXIMUM_PAYLOAD);

    forwardConvolutionFunction2(sec,data_max,sector_processing_order[sec_cnt].y,shared_out_data,gdata_cache,data,crds,gdata,sectors,sector_centers);
    __syncthreads();
    sec_cnt = sec_cnt + gridDim.x;
  } //sector check
}

__global__ void forwardConvolutionKernel2D( CufftType* data, 
  DType* crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared_out_data[];//externally managed shared memory
  __shared__ int sec[THREAD_BLOCK_SIZE];
  sec[threadIdx.x]= blockIdx.x;
  //init shared memory
  for (int c=threadIdx.z; c < GI.n_coils_cc; c+=blockDim.z)
  { 
    shared_out_data[threadIdx.x + c * blockDim.x].x = 0.0f;//Re
    shared_out_data[threadIdx.x + c * blockDim.x].y = 0.0f;//Im
  }
  __syncthreads();
  //start convolution
  while (sec[threadIdx.x] < N)
  {
    int ind, imin, imax, jmin, jmax;
    DType dx_sqr, dy_sqr, val, ix, jy;

    __shared__ IndType2 center;
    center.x = sector_centers[sec[threadIdx.x] * 2];
    center.y = sector_centers[sec[threadIdx.x] * 2 + 1];

    //Grid Points over Threads
    int data_cnt = sectors[sec[threadIdx.x]] + threadIdx.x;
    __shared__ int data_max;
    data_max = sectors[sec[threadIdx.x]+1];	
    __shared__ int sector_ind_offset; 
    sector_ind_offset = computeXY2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,GI.gridDims);

    while (data_cnt < data_max)
    {
      DType2 data_point; //datapoint per thread
      data_point.x = crds[data_cnt];
      data_point.y = crds[data_cnt +GI.data_count];

      // set the boundaries of final dataset for gpuNUFFT this point
      ix = mapKSpaceToGrid(data_point.x,GI.gridDims.x,center.x,GI.sector_offset);
      set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
      jy = mapKSpaceToGrid(data_point.y,GI.gridDims.y,center.y,GI.sector_offset);
      set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);

      // convolve neighboring cartesian points to this data point
      for (int j = jmin; j <= jmax; j++)
      {
        jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
        dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
        dy_sqr *= dy_sqr;
        if (dy_sqr < GI.radiusSquared)	
        {			
          for (int i = imin; i <= imax; i++)
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
                ind = computeXY2Lin(calculateOppositeIndex(i,center.x,GI.gridDims.x,GI.sector_offset),
                calculateOppositeIndex(j,center.y,GI.gridDims.y,GI.sector_offset),
                GI.gridDims);
              else
                ind = (sector_ind_offset + computeXY2Lin(i,j,GI.gridDims));

              for (int c=threadIdx.z; c < GI.n_coils_cc; c+=blockDim.z)
              { 
                shared_out_data[threadIdx.x + c * blockDim.x].x += gdata[ind+ c*GI.gridDims_count].x * val; 
                shared_out_data[threadIdx.x + c * blockDim.x].y += gdata[ind+ c*GI.gridDims_count].y * val;
              }
            }// kernel bounds check x, spherical support 
          } // x loop
        } // kernel bounds check y, spherical support  
      } // y loop

      for (int c=threadIdx.z; c < GI.n_coils_cc; c+=blockDim.z)
      { 
        data[data_cnt + c*GI.data_count].x = shared_out_data[threadIdx.x + c * blockDim.x].x;
        data[data_cnt + c*GI.data_count].y = shared_out_data[threadIdx.x + c * blockDim.x].y;

        shared_out_data[threadIdx.x + c*blockDim.x].x = (DType)0.0;//Re
        shared_out_data[threadIdx.x + c*blockDim.x].y = (DType)0.0;//Im
      }
        data_cnt = data_cnt + blockDim.x;
    } //data points per sector
    __syncthreads();
    sec[threadIdx.x]= sec[threadIdx.x] + gridDim.x;
  } //sector check
}

__device__ void forwardConvolutionFunction2D(int* sec, int sec_max, int sec_offset, DType2* sdata, CufftType* gdata_cache, DType2* data, DType* crds, CufftType* gdata, IndType* sectors, IndType* sector_centers)
{
  int ind, imin, imax, jmin, jmax, ii, jj;
  DType dx_sqr, dy_sqr, val, ix, jy;

  __shared__ IndType2 center;
  center.x = sector_centers[sec[threadIdx.x] * 2];
  center.y = sector_centers[sec[threadIdx.x] * 2 + 1];

  __shared__ int sector_ind_offset;
  sector_ind_offset = computeXY2Lin(center.x - GI.sector_offset,center.y - GI.sector_offset,GI.gridDims);

    // init sector cache 
  // preload sector grid data into cache
  for (int ind=threadIdx.x; ind<GI.sector_dim; ind+=blockDim.x)
  {
    int grid_index;
    getCoordsFromIndex2D(ind,&ii,&jj,GI.sector_pad_width);

    // multiply data by current kernel val 
    // grid complex or scalar 
    if (isOutlier2D(ii,jj,center.x,center.y,GI.gridDims.x,GI.sector_offset))
      //calculate opposite index
      grid_index = getIndex2D(calculateOppositeIndex(ii,center.x,GI.gridDims.x,GI.sector_offset),
      calculateOppositeIndex(jj,center.y,GI.gridDims.y,GI.sector_offset),
      GI.gridDims.x);
    else
      grid_index = (sector_ind_offset + getIndex2D(ii,jj,GI.gridDims.x));

    for (int c=threadIdx.z; c < GI.n_coils_cc; c+=blockDim.z)
    {
      gdata_cache[ind + c*GI.sector_dim].x = gdata[grid_index + c*GI.gridDims_count].x;
      gdata_cache[ind + c*GI.sector_dim].y = gdata[grid_index + c*GI.gridDims_count].y;
    }
  }
  __syncthreads();

  //Grid Points over Threads
  int data_cnt = sectors[sec[threadIdx.x]] + threadIdx.x + sec_offset;
    
  while (data_cnt < sec_max)
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
    for (int j = jmin; j <= jmax; j++)
    {
      jy = mapGridToKSpace(j,GI.gridDims.y,center.y,GI.sector_offset);
      dy_sqr = (jy - data_point.y) * GI.aniso_y_scale;
      dy_sqr *= dy_sqr;
      if (dy_sqr < GI.radiusSquared)	
      {				
        for (int i = imin; i <= imax; i++)
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

            for (int c=threadIdx.z; c < GI.n_coils_cc; c+=blockDim.z)
            {
              sdata[threadIdx.x + c*blockDim.x].x += gdata_cache[ind + c*GI.sector_dim].x * val; 
              sdata[threadIdx.x + c*blockDim.x].y += gdata_cache[ind + c*GI.sector_dim].y * val;
            }
          }// kernel bounds check x, spherical support 
        } // x loop
      } // kernel bounds check y, spherical support  
    } // y loop

    for (int c=threadIdx.z; c < GI.n_coils_cc; c+=blockDim.z)
    {
      atomicAdd(&(data[data_cnt + c*GI.data_count].x),sdata[threadIdx.x + c*blockDim.x].x);
      atomicAdd(&(data[data_cnt + c*GI.data_count].y),sdata[threadIdx.x + c*blockDim.x].y);
      sdata[threadIdx.x + c*blockDim.x].x = (DType)0.0;//Re
      sdata[threadIdx.x + c*blockDim.x].y = (DType)0.0;//Im
    }

    data_cnt = data_cnt + blockDim.x;
  } //data points per sector
}

//cached version of above kernel
__global__ void forwardConvolutionKernel22D(CufftType* data, 
  DType*     crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared[];//externally managed shared memory
  CufftType* shared_out_data =(CufftType*) &shared[0];  
  CufftType* gdata_cache =(CufftType*) &shared[blockDim.x * GI.n_coils_cc]; 

  __shared__ int sec[THREAD_BLOCK_SIZE];
  sec[threadIdx.x]= blockIdx.x;

  //init shared memory
  for (int c=threadIdx.z; c < GI.n_coils_cc; c+=blockDim.z)
  { 
    shared_out_data[threadIdx.x + c*blockDim.x].x = 0.0f;//Re
    shared_out_data[threadIdx.x + c*blockDim.x].y = 0.0f;//Im
  }
  __syncthreads();
  //start convolution
  while (sec[threadIdx.x] < N)
  {
    __shared__ int data_max;
    data_max = sectors[sec[threadIdx.x]+1];

    forwardConvolutionFunction2D(sec,data_max,0,shared_out_data,gdata_cache,data,crds,gdata,sectors,sector_centers);

    __syncthreads();
    sec[threadIdx.x]= sec[threadIdx.x] + gridDim.x;
  } //sector check
}

__global__ void balancedForwardConvolutionKernel22D(CufftType* data, 
  DType*     crds, 
  CufftType* gdata,
  IndType* sectors, 
  IndType2* sector_processing_order,
  IndType* sector_centers,
  int N)
{
  extern __shared__ CufftType shared[];//externally managed shared memory
  CufftType* shared_out_data =(CufftType*) &shared[0];  
  CufftType* gdata_cache =(CufftType*) &shared[blockDim.x * GI.n_coils_cc];

  int sec_cnt= blockIdx.x;
  __shared__ int sec[THREAD_BLOCK_SIZE];
  
  //init shared memory
  for (int c=threadIdx.z; c < GI.n_coils_cc; c+=blockDim.z)
  {
    shared_out_data[threadIdx.x + c * blockDim.x].x = 0.0f;//Re
    shared_out_data[threadIdx.x + c * blockDim.x].y = 0.0f;//Im
  }

  __syncthreads();
  //start convolution
  while (sec_cnt < N)
  {
    sec[threadIdx.x] = sector_processing_order[sec_cnt].x;
    __shared__ int data_max;
    data_max = min(sectors[sec[threadIdx.x]+1],sectors[sec[threadIdx.x]] + sector_processing_order[sec_cnt].y+MAXIMUM_PAYLOAD);

    forwardConvolutionFunction2D(sec,data_max,sector_processing_order[sec_cnt].y,shared_out_data,gdata_cache,data,crds,gdata,sectors,sector_centers);

    __syncthreads();
    sec_cnt = sec_cnt + gridDim.x;
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
  // cached version proved to be
  // faster than non-cached version 
  // even in 2d case
  bool useCache = true;
  if (!useCache)
  {
    int thread_size =THREAD_BLOCK_SIZE;
    long shared_mem_size = thread_size * sizeof(CufftType) * gi_host->n_coils_cc;//empiric

    dim3 block_dim(thread_size);
    dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,thread_size));

    if (DEBUG)
      printf("convolution requires %ld bytes of shared memory!\n",shared_mem_size);
    if (gi_host->is2Dprocessing)
    {
      dim3 block_dim(thread_size, 1, DEFAULT_VALUE(gi_host->n_coils_cc > 8 ? 8 : gi_host->n_coils_cc));
      forwardConvolutionKernel2D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
    }
    else
      forwardConvolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
  }
  else
  {
    int thread_size = THREAD_BLOCK_SIZE;
    long shared_mem_size = (thread_size + gi_host->sector_dim) * gi_host->n_coils_cc * sizeof(CufftType);//empiric

    dim3 block_dim(thread_size);
    dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,thread_size));

    if (DEBUG)
      printf("forward convolution requires %ld bytes of shared memory!\n",shared_mem_size);
    if (gi_host->is2Dprocessing)
    {
      dim3 block_dim(thread_size, 1, DEFAULT_VALUE(gi_host->n_coils_cc > 4 ? 2 : gi_host->n_coils_cc));
      forwardConvolutionKernel22D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_centers_d,gi_host->sector_count);
    }
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
  int thread_size =THREAD_BLOCK_SIZE;//empiric
  long shared_mem_size = (thread_size + gi_host->sector_dim ) * gi_host->n_coils_cc * sizeof(CufftType);

  dim3 block_dim(thread_size);
  dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,thread_size));

  if (DEBUG)
    printf("balanced convolution requires %ld bytes of shared memory!\n",shared_mem_size);
  if (gi_host->is2Dprocessing)
    {
      dim3 block_dim(160, 1, DEFAULT_VALUE(gi_host->n_coils_cc > 4 ? 2 : gi_host->n_coils_cc));
      balancedForwardConvolutionKernel22D<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_processing_order_d,sector_centers_d,gi_host->sectorsToProcess);
    }
  else
    balancedForwardConvolutionKernel2<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,sectors_d,sector_processing_order_d,sector_centers_d,gi_host->sectorsToProcess);
}

#endif
