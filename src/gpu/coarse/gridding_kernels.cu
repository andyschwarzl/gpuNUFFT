#include "gridding_kernels.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"

// convolve every data point on grid position -> controlled by threadIdx.x .y and .z 
// shared data holds grid values as software managed cache
//
// 
__global__ void convolutionKernel( DType* data, 
							    DType* crds, 
							    CufftType* gdata,
							    DType* kernel, 
							    int* sectors, 
								int* sector_centers,
								DType* temp_gdata,
								int N
								)
{
	extern __shared__ DType sdata[];//externally managed shared memory

	__shared__ int sec;
	sec = blockIdx.x;
	//init shared memory
	for (int z=threadIdx.z;z<GI.sector_pad_width; z += blockDim.z)
	{
		int y=threadIdx.y;
		int x=threadIdx.x;
		int s_ind = 2* getIndex(x,y,z,GI.sector_pad_width) ;
		sdata[s_ind] = 0.0f;//Re
		sdata[s_ind+1]=0.0f;//Im
	}
	__syncthreads();
	//start convolution
	while (sec < N)
	{
		int ind, k, i, j;
		__shared__ int max_dim, imin, imax,jmin,jmax,kmin,kmax;

		DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

		__shared__ int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];

		//Grid Points over threads
		int data_cnt;
		data_cnt = sectors[sec];
		
		max_dim =  GI.sector_pad_max;		
		while (data_cnt < sectors[sec+1])
		{
			__shared__ DType3 data_point; //datapoint shared in every thread
			data_point.x = crds[3*data_cnt];
			data_point.y = crds[3*data_cnt +1];
			data_point.z = crds[3*data_cnt +2];
			// set the boundaries of final dataset for gridding this point
			ix = (data_point.x + 0.5f) * (GI.grid_width) - center.x + GI.sector_offset;
			set_minmax(&ix, &imin, &imax, max_dim, GI.kernel_radius);
			jy = (data_point.y + 0.5f) * (GI.grid_width) - center.y + GI.sector_offset;
			set_minmax(&jy, &jmin, &jmax, max_dim, GI.kernel_radius);
			kz = (data_point.z + 0.5f) * (GI.grid_width) - center.z + GI.sector_offset;
			set_minmax(&kz, &kmin, &kmax, max_dim, GI.kernel_radius);
				                
			// grid this point onto the neighboring cartesian points
			for (k=threadIdx.z;k<=kmax; k += blockDim.z)
			{
				if (k<=kmax && k>=kmin)
				{
					kz = static_cast<DType>((k + center.z - GI.sector_offset)) / static_cast<DType>((GI.grid_width)) - 0.5f;//(k - center_z) *width_inv;
					dz_sqr = kz - data_point.z;
					dz_sqr *= dz_sqr;
					if (dz_sqr < GI.radiusSquared)
					{
						j=threadIdx.y;
						if (j<=jmax && j>=jmin)
						{
							jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.grid_width)) - 0.5f;   //(j - center_y) *width_inv;
							dy_sqr = jy - data_point.y;
							dy_sqr *= dy_sqr;
							if (dy_sqr < GI.radiusSquared)	
							{
								i=threadIdx.x;
								
								if (i<=imax && i>=imin)
								{
									ix = static_cast<DType>(i + center.x - GI.sector_offset) / static_cast<DType>((GI.grid_width)) - 0.5f;// (i - center_x) *width_inv;
									dx_sqr = ix - data_point.x;
									dx_sqr *= dx_sqr;
									if (dx_sqr < GI.radiusSquared)	
									{
										//get kernel value
										//Calculate Separable Filters 
										val = kernel[(int) round(dz_sqr * GI.dist_multiplier)] *
											  kernel[(int) round(dy_sqr * GI.dist_multiplier)] *
											  kernel[(int) round(dx_sqr * GI.dist_multiplier)];
										ind = 2* getIndex(i,j,k,GI.sector_pad_width);
								
										// multiply data by current kernel val 
										// grid complex or scalar 
										sdata[ind]   += val * data[2*data_cnt];
										sdata[ind+1] += val * data[2*data_cnt+1];
									} // kernel bounds check x, spherical support 
								} // x 	 
							} // kernel bounds check y, spherical support 
						} // y 
					} //kernel bounds check z 
				} // z
			}//for loop over z entries
			
			data_cnt++;
		} //grid points per sector
	
	    //write shared data to temporary output grid
		int sector_ind_offset = sec * GI.sector_dim;
		
		for (k=threadIdx.z;k<GI.sector_pad_width; k += blockDim.z)
		{
			i=threadIdx.x;
			j=threadIdx.y;
			
			int s_ind = 2* getIndex(i,j,k,GI.sector_pad_width) ;//index in shared grid
			ind = 2*sector_ind_offset + s_ind;//index in temp output grid
			
			temp_gdata[ind] = sdata[s_ind];//Re
			temp_gdata[ind+1] = sdata[s_ind+1];//Im
			__syncthreads();
			sdata[s_ind] = (DType)0.0;
			sdata[s_ind+1] = (DType)0.0;
		}
		__syncthreads();
		sec = sec + gridDim.x;
	}//sec < sector_count
}

__global__ void composeOutputKernel(DType* temp_gdata, CufftType* gdata, int* sector_centers)
{
	for (int sec = 0; sec < GI.sector_count; sec++)
	{
		__syncthreads();
		__shared__ int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];
		__shared__ int sector_ind_offset;
		sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.grid_width);
		__shared__ int sector_grid_offset;
		sector_grid_offset = sec * GI.sector_dim;
		//write data from temp grid to overall output grid
		for (int z=threadIdx.z;z<GI.sector_pad_width; z += blockDim.z)
		{
			int x=threadIdx.x;
			int y=threadIdx.y;
			int s_ind = 2* (sector_grid_offset + getIndex(x,y,z,GI.sector_pad_width));
			int ind = (sector_ind_offset + getIndex(x,y,z,GI.grid_width));
			if (isOutlier(x,y,z,center.x,center.y,center.z,GI.grid_width,GI.sector_offset))
				continue;
			gdata[ind].x += temp_gdata[s_ind];//Re
			gdata[ind].y += temp_gdata[s_ind+1];//Im
		}
	}
}

void performConvolution( DType* data_d, 
						 DType* crds_d, 
						 CufftType* gdata_d,
						 DType* kernel_d, 
						 int* sectors_d, 
						 int* sector_centers_d,
						 DType* temp_gdata_d,
						 GriddingInfo* gi_host
						)
{
	long shared_mem_size = 2*gi_host->sector_dim*sizeof(DType);
	
	dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,N_THREADS_PER_SECTOR);
	dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,(gi_host->sector_pad_width)*(gi_host->sector_pad_width)*(N_THREADS_PER_SECTOR)));
	if (DEBUG)
		printf("convolution requires %d bytes of shared memory!\n",shared_mem_size);
	convolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,gi_host->sector_count);
}

//very slow way of composing the output, should only be used on compute capabilties lower than 2.0
void composeOutput(DType* temp_gdata_d, CufftType* gdata_d, int* sector_centers_d, GriddingInfo* gi_host)
{
	dim3 grid_dim(1);
	dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,N_THREADS_PER_SECTOR);
	
	composeOutputKernel<<<grid_dim,block_dim>>>(temp_gdata_d,gdata_d,sector_centers_d);
}

__global__ void forwardConvolutionKernel( CufftType* data, 
										  DType* crds, 
										  CufftType* gdata,
										  DType* kernel, 
										  int* sectors, 
										  int* sector_centers,
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

		__shared__ int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];
		//__syncthreads();
		//Grid Points over Threads
		int data_cnt = sectors[sec] + threadIdx.x;

		__shared__ int sector_ind_offset;
		sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.grid_width);

		while (data_cnt < sectors[sec+1])
		{
			DType3 data_point; //datapoint per thread
			data_point.x = crds[3*data_cnt];
			data_point.y = crds[3*data_cnt +1];
			data_point.z = crds[3*data_cnt +2];

			// set the boundaries of final dataset for gridding this point
			ix = (data_point.x + 0.5f) * (GI.grid_width) - center.x + GI.sector_offset;
			set_minmax(&ix, &imin, &imax, GI.sector_pad_max, GI.kernel_radius);
			jy = (data_point.y + 0.5f) * (GI.grid_width) - center.y + GI.sector_offset;
			set_minmax(&jy, &jmin, &jmax, GI.sector_pad_max, GI.kernel_radius);
			kz = (data_point.z + 0.5f) * (GI.grid_width) - center.z + GI.sector_offset;
			set_minmax(&kz, &kmin, &kmax, GI.sector_pad_max, GI.kernel_radius);

			// convolve neighboring cartesian points to this data point
			k = kmin;
			while (k<=kmax && k>=kmin)
			{
				kz = static_cast<DType>((k + center.z - GI.sector_offset)) / static_cast<DType>((GI.grid_width)) - 0.5f;//(k - center_z) *width_inv;
				dz_sqr = kz - data_point.z;
				dz_sqr *= dz_sqr;
				
				if (dz_sqr < GI.radiusSquared)
				{
					j=jmin;
					while (j<=jmax && j>=jmin)
					{
						jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.grid_width)) - 0.5f;   //(j - center_y) *width_inv;
						dy_sqr = jy - data_point.y;
						dy_sqr *= dy_sqr;
						if (dy_sqr < GI.radiusSquared)	
						{
							i=imin;								
							while (i<=imax && i>=imin)
							{
								ix = static_cast<DType>(i + center.x - GI.sector_offset) / static_cast<DType>((GI.grid_width)) - 0.5f;// (i - center_x) *width_inv;
								dx_sqr = ix - data_point.x;
								dx_sqr *= dx_sqr;
								if (dx_sqr < GI.radiusSquared)	
								{
									// get kernel value
									// calc as separable filter
									val = kernel[(int) round(dz_sqr * GI.dist_multiplier)] *
											kernel[(int) round(dy_sqr * GI.dist_multiplier)] *
											kernel[(int) round(dx_sqr * GI.dist_multiplier)];
									
									ind = (sector_ind_offset + getIndex(i,j,k,GI.grid_width));

									// multiply data by current kernel val 
									// grid complex or scalar 
									if (isOutlier(i,j,k,center.x,center.y,center.z,GI.grid_width,GI.sector_offset))
									{
										i++;
										continue;
									}
				
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

void performForwardConvolution( CufftType*		data_d, 
								DType*			crds_d, 
								CufftType*		gdata_d,
								DType*			kernel_d, 
								int*			sectors_d, 
								int*			sector_centers_d,
								GriddingInfo*	gi_host
								)
{
	int thread_size = 128;
	long shared_mem_size = thread_size * sizeof(CufftType);//empiric

//	dim3 block_dim(128);
//	dim3 grid_dim(gi_host->sector_count);

	dim3 block_dim(thread_size);
	dim3 grid_dim(getOptimalGridDim(gi_host->sector_count,thread_size));
	
	if (DEBUG)
		printf("convolution requires %d bytes of shared memory!\n",shared_mem_size);
	forwardConvolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,gi_host->sector_count);
}
