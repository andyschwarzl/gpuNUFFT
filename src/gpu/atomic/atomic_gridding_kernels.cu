#include "gridding_kernels.hpp"
#include "cuda_utils.cuh"

// convolve every data point on grid position -> controlled by threadIdx.x .y and .z 
// shared data holds grid values as software managed cache
//
// like non atomic gridding, but without temporaray grid
// atomic write to global output grid
// TODO evaluate performance
__global__ void convolutionKernelFromGrid(  DType* data, 
											DType* crds, 
											CufftType* gdata,
											DType* kernel, 
											int* sectors, 
											int* sector_centers
											)
{
	extern __shared__ DType sdata[];//externally managed shared memory

	int sec= blockIdx.x;
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
	if (sec < GI.sector_count)
	{
		int ind, k, i, j;
		__shared__ int max_x, max_y, max_z, imin, imax,jmin,jmax,kmin,kmax;

		DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

		__shared__ int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];

		//Grid Positions controlled over thread inidices
		int data_cnt;
		data_cnt = sectors[sec];

		//loop over all data points of the current sector, and check if grid position lies inside 
		//affected region, if so, add data point weighted to grid position value
		while (data_cnt < sectors[sec+1])
		{
			__shared__ DType3 data_point; //datapoint shared in every thread
			data_point.x = crds[3*data_cnt];
			data_point.y = crds[3*data_cnt +1];
			data_point.z = crds[3*data_cnt +2];

			max_x = GI.sector_pad_width-1;
			max_y = GI.sector_pad_width-1;
			max_z = GI.sector_pad_width-1;

			// set the boundaries of final dataset for gridding this point
			ix = (data_point.x + 0.5f) * (GI.grid_width) - center.x + GI.sector_offset;
			set_minmax(&ix, &imin, &imax, max_x, GI.kernel_radius);
			jy = (data_point.y + 0.5f) * (GI.grid_width) - center.y + GI.sector_offset;
			set_minmax(&jy, &jmin, &jmax, max_y, GI.kernel_radius);
			kz = (data_point.z + 0.5f) * (GI.grid_width) - center.z + GI.sector_offset;
			set_minmax(&kz, &kmin, &kmax, max_z, GI.kernel_radius);
				                
			// grid this point onto its cartesian points neighbors
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
			__syncthreads();
			data_cnt++;
		} //grid points per sector
	
		//write shared data to output grid
		__syncthreads();
		//int sector_ind_offset = sec * GI.sector_dim;
		int sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.grid_width);
		//each thread position
		for (int z=threadIdx.z;z<GI.sector_pad_width; z += blockDim.z)
		{
			int y=threadIdx.y;
			int x=threadIdx.x;
			
			int s_ind = 2 * getIndex(x,y,z,GI.sector_pad_width) ;//index in shared grid
			int ind = sector_ind_offset + getIndex(x,y,z,GI.grid_width);//index in output grid
			
			if (isOutlier(x,y,z,center.x,center.y,center.z,GI.grid_width,GI.sector_offset))
				continue;

			atomicAdd(&(gdata[ind].x),sdata[s_ind]);//Re
			atomicAdd(&(gdata[ind].y),sdata[s_ind+1]);//Im
		}
	}//sec < sector_count	
}

__global__ void convolutionKernel2( DType* data, 
									DType* crds, 
									CufftType* gdata,
									DType* kernel, 
									int* sectors, 
									int* sector_centers
									)
{
	extern __shared__ DType sdata[];//externally managed shared memory
	int sec= blockIdx.x;

	if (sec < GI.sector_count)
	{
		//init shared memory
		for (int s_ind=threadIdx.x;s_ind<GI.sector_dim; s_ind+= blockDim.x)
		{
			sdata[2*s_ind] = 0.0f;//Re
			sdata[2*s_ind+1]=0.0f;//Im
		}
		__syncthreads();
	
		//start convolution
		int ind, k, i, j;
		__shared__ int max_x, max_y, max_z, imin, imax,jmin,jmax,kmin,kmax;

		DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

		__shared__ int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];

		//Grid Points over Threads
		int data_cnt = sectors[sec] + threadIdx.x;

		//loop over all data points of the current sector, and check if grid position lies inside 
		//affected region, if so, add data point weighted to grid position value
		while (data_cnt < sectors[sec+1])
		{
			__shared__ DType3 data_point; //datapoint shared in every thread
			data_point.x = crds[3*data_cnt];
			data_point.y = crds[3*data_cnt +1];
			data_point.z = crds[3*data_cnt +2];

			max_x = GI.sector_pad_width-1;
			max_y = GI.sector_pad_width-1;
			max_z = GI.sector_pad_width-1;

			// set the boundaries of final dataset for gridding this point
			ix = (data_point.x + 0.5f) * (GI.grid_width) - center.x + GI.sector_offset;
			set_minmax(&ix, &imin, &imax, max_x, GI.kernel_radius);
			jy = (data_point.y + 0.5f) * (GI.grid_width) - center.y + GI.sector_offset;
			set_minmax(&jy, &jmin, &jmax, max_y, GI.kernel_radius);
			kz = (data_point.z + 0.5f) * (GI.grid_width) - center.z + GI.sector_offset;
			set_minmax(&kz, &kmin, &kmax, max_z, GI.kernel_radius);
				                
			// grid this point onto its cartesian points neighbors
			k =kmin;
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
							i= imin;						
							while (i<=imax && i>=imin)
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
									atomicAdd(&(sdata[ind]),val * data[2*data_cnt]);
									atomicAdd(&(sdata[ind+1]),val * data[2*data_cnt+1]);
								} // kernel bounds check x, spherical support 
								i++;
							} // x 	 
						} // kernel bounds check y, spherical support 
						j++;
					} // y 
				} //kernel bounds check z 
				k++;
			} // z
			data_cnt = data_cnt + blockDim.x;
		} //grid points per sector
	
		//write shared data to output grid
		__syncthreads();
		//int sector_ind_offset = sec * GI.sector_dim;
		int sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.grid_width);
		
		//each thread writes one position from shared mem to global mem
		for (int s_ind=threadIdx.x;s_ind<GI.sector_dim; s_ind += blockDim.x)
		{
			/*if (sector_ind_offset + s_ind < GI.grid_width_dim)
			{
				atomicAdd(&(gdata[sector_ind_offset + s_ind].x),0.0f);
				atomicAdd(&(gdata[sector_ind_offset + s_ind+1].y),0.0f);
			}*/

			int z = (int)(s_ind / (GI.grid_width*GI.grid_width)) ;
			int r = s_ind - z * GI.grid_width * GI.grid_width;
			int y = (int)(r / GI.grid_width);
			int x = s_ind % GI.grid_width;

			int ind = sector_ind_offset + getIndex(x,y,z,GI.grid_width);//index in output grid
			
			if (isOutlier(x,y,z,center.x,center.y,center.z,GI.grid_width,GI.sector_offset))
				continue;

			atomicAdd(&(gdata[ind].x),1.0f);//sdata[2*s_ind]);//Re
			atomicAdd(&(gdata[ind].y),1.0f);//sdata[2*s_ind+1]);//Im
		}
	}//sec < sector_count	
}

//
// convolve every data point on grid position -> controlled by threadIdx.x .y and .z 
// shared data holds grid values as software managed cache
//
//
__global__ void convolutionKernel( DType* data, 
							    DType* crds, 
							    CufftType* gdata,
							    DType* kernel, 
							    int* sectors, 
								int* sector_centers
								)
{
	extern __shared__ DType sdata[]; //externally managed shared memory

	int  sec= blockIdx.x;
	//start convolution
	if (sec < GI.sector_count)
	{
		//shared???
		int ind, imin, imax, jmin, jmax,kmin,kmax, k, i, j;
		__shared__ int max_x;
		__shared__ int max_y;
		__shared__ int max_z;

		DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

		int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];

		//Grid Points over Threads
		int data_cnt = sectors[sec] + threadIdx.x;

		int sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.grid_width);
		
		max_x = GI.sector_pad_width-1;
		max_y = GI.sector_pad_width-1;
		max_z = GI.sector_pad_width-1;
		
		while (data_cnt < sectors[sec+1])
		{
			DType3 data_point; //datapoint per thread
			data_point.x = crds[3*data_cnt];
			data_point.y = crds[3*data_cnt +1];
			data_point.z = crds[3*data_cnt +2];
			
			// set the boundaries of final dataset for gridding this point
			ix = (data_point.x + 0.5f) * (GI.grid_width) - center.x + GI.sector_offset;
			set_minmax(&ix, &imin, &imax, max_x, GI.kernel_radius);
			jy = (data_point.y + 0.5f) * (GI.grid_width) - center.y + GI.sector_offset;
			set_minmax(&jy, &jmin, &jmax, max_y, GI.kernel_radius);
			kz = (data_point.z + 0.5f) * (GI.grid_width) - center.z + GI.sector_offset;
			set_minmax(&kz, &kmin, &kmax, max_z, GI.kernel_radius);

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
									//Berechnung mit Separable Filters 
									val = kernel[(int) round(dz_sqr * GI.dist_multiplier)] *
											kernel[(int) round(dy_sqr * GI.dist_multiplier)] *
											kernel[(int) round(dx_sqr * GI.dist_multiplier)];
									
									ind = sector_ind_offset + getIndex(i,j,k,GI.grid_width);//index in output grid
			
									if (isOutlier(i,j,k,center.x,center.y,center.z,GI.grid_width,GI.sector_offset))
									{
										i++;
										continue;
									}

									atomicAdd(&(gdata[ind].x),val * data[2*data_cnt]);//Re
									atomicAdd(&(gdata[ind].y),val * data[2*data_cnt+1]);//Im
								}// kernel bounds check x, spherical support 
								i++;
							} // x loop
						} // kernel bounds check y, spherical support  
						j++;
					} // y loop
				} //kernel bounds check z 
				k++;
			} // z loop
			data_cnt = data_cnt + blockDim.x;
		} //data points per sector
	} //sector check
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
	//XXX other Kernel - slow //TODO evaluate

	/*long shared_mem_size = 2*gi_host->sector_dim*sizeof(DType);

	dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,N_THREADS_PER_SECTOR);
	dim3 grid_dim(gi_host->sector_count);
	
	printf("adjoint convolution requires %d bytes of shared memory!\n",shared_mem_size);
	convolutionKernelFromGrid<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d);
	*/
	//TODO how to calculate shared_mem_size???, shared_mem_needed?
	/*long shared_mem_size = 128 * sizeof(CufftType);//empiric

	dim3 block_dim(128);
	dim3 grid_dim(gi_host->sector_count);
	
	printf("convolution requires %d bytes of shared memory!\n",shared_mem_size);
	convolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d);
	*/
	long shared_mem_size = 2*gi_host->sector_dim*sizeof(DType);

	dim3 block_dim(128);
	dim3 grid_dim(gi_host->sector_count);
	
	printf("adjoint convolution requires %d bytes of shared memory!\n",shared_mem_size);
	convolutionKernel2<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d);
	printf("...finished with: %s\n", cudaGetErrorString(cudaGetLastError()));
}

__global__ void forwardConvolutionKernel( CufftType* data, 
										  DType* crds, 
										  CufftType* gdata,
										  DType* kernel, 
										  int* sectors, 
										  int* sector_centers
											)
{
	extern __shared__ CufftType shared_out_data[];//externally managed shared memory
	//test
	CufftType out_data;


	int sec= blockIdx.x;
	//init shared memory
	//out_data[threadIdx.x].x = 0.0f;//Re
	//out_data[threadIdx.x].y = 0.0f;//Im

	out_data.x = 0.0f;//Re
	out_data.y = 0.0f;//Im

	__syncthreads();

	//start convolution
	if (sec < GI.sector_count)
	{
		//shared???
		int ind, max_x, max_y, max_z, imin, imax, jmin, jmax,kmin,kmax, k, i, j;
		DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

		int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];

		//Grid Points over Threads
		int data_cnt = sectors[sec] + threadIdx.x;
		//out_data[data_cnt].x = 0.0f;//Re
		//out_data[data_cnt].y = 0.0f;//Im
		out_data.x = 0.0f;//Re
		out_data.y = 0.0f;//Im
		//int sector_grid_offset = sec * GI.sector_dim;
		int sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.grid_width);
		
		while (data_cnt < sectors[sec+1])
		{
			DType3 data_point; //datapoint per thread
			data_point.x = crds[3*data_cnt];
			data_point.y = crds[3*data_cnt +1];
			data_point.z = crds[3*data_cnt +2];

			max_x = GI.sector_pad_width-1;
			max_y = GI.sector_pad_width-1;
			max_z = GI.sector_pad_width-1;

			// set the boundaries of final dataset for gridding this point
			ix = (data_point.x + 0.5f) * (GI.grid_width) - center.x + GI.sector_offset;
			set_minmax(&ix, &imin, &imax, max_x, GI.kernel_radius);
			jy = (data_point.y + 0.5f) * (GI.grid_width) - center.y + GI.sector_offset;
			set_minmax(&jy, &jmin, &jmax, max_y, GI.kernel_radius);
			kz = (data_point.z + 0.5f) * (GI.grid_width) - center.z + GI.sector_offset;
			set_minmax(&kz, &kmin, &kmax, max_z, GI.kernel_radius);

			// convolve neighboring cartesian points to this data point
			k = kmin;
			//data[data_cnt].x = -1.0f;//(float)k;
			//data[data_cnt].y =-1.0f;// (float)kmax;
			while (k<=kmax && k>=kmin)
			{
				kz = static_cast<DType>((k + center.z - GI.sector_offset)) / static_cast<DType>((GI.grid_width)) - 0.5f;//(k - center_z) *width_inv;
				dz_sqr = kz - data_point.z;
				dz_sqr *= dz_sqr;
				
				//data[data_cnt].x = kz;
				//data[data_cnt].y = center.z;//static_cast<DType>((k + center.z - 8 - GI.sector_offset)) / static_cast<DType>((GI.grid_width)) - 0.5f;
				if (dz_sqr < GI.radiusSquared)
				{
					//data[data_cnt].x = 1.1f;
					//data[data_cnt].y = 1.1f;
					j=jmin;
					while (j<=jmax && j>=jmin)
					{
						//data[data_cnt].x = 1.2f;
						//data[data_cnt].y = 1.2f;
						jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.grid_width)) - 0.5f;   //(j - center_y) *width_inv;
						dy_sqr = jy - data_point.y;
						dy_sqr *= dy_sqr;
						if (dy_sqr < GI.radiusSquared)	
						{
							//data[data_cnt].x = 1.3f;
							//data[data_cnt].y = 1.3f;
							i=imin;								
							while (i<=imax && i>=imin)
							{
								//data[data_cnt].x = 1.4f;
								//data[data_cnt].y = 1.4f;
								ix = static_cast<DType>(i + center.x - GI.sector_offset) / static_cast<DType>((GI.grid_width)) - 0.5f;// (i - center_x) *width_inv;
								dx_sqr = ix - data_point.x;
								dx_sqr *= dx_sqr;
								if (dx_sqr < GI.radiusSquared)	
								{
									//data[data_cnt].x = 1.5f;
									//data[data_cnt].y = 1.5f;
									// get kernel value
									//Berechnung mit Separable Filters 
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
				
									//out_data[data_cnt].x = 1.0f; //val * gdata[ind].x;
									//out_data[data_cnt].y = 1.0f; //val * gdata[ind].y;		
									out_data.x += gdata[ind].x * val; //+= /*val **/ gdata[ind].x;
									out_data.y += gdata[ind].y * val; //+= /*val **/ gdata[ind].y;
									
								}// kernel bounds check x, spherical support 
								i++;
							} // x loop
						} // kernel bounds check y, spherical support  
						j++;
					} // y loop
				} //kernel bounds check z 
				k++;
			} // z loop
			//data[data_cnt] = out_data[data_cnt];
			data[data_cnt].x = out_data.x;// / sqrt((DType)GI.kernel_width*GI.kernel_width*GI.kernel_width);
			data[data_cnt].y = out_data.y;// / sqrt((DType)GI.kernel_width*GI.kernel_width*GI.kernel_width);
			
			data_cnt = data_cnt + blockDim.x;

			//out_data[data_cnt].x = (DType)0.0f;
			//out_data[data_cnt].y = (DType)0.0f;
			out_data.x = 0.0f;//Re
			out_data.y = 0.0f;//Im
			//data[data_cnt] = out_data[data_cnt];
			//data_cnt++;
		} //data points per sector
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
	//TODO how to calculate shared_mem_size???, shared_mem_needed?
	long shared_mem_size = 128 * sizeof(CufftType);//empiric

	dim3 block_dim(128);
	dim3 grid_dim(gi_host->sector_count);
	
	printf("convolution requires %d bytes of shared memory!\n",shared_mem_size);
	forwardConvolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d);
}
