#include "gridding_kernels.hpp"
#include "cuda_utils.cuh"

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
								DType* temp_gdata
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

		//Grid Points over threads
		int data_cnt;
		data_cnt = sectors[sec];
			
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
			set_minmax(ix, &imin, &imax, max_x, GI.kernel_radius);
			jy = (data_point.y + 0.5f) * (GI.grid_width) - center.y + GI.sector_offset;
			set_minmax(jy, &jmin, &jmax, max_y, GI.kernel_radius);
			kz = (data_point.z + 0.5f) * (GI.grid_width) - center.z + GI.sector_offset;
			set_minmax(kz, &kmin, &kmax, max_z, GI.kernel_radius);
				                
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
			__syncthreads();
			data_cnt++;
		} //grid points per sector
	
	  //write shared data to temporary output grid
		int sector_ind_offset = sec * GI.sector_dim;
		for (int z=threadIdx.z;z<GI.sector_pad_width; z += blockDim.z)
		{
			int y=threadIdx.y;
			int x=threadIdx.x;
			
			int s_ind = 2* getIndex(x,y,z,GI.sector_pad_width) ;//index in shared grid
			ind = 2*sector_ind_offset + s_ind;//index in temp output grid
						
			temp_gdata[ind] = sdata[s_ind];//Re
			temp_gdata[ind+1] = sdata[s_ind+1];//Im
		}
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
			int y=threadIdx.y;
			int x=threadIdx.x;
			int s_ind = 2* (sector_grid_offset + getIndex(x,y,z,GI.sector_pad_width));
			int ind = (sector_ind_offset + getIndex(x,y,z,GI.grid_width));
			if (isOutlier(x,y,z,center.x,center.y,center.z,GI.grid_width,GI.sector_offset))
				continue;
			gdata[ind].x += temp_gdata[s_ind];//Re
			gdata[ind].y += temp_gdata[s_ind+1];//Im
		}
	}
}


__global__ void deapodizationKernel(CufftType* gdata, DType beta, DType norm_val)
{
	int x=blockIdx.x;
	int y=blockIdx.y;
	int z=threadIdx.x;

	int ind = getIndex(x,y,z,GI.im_width);
	
	DType deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
	
	//check if deapodization value is valid number
	if (!isnan(deapo))// == deapo)
	{
		gdata[ind].x = gdata[ind].x / deapo;//Re
		gdata[ind].y = gdata[ind].y / deapo;//Im
	}
}

__global__ void cropKernel(CufftType* gdata,CufftType* imdata, int offset)
{
	int x=blockIdx.x; //[0,N-1] N...im_width
	int y=blockIdx.y; //[0,N-1] N...im_width
	int z=threadIdx.x;//[0,N-1] N...im_width
	int grid_ind = getIndex(offset+x,offset+y,offset+z,GI.grid_width);
	int im_ind = getIndex(x,y,z,GI.im_width);

	imdata[im_ind] = gdata[grid_ind];
}

__global__ void fftShiftKernel(CufftType* gdata, int offset)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;

	//calculate "opposite" coord pair
	int x_opp = (x + offset) % GI.grid_width;
	int y_opp = (y + offset) % GI.grid_width;
	int z_opp = (z + offset) % GI.grid_width;

	//swap points
	CufftType temp = gdata[getIndex(x,y,z,GI.grid_width)];
	gdata[getIndex(x,y,z,GI.grid_width)] = gdata[getIndex(x_opp,y_opp,z_opp,GI.grid_width)];
	gdata[getIndex(x_opp,y_opp,z_opp,GI.grid_width)] = temp;

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
	dim3 grid_dim(gi_host->sector_count);
	
	printf("convolution requires %d bytes of shared memory!\n",shared_mem_size);
	convolutionKernel<<<grid_dim,block_dim,shared_mem_size>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d);
}

//very slow way of composing the output 
void composeOutput(DType* temp_gdata_d, CufftType* gdata_d, int* sector_centers_d, GriddingInfo* gi_host)
{
	dim3 grid_dim(1);
	dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,N_THREADS_PER_SECTOR);
	
	composeOutputKernel<<<grid_dim,block_dim>>>(temp_gdata_d,gdata_d,sector_centers_d);
}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performDeapodization(CufftType* imdata_d,
						  GriddingInfo* gi_host)
{
	DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

	dim3 grid_dim(gi_host->im_width,gi_host->im_width,1);	
	dim3 block_dim(gi_host->im_width);
	//Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
	DType norm_val = calculateDeapodizationValue(0,gi_host->grid_width_inv,gi_host->kernel_width,beta);
	norm_val = norm_val * norm_val * norm_val;

	deapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val);
}


void performCrop(CufftType* gdata_d,
				 CufftType* imdata_d,
				 GriddingInfo* gi_host)
{
	/*crop data 
    ind_off = (a.params.im_width * (double(a.params.osr)-1) / 2) + 1;
    ind_start = ind_off;
    ind_end = ind_start + a.params.im_width -1;
    ress = m(ind_start:ind_end,ind_start:ind_end,ind_start:ind_end,:);*/
	int ind_off = (int)(gi_host->im_width * ((DType)gi_host->osr - 1.0f)/(DType)2);
	printf("start cropping image with offset %d\n",ind_off);

	dim3 grid_dim(gi_host->im_width,gi_host->im_width,1);
	dim3 block_dim(gi_host->im_width);
	cropKernel<<<grid_dim,block_dim>>>(gdata_d,imdata_d,ind_off);
}

void performFFTShift(CufftType* gdata_d,
					 FFTShiftDir shift_dir,
					 int width)
{
	dim3 grid_dim((int)ceil(width/(DType)2.0));
	dim3 block_dim(width,width);
	int offset= 0;

	if (shift_dir == FORWARD)
	{
		offset = (int)ceil((DType)(width / (DType)2.0));
	}
	else
	{
		offset = (int)floor((DType)(width / (DType)2.0));
	}
	fftShiftKernel<<<block_dim,grid_dim>>>(gdata_d,offset);
}


__global__ void forwardDeapodizationKernel(DType* imdata, DType beta, DType norm_val)
{
	int x=blockIdx.x;
	int y=blockIdx.y;
	int z=threadIdx.x;

	int ind = 2*getIndex(x,y,z,GI.im_width);
	
	DType deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
	
	//TODO reciporcal or not????
	//check if deapodization value is valid number
	if (!isnan(deapo))// == deapo)
	{
		imdata[ind] = imdata[ind] / deapo; // / deapo;//Re
		imdata[ind+1] = imdata[ind+1] / deapo ; /// deapo;//Im
	}
}

__global__ void paddingKernel(DType* imdata,CufftType* gdata, int offset)
{
	int x=blockIdx.x; //[0,N-1] N...im_width
	int y=blockIdx.y; //[0,N-1] N...im_width
	int z=threadIdx.x;//[0,N-1] N...im_width

	int grid_ind =  getIndex(offset + x,offset + y,offset +z,GI.grid_width);

	int im_ind = 2*getIndex(x,y,z,GI.im_width);

	gdata[grid_ind].x = imdata[im_ind];
	gdata[grid_ind].y = imdata[im_ind+1];
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

		__shared__ int3 center;
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
			set_minmax(ix, &imin, &imax, max_x, GI.kernel_radius);
			jy = (data_point.y + 0.5f) * (GI.grid_width) - center.y + GI.sector_offset;
			set_minmax(jy, &jmin, &jmax, max_y, GI.kernel_radius);
			kz = (data_point.z + 0.5f) * (GI.grid_width) - center.z + GI.sector_offset;
			set_minmax(kz, &kmin, &kmax, max_z, GI.kernel_radius);

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
									out_data.y -= gdata[ind].y * val; //+= /*val **/ gdata[ind].y;
									
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

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performForwardDeapodization(DType* imdata_d,
						  GriddingInfo* gi_host)
{
	DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

	dim3 grid_dim(gi_host->im_width,gi_host->im_width,1);	
	dim3 block_dim(gi_host->im_width);
	//Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
	DType norm_val = calculateDeapodizationValue(0,gi_host->grid_width_inv,gi_host->kernel_width,beta);
	norm_val = norm_val * norm_val * norm_val;

	forwardDeapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val);
}

void performPadding(DType* imdata_d,
					CufftType* gdata_d,					
					GriddingInfo* gi_host)
{
	int ind_off = (int)(gi_host->im_width * ((DType)gi_host->osr -1.0f)/(DType)2.0f);

	printf("start cropping image with offset %d\n",ind_off);

	dim3 grid_dim(gi_host->im_width,gi_host->im_width,1);
	dim3 block_dim(gi_host->im_width);
	paddingKernel<<<grid_dim,block_dim>>>(imdata_d,gdata_d,ind_off);
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