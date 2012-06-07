#include "cuda_utils.hpp"
#include "cuda_utils.cuh" 

__global__ void griddingKernel( DType* data, 
							    DType* crds, 
							    CufftType* gdata,
							    DType* kernel, 
							    int* sectors, 
								int* sector_centers
								)
{
	__shared__ DType sdata[2*SECTOR_WIDTH*SECTOR_WIDTH*SECTOR_WIDTH]; //ca. 8kB -> 2 Blöcke je SM ???

	int  sec= blockIdx.x;
	//TODO static or dynamic?
	//manually cast to correct type/pos
	//DType* sdata = (DType*)sdata_arr;
	for (int i=0; i<2*SECTOR_WIDTH*SECTOR_WIDTH*SECTOR_WIDTH;i++)
		sdata[i]=0.0f;

	if (sec < GI.sector_count)
	{
		int ind, max_x, max_y, max_z, imin, imax, jmin, jmax,kmin,kmax, k, i, j;

		DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

		__shared__ int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];

			//Grid Points ueber Threads abwickeln
			int data_cnt = sectors[sec];
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
				ix = (data_point.x + 0.5f) * (GI.width) - center.x + GI.sector_offset;
				set_minmax(ix, &imin, &imax, max_x, GI.kernel_radius);
				jy = (data_point.y + 0.5f) * (GI.width) - center.y + GI.sector_offset;
				set_minmax(jy, &jmin, &jmax, max_y, GI.kernel_radius);
				kz = (data_point.z + 0.5f) * (GI.width) - center.z + GI.sector_offset;
				set_minmax(kz, &kmin, &kmax, max_z, GI.kernel_radius);

				// grid this point onto the neighboring cartesian points
				for (k=threadIdx.z;k<=kmax; k += blockDim.z)
				{
					if (k<=kmax && k>=kmin)
					{
						kz = static_cast<DType>((k + center.z - GI.sector_offset)) / static_cast<DType>((GI.width)) - 0.5f;//(k - center_z) *width_inv;
						dz_sqr = kz - data_point.z;
						dz_sqr *= dz_sqr;
						if (dz_sqr < GI.radiusSquared)
						{
							j=threadIdx.y;
							if (j<=jmax && j>=jmin)
							{
								jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.width)) - 0.5f;   //(j - center_y) *width_inv;
								dy_sqr = jy - data_point.y;
								dy_sqr *= dy_sqr;
								if (dy_sqr < GI.radiusSquared)	
								{
									i=threadIdx.x;
								
									if (i<=imax && i>=imin)
									{
										ix = static_cast<DType>(i + center.x - GI.sector_offset) / static_cast<DType>((GI.width)) - 0.5f;// (i - center_x) *width_inv;
										dx_sqr = ix - data_point.x;
										dx_sqr *= dx_sqr;
										if (dx_sqr < GI.radiusSquared)	
										{
											// get kernel value
											//Berechnung mit Separable Filters 
											val = kernel[(int) round(dz_sqr * GI.dist_multiplier)] *
													kernel[(int) round(dy_sqr * GI.dist_multiplier)] *
													kernel[(int) round(dx_sqr * GI.dist_multiplier)];
											ind = 2* getIndex(i,j,k,GI.sector_pad_width);
								
											// multiply data by current kernel val 
											// grid complex or scalar 
										
										atomicAdd((float*)(&sdata[ind]), val * data[2*data_cnt]);
										atomicAdd((float*)&sdata[ind+1], val * data[2*data_cnt+1]);
										
										__syncthreads();
										}
									} // kernel bounds check x, spherical support 
								} // x 	 
							} // kernel bounds check y, spherical support 
						} // y 
					} //kernel bounds check z 
				} // z 
				data_cnt += N_THREADS_PER_SECTOR;
				}
			//} //data points per sector
		__syncthreads();

		//TODO copy data from sectors to original grid
		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		{
			//int max_im_index = GI.width;
			//int sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.width);
			int sector_ind_offset = sec * GI.sector_pad_width*GI.sector_pad_width*GI.sector_pad_width;
			for (int z = 0; z < GI.sector_pad_width; z++)
				for (int y = 0; y < GI.sector_pad_width; y++)
				{
					for (int x = 0; x < GI.sector_pad_width; x++)
					{
						int s_ind = 2* getIndex(x,y,z,GI.sector_pad_width) ;
						ind = 2*(sector_ind_offset + getIndex(x,y,z,GI.width));
						//TODO auslagern
						if (isOutlier(x,y,z,center.x,center.y,center.z,GI.width,GI.sector_offset))
							continue;
						
						atomicAdd((float*)&gdata[ind],sdata[s_ind]); //Re
						atomicAdd((float*)&gdata[ind+1],sdata[s_ind+1]);//Im
					}
				}
		}
	}//sec < sector_count
	
}

void gridding3D_gpu(DType* data, 
					int data_cnt,
					DType* crds, 
					CufftType* gdata,
					int gdata_cnt,
					DType* kernel,
					int kernel_cnt,
					int* sectors, 
					int sector_count, 
					int* sector_centers,
					int sector_width,
					int kernel_width, 
					int kernel_count, 
					int width)
{
	assert(sectors != NULL);
	
	//split and run sectors into blocks
	//and each data point to one thread inside this block 

	GriddingInfo* gi_host = initAndCopyGriddingInfo(sector_count,sector_width,kernel_width,kernel_count,width);
	
	DType* data_d, *crds_d, *kernel_d;
	CufftType* gdata_d;
	int* sector_centers_d, *sectors_d;

	printf("allocate and copy gdata of size %d...\n",gdata_cnt);
	allocateAndCopyToDeviceMem<CufftType>(&gdata_d,gdata,gdata_cnt);//Konvention!!!

	printf("allocate and copy data of size %d...\n",2*data_cnt);
	allocateAndCopyToDeviceMem<DType>(&data_d,data,2*data_cnt);

	printf("allocate and copy coords of size %d...\n",3*data_cnt);
	allocateAndCopyToDeviceMem<DType>(&crds_d,crds,3*data_cnt);
	
	printf("allocate and copy kernel of size %d...\n",kernel_cnt);
	allocateAndCopyToDeviceMem<DType>(&kernel_d,kernel,kernel_cnt);
	printf("allocate and copy sectors of size %d...\n",2*sector_count);
	allocateAndCopyToDeviceMem<int>(&sectors_d,sectors,2*sector_count);
	printf("allocate and copy sector_centers of size %d...\n",3*sector_count);
	allocateAndCopyToDeviceMem<int>(&sector_centers_d,sector_centers,3*sector_count);
	
	dim3 block_dim(SECTOR_WIDTH,SECTOR_WIDTH,2);

    griddingKernel<<<sector_count,block_dim>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d);

	copyFromDevice<CufftType>(gdata_d,gdata,gdata_cnt);
	
	freeDeviceMem(data_d);
	freeDeviceMem(crds_d);
	freeDeviceMem(gdata_d);
	freeDeviceMem(kernel_d);
	freeDeviceMem(sectors_d);
	freeDeviceMem(sector_centers_d);
	free(gi_host);
}
