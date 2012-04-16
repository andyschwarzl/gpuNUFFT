#include "cuda_utils.hpp"
#include "cuda_utils.cuh"

#include "gridding_gpu.hpp"

//Simple Test Kernel 
#define N 1000 //DIM ^3 
#define DIM 10
__global__ void kernel_call(int *a)
{
    int tx = threadIdx.x;
		int ty = threadIdx.y;
		int tz = threadIdx.z;

		int index = tx + DIM * (ty + tz * DIM);
		
		while (index < N)
		{
			a[index] = index;
			tz += blockDim.z;
			index = tx + DIM * (ty + tz * DIM);
		}
}

#define N_THREADS_PER_SECTOR 2 //16x16
#define SECTOR_WIDTH 10

__global__ void griddingKernel( DType* data, 
							    DType* crds, 
							    CufftType* gdata,
							    DType* kernel, 
							    int* sectors, 
								int* sector_centers,
								DType* temp_gdata
								)
{
	//TODO static or dynamic?
	__shared__ DType sdata[2*SECTOR_WIDTH*SECTOR_WIDTH*SECTOR_WIDTH]; //ca. 8kB -> 2 Blöcke je SM ???

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
		__shared__ int max_x;
		__shared__ int max_y;
		__shared__ int max_z; 
		__shared__ int imin;
		__shared__ int imax;
		__shared__ int jmin;
		__shared__ int jmax;
		__shared__ int kmin;
		__shared__ int kmax;

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
				ix = (data_point.x + 0.5f) * (GI.width) - center.x + GI.sector_offset;
				set_minmax(ix, &imin, &imax, max_x, GI.kernel_radius);
				jy = (data_point.y + 0.5f) * (GI.width) - center.y + GI.sector_offset;
				set_minmax(jy, &jmin, &jmax, max_y, GI.kernel_radius);
				kz = (data_point.z + 0.5f) * (GI.width) - center.z + GI.sector_offset;
				set_minmax(kz, &kmin, &kmax, max_z, GI.kernel_radius);

				// grid this point onto the neighboring cartesian points
				for (k=threadIdx.z;k<=kmax; k += blockDim.z)
				{
					//__syncthreads();
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
		int sector_ind_offset = sec * GI.sector_pad_width*GI.sector_pad_width*GI.sector_pad_width;
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

__global__ void composeOutput(DType* temp_gdata, CufftType* gdata, int* sector_centers)
{
	for (int sec = 0; sec < GI.sector_count; sec++)
	{
		__syncthreads();
		__shared__ int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];
		__shared__ int sector_ind_offset;
		sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.width);
		__shared__ int sector_grid_offset;
		sector_grid_offset = sec * GI.sector_pad_width*GI.sector_pad_width*GI.sector_pad_width;
		//write data from temp grid to overall output grid
		for (int z=threadIdx.z;z<GI.sector_pad_width; z += blockDim.z)
		{
			int y=threadIdx.y;
			int x=threadIdx.x;
			int s_ind = 2* (sector_grid_offset + getIndex(x,y,z,GI.sector_pad_width));
			int ind = (sector_ind_offset + getIndex(x,y,z,GI.width));
			if (isOutlier(x,y,z,center.x,center.y,center.z,GI.width,GI.sector_offset))
						continue;
			gdata[ind].x += temp_gdata[s_ind];//Re
			gdata[ind].y += temp_gdata[s_ind+1];//Im
		}
	}
}

void runSimpleKernelCall()
{
	printf("starting gpu implementation\n");
	int test_a[N];
	int* test_ad;

	allocateAndCopyToDeviceMem<int>(&test_ad,test_a,N);

	dim3 grid_size = 1;
	dim3 thread_size(10,10,3);
	printf("dimensions %d,%d,%d \n",thread_size.x,thread_size.y,thread_size.z);
	kernel_call<<<grid_size,thread_size>>>(test_ad);
	
	copyFromDevice<int>(test_ad,test_a,N);

	printf("output: ");
	for (int i = 0; i < N; i++)
		 printf("%d ",test_a[i]);
	printf("\n");
	
	freeDeviceMem(test_ad);
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
					int width,
					const GriddingOutput gridding_out)
{
	//runSimpleKernelCall();
		
	assert(sectors != NULL);
	
	//split and run sectors into blocks
	//and each data point to one thread inside this block 

	GriddingInfo* gi_host = initAndCopyGriddingInfo(sector_count,sector_width,kernel_width,kernel_count,width);
	
	DType* data_d, *crds_d, *kernel_d, *temp_gdata_d;
	CufftType *gdata_d;
	int* sector_centers_d, *sectors_d;

	printf("allocate and copy gdata of size %d...\n",gdata_cnt);
	allocateAndCopyToDeviceMem<CufftType>(&gdata_d,gdata,gdata_cnt);//Konvention!!!

	printf("allocate and copy data of size %d...\n",2*data_cnt);
	allocateAndCopyToDeviceMem<DType>(&data_d,data,2*data_cnt);

	int temp_grid_cnt = 2 * sector_count * gi_host->sector_dim;
	//TODO delete
	DType* temp_gdata = (DType*) calloc(temp_grid_cnt,sizeof(DType));

	printf("allocate temp grid data of size %d...\n",temp_grid_cnt);
	allocateAndCopyToDeviceMem<DType>(&temp_gdata_d,temp_gdata,temp_grid_cnt);
	
	printf("allocate and copy coords of size %d...\n",3*data_cnt);
	allocateAndCopyToDeviceMem<DType>(&crds_d,crds,3*data_cnt);
	
	printf("allocate and copy kernel of size %d...\n",kernel_cnt);
	allocateAndCopyToDeviceMem<DType>(&kernel_d,kernel,kernel_cnt);
	printf("allocate and copy sectors of size %d...\n",2*sector_count);
	allocateAndCopyToDeviceMem<int>(&sectors_d,sectors,2*sector_count);
	printf("allocate and copy sector_centers of size %d...\n",3*sector_count);
	allocateAndCopyToDeviceMem<int>(&sector_centers_d,sector_centers,3*sector_count);
	
	dim3 block_dim(SECTOR_WIDTH,SECTOR_WIDTH,N_THREADS_PER_SECTOR);

    griddingKernel<<<sector_count,block_dim>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d);

	//compose total output from local blocks 
	composeOutput<<<1,block_dim>>>(temp_gdata_d,gdata_d,sector_centers_d);

	if (gridding_out == CONVOLUTION)
	{
		printf("stopping output after CONVOLUTION step\n");
		//get output
		copyFromDevice<CufftType>(gdata_d,gdata,gdata_cnt);
		//free memory //printf("%p\n%p\n%p\n%p\n%p\n%p\n%p\n",data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d);
		freeTotalDeviceMemory(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,NULL);//NULL as stop token
		free(gi_host);
		return;
	}

	//TODO Inverse fft
	cufftHandle fft_plan;
	cufftPlan3d(&fft_plan, GI.width,GI.width,GI.width, CUFFT_C2C) ;
	int err;
	
	//Inverse FFT
	if (err=cufftExecC2C(fft_plan, gdata_d, gdata_d, CUFFT_INVERSE) != CUFFT_SUCCESS)
	{
      printf("cufft has failed with err %i \n",err);
      return;
	}
	if (gridding_out == FFT)
	{
		printf("stopping output after FFT step\n");
		//get output
		copyFromDevice<CufftType>(gdata_d,gdata,gdata_cnt);
		//free memory
		freeTotalDeviceMemory(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,NULL);//NULL as stop token
		free(gi_host);
		return;
	}
	
	//TODO deapodization
	copyFromDevice<CufftType>(gdata_d,gdata,gdata_cnt);
	
	freeDeviceMem(data_d);
	freeDeviceMem(crds_d);
	freeDeviceMem(gdata_d);
	freeDeviceMem(kernel_d);
	freeDeviceMem(sectors_d);
	freeDeviceMem(sector_centers_d);
	freeDeviceMem(temp_gdata_d);
	free(gi_host);
}
