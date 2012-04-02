#include "gridding_gpu.hpp"

#include "cuda_utils.hpp"

//Simple Test Kernel 
__global__ void kernel_call(int *a)
{
    int tx = threadIdx.x;
    
    switch( tx % 2 )
    {
        case 0:
     a[tx] = a[tx] + 2;
     break;
        case 1:
     a[tx] = a[tx] + 3;
     break;
    }
}

__constant__ GriddingInfo GI;
//extern __shared__ DType sdata_arr[];

#define N_THREADS_PER_SECTOR 256 //16x16

__global__ void griddingKernel( DType* data, 
							    DType* crds, 
							    DType* gdata,
							    DType* kernel, 
							    int* sectors, 
								int* sector_centers
								)
{
	__shared__ float sdata[2*10*10*10]; //ca. 8kB -> 2 Blöcke je SM ???

	int  sec= blockIdx.x;
	//TODO static or dynamic?
	//manually cast to correct type/pos
	//DType* sdata = (DType*)sdata_arr;
	for (int i=0; i<2*GI.sector_dim;i++)
		sdata[i]=0.0f;

	if (sec < GI.sector_count)
	{
		int ind, max_x, max_y, max_z, imin, imax, jmin, jmax,kmin,kmax, k, i, j;

		DType dx_sqr, dy_sqr, dz_sqr, val, ix, jy, kz;

		__shared__ int3 center;
		center.x = sector_centers[sec * 3];
		center.y = sector_centers[sec * 3 + 1];
		center.z = sector_centers[sec * 3 + 2];

			//Data Points ueber Threads abwickeln
			//for (int data_cnt = sectors[sec]+threadIdx.x; data_cnt < sectors[sec+1];data_cnt += N_THREADS_PER_SECTOR)
			//{
				int data_cnt = sectors[sec]+threadIdx.x;
				while (data_cnt < sectors[sec+1])
				{
				DType3 data_point; //shared????
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
				for (k=kmin; k<=kmax; k++)	
				{
					kz = static_cast<DType>((k + center.z - GI.sector_offset)) / static_cast<DType>((GI.width)) - 0.5f;//(k - center_z) *width_inv;
					dz_sqr = kz - data_point.z;
					dz_sqr *= dz_sqr;
					if (dz_sqr < GI.radiusSquared)
					{
						for (j=jmin; j<=jmax; j++)	
						{
							jy = static_cast<DType>(j + center.y - GI.sector_offset) / static_cast<DType>((GI.width)) - 0.5f;   //(j - center_y) *width_inv;
							dy_sqr = jy - data_point.y;
							dy_sqr *= dy_sqr;
							if (dy_sqr < GI.radiusSquared)	
							{
								for (i=imin; i<=imax; i++)	
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
										
										//atomicFloatAdd(&(sdata[ind]), val * data[2*data_cnt]);
										//atomicFloatAdd(&(sdata[ind+1]),val * data[2*data_cnt+1]);
									} // kernel bounds check x, spherical support 
								} // x 	 
							} // kernel bounds check y, spherical support 
						} // y 
					} //kernel bounds check z 
				} // z 
				__syncthreads();
				data_cnt += N_THREADS_PER_SECTOR;
				}
			//} //data points per sector
	
		__syncthreads();

		//TODO copy data from sectors to original grid
		if (threadIdx.x == 0)
		{
			int max_im_index = GI.width;
			int sector_ind_offset = getIndex(center.x - GI.sector_offset,center.y - GI.sector_offset,center.z - GI.sector_offset,GI.width);

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

						gdata[ind] += sdata[s_ind]; //Re
						gdata[ind+1] += sdata[s_ind+1];//Im
					}
				}
		}
	}//sec < sector_count
	
}

void runSimpleKernelCall()
{
	printf("starting gpu implementation\n");
	int test_h[3];
	test_h[0] = 3;
	test_h[1] = 4;
	test_h[2] = 6;	
	int* test_d;
	printf("input: %d , %d , %d\n",test_h[0],test_h[1],test_h[2]);
	
	allocateAndCopyToDeviceMem<int>(&test_d,test_h,3);

	dim3 grid_size = 1;
	dim3 thread_size = 3;

	kernel_call<<<grid_size,thread_size>>>(test_d);
	
	copyFromDevice<int>(test_d,test_h,3);

	printf("output: %d , %d , %d\n",test_h[0],test_h[1],test_h[2]);
	
	freeDeviceMem(test_d);
}

void initAndCopyGriddingInfo(int sector_count, 
							 int sector_width,
							 int kernel_width,
							 int kernel_count, 
							 int width)
{
	GriddingInfo* gi_host = (GriddingInfo*)malloc(sizeof(GriddingInfo));

	gi_host->sector_count = sector_count;
	gi_host->sector_width = sector_width;
	
	gi_host->kernel_width = kernel_width; 
	gi_host->kernel_count = kernel_count;
	gi_host->width = width;

	DType kernel_radius = static_cast<DType>(kernel_width) / 2.0f;
	DType radius = kernel_radius / static_cast<DType>(width);
	DType width_inv = 1.0f / width;
	DType radiusSquared = radius * radius;
	DType kernelRadius_invSqr = 1 / radiusSquared;
	DType dist_multiplier = (kernel_count - 1) * kernelRadius_invSqr;
	printf("radius rel. to grid width %f\n",radius);
	int sector_pad_width = sector_width + 2*(int)(floor(kernel_width / 2.0f));
	int sector_dim = sector_pad_width  * sector_pad_width  * sector_pad_width ;
	int sector_offset = (int)(floor(sector_pad_width / 2.0f));

	gi_host->kernel_radius = kernel_radius;
	gi_host->sector_pad_width = sector_pad_width;
	gi_host->sector_dim = sector_dim;
	gi_host->sector_offset = sector_offset;
	gi_host->radiusSquared = radiusSquared;
	gi_host->dist_multiplier = dist_multiplier;

	printf("sector offset = %d\n",sector_offset);
	
	gi_host->sector_pad_width = sector_pad_width;
	
	printf("copy Gridding Info to symbol memory...\n");
	cudaMemcpyToSymbol(GI, gi_host,sizeof(GriddingInfo));
	free(gi_host);
	printf("...done!\n");
}

void gridding3D_gpu(DType* data, 
					int data_cnt,
					DType* crds, 
					DType* gdata,
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
	runSimpleKernelCall();
		
	assert(sectors != NULL);
	
	//split and run sectors into blocks
	//and each data point to one thread inside this block 

	initAndCopyGriddingInfo(sector_count,sector_width,kernel_width,kernel_count,width);
	
	DType* data_d, *crds_d, *gdata_d, *kernel_d;
	int* sector_centers_d, *sectors_d;
	printf("allocate and copy gdata of size %d...\n",gdata_cnt);
	allocateAndCopyToDeviceMem<DType>(&gdata_d,gdata,gdata_cnt);//Konvention!!!
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
	
	griddingKernel<<<sector_count,N_THREADS_PER_SECTOR>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d);
	
	copyFromDevice<DType>(gdata_d,gdata,gdata_cnt);
	
	freeDeviceMem(data_d);
	freeDeviceMem(crds_d);
	freeDeviceMem(gdata_d);
	freeDeviceMem(kernel_d);
	freeDeviceMem(sectors_d);
	freeDeviceMem(sector_centers_d);
}