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

__global__ void griddingKernel( float* data, 
							    float* crds, 
							    float* gdata,
							    float* kernel, 
							    int* sectors, 
								int* sector_centers
								)
{
	__shared__ float sdata[2* 10*10*10];
	int sec = blockIdx.x;

	if (sec == 0)
		for (int i=0; i<2*10*10*10;i++)
			sdata[i]=0.0f;

	if (sec < GI.sector_count)
	{
		
		int ind, center_x, center_y, center_z, max_x, max_y, max_z, imin, imax, jmin, jmax,kmin,kmax, k, i, j;

		float dx_sqr, dy_sqr, dz_sqr, val, x, y, z, ix, jy, kz;
	
			center_x = sector_centers[sec * 3];
			center_y = sector_centers[sec * 3 + 1];
			center_z = sector_centers[sec * 3 + 2];

			//printf("\nhandling center (%d,%d,%d) in sector %d\n",center_x,center_y,center_z,sec);
			for (int data_cnt = sectors[sec]; data_cnt < sectors[sec+1];data_cnt++)
			{
				//printf("handling %d data point = %f\n",data_cnt+1,data[data_cnt]);

				x = crds[3*data_cnt];
				y = crds[3*data_cnt +1];
				z = crds[3*data_cnt +2];
				//printf("data k-space coords (%f, %f, %f)\n",x,y,z);
			
				max_x = GI.sector_pad_width-1;
				max_y = GI.sector_pad_width-1;
				max_z = GI.sector_pad_width-1;

				// set the boundaries of final dataset for gridding this point
				ix = (x + 0.5f) * (GI.width) - center_x + GI.sector_offset;
				set_minmax(ix, &imin, &imax, max_x, GI.kernel_radius);
				jy = (y + 0.5f) * (GI.width) - center_y + GI.sector_offset;
				set_minmax(jy, &jmin, &jmax, max_y, GI.kernel_radius);
				kz = (z + 0.5f) * (GI.width) - center_z + GI.sector_offset;
				set_minmax(kz, &kmin, &kmax, max_z, GI.kernel_radius);

				// grid this point onto the neighboring cartesian points
				for (k=kmin; k<=kmax; k++)	
				{
					kz = static_cast<float>((k + center_z - GI.sector_offset)) / static_cast<float>((GI.width)) - 0.5f;//(k - center_z) *width_inv;
					dz_sqr = kz - z;
					dz_sqr *= dz_sqr;
					if (dz_sqr < GI.radiusSquared)
					{
						for (j=jmin; j<=jmax; j++)	
						{
							jy = static_cast<float>(j + center_y - GI.sector_offset) / static_cast<float>((GI.width)) - 0.5f;   //(j - center_y) *width_inv;
							dy_sqr = jy - y;
							dy_sqr *= dy_sqr;
							if (dy_sqr < GI.radiusSquared)	
							{
								for (i=imin; i<=imax; i++)	
								{
									ix = static_cast<float>(i + center_x - GI.sector_offset) / static_cast<float>((GI.width)) - 0.5f;// (i - center_x) *width_inv;
									dx_sqr = ix - x;
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
										sdata[ind] += val * data[2*data_cnt];
										sdata[ind+1] += val * data[2*data_cnt+1];
									} // kernel bounds check x, spherical support 
								} // x 	 
							} // kernel bounds check y, spherical support 
						} // y 
					} //kernel bounds check z 
				} // z 
			} //data points per sector
	
		//}//sectors
	
		//TODO copy data from sectors to original grid
		int max_im_index = GI.width;
		for (int sec = 0; sec < GI.sector_count; sec++)
		{
			//printf("DEBUG: showing entries of sector %d in z = 5 plane...\n",sec);
			center_x = sector_centers[sec * 3];
			center_y = sector_centers[sec * 3 + 1];
			center_z = sector_centers[sec * 3 + 2];
		
			int sector_ind_offset = getIndex(center_x - GI.sector_offset,center_y - GI.sector_offset,center_z - GI.sector_offset,GI.width);

			//printf("sector index offset in resulting grid: %d\n", sector_ind_offset);
			for (int z = 0; z < GI.sector_pad_width; z++)
				for (int y = 0; y < GI.sector_pad_width; y++)
				{
					for (int x = 0; x < GI.sector_pad_width; x++)
					{
						int s_ind = 2* getIndex(x,y,z,GI.sector_pad_width) ;
						ind = 2*(sector_ind_offset + getIndex(x,y,z,GI.width));
						//if (z==3)
						//	printf("%.4f ",sdata[sec][s_ind]);
						//TODO auslagern
						if (isOutlier(x,y,z,center_x,center_y,center_z,GI.width,GI.sector_offset))
							continue;
					
						gdata[ind] = sdata[s_ind]; //Re
						gdata[ind+1] = sdata[s_ind+1];//Im
					}
					//if (z==3) printf("\n");
				}
				//printf("----------------------------------------------------\n");
			//free(sdata[sec]);
		}
		//free(sdata);
	}//sec < sector_count
	
}

void runSimpleKernelCall()
{
	printf("starting gpu implementation\n");
	int test_h[3];
	test_h[0] = 1;
	test_h[1] = 2;
	test_h[2] = 3;	
	int* test_d;
	printf("input: %d , %d , %d\n",test_h[0],test_h[1],test_h[2]);
	
	//allocateDeviceMem<int>(&test_d,3);
	//copyToDevice(test_h,test_d,3);
	
	allocateAndCopyToDeviceMem<int>(&test_d,test_h,3);

	dim3 grid_size = 1;
	dim3 thread_size = 3;
	kernel_call<<<grid_size,thread_size>>>(test_d);
	
	copyFromDevice(test_d,test_h,3);

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

	float kernel_radius = static_cast<float>(kernel_width) / 2.0f;
	float radius = kernel_radius / static_cast<float>(width);
	float width_inv = 1.0f / width;
	float radiusSquared = radius * radius;
	float kernelRadius_invSqr = 1 / radiusSquared;
	float dist_multiplier = (kernel_count - 1) * kernelRadius_invSqr;
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

	printf("sector offset = %d",sector_offset);
	
	gi_host->sector_pad_width = sector_pad_width;
	

	cudaMemcpyToSymbol(GI, gi_host,sizeof(GriddingInfo));
	free(gi_host);
}

void gridding3D_gpu(float* data, 
					int data_cnt,
					float* crds, 
					float* gdata,
					int gdata_cnt,
					float* kernel,
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
	
	float* data_d, *crds_d, *gdata_d, *kernel_d;
	int* sector_centers_d, *sectors_d;

	allocateAndCopyToDeviceMem<float>(&data_d,data,2*data_cnt);
	allocateAndCopyToDeviceMem<float>(&crds_d,crds,3*data_cnt);
	allocateAndCopyToDeviceMem<float>(&gdata_d,gdata,gdata_cnt);//Konvention!!!
	allocateAndCopyToDeviceMem<float>(&kernel_d,kernel,kernel_cnt);
	allocateAndCopyToDeviceMem<int>(&sectors_d,sectors,2*sector_count);
	allocateAndCopyToDeviceMem<int>(&sector_centers_d,sector_centers,3*sector_count);
	
	griddingKernel<<<1,1>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d);

	copyFromDevice(gdata_d,gdata,gdata_cnt);

	freeDeviceMem(data_d);
	freeDeviceMem(crds_d);
	freeDeviceMem(gdata_d);
	freeDeviceMem(kernel_d);
	freeDeviceMem(sectors_d);
	freeDeviceMem(sector_centers_d);
}