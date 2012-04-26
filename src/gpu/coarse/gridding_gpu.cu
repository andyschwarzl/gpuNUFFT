#include "gridding_kernels.cuh"
#include "cuda_utils.hpp"

//TODO inverse gridding from grid to k-space

//grid from k-space to grid
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
					DType osr,
					const GriddingOutput gridding_out)
{
	assert(sectors != NULL);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
    
	//split and run sectors into blocks
	//and each data point to one thread inside this block 
	GriddingInfo* gi_host = initAndCopyGriddingInfo(sector_count,sector_width,kernel_width,kernel_count,width,osr);
	
	DType* data_d, *crds_d, *kernel_d, *temp_gdata_d;
	CufftType *gdata_d;
	int* sector_centers_d, *sectors_d;

	printf("allocate and copy gdata of size %d...\n",gdata_cnt);
	allocateAndCopyToDeviceMem<CufftType>(&gdata_d,gdata,gdata_cnt);//Konvention!!!

	printf("allocate and copy data of size %d...\n",2*data_cnt);
	allocateAndCopyToDeviceMem<DType>(&data_d,data,2*data_cnt);

	int temp_grid_cnt = 2 * sector_count * gi_host->sector_dim;
	printf("allocate temp grid data of size %d...\n",temp_grid_cnt);
	allocateAndSetMem<DType>(&temp_gdata_d,temp_grid_cnt,0);
	
	printf("allocate and copy coords of size %d...\n",3*data_cnt);
	allocateAndCopyToDeviceMem<DType>(&crds_d,crds,3*data_cnt);
	
	printf("allocate and copy kernel of size %d...\n",kernel_cnt);
	allocateAndCopyToDeviceMem<DType>(&kernel_d,kernel,kernel_cnt);
	printf("allocate and copy sectors of size %d...\n",sector_count+1);
	allocateAndCopyToDeviceMem<int>(&sectors_d,sectors,sector_count+1);
	printf("allocate and copy sector_centers of size %d...\n",3*sector_count);
	allocateAndCopyToDeviceMem<int>(&sector_centers_d,sector_centers,3*sector_count);
	printf("sector pad width: %d\n",gi_host->sector_pad_width);
	dim3 block_dim(gi_host->sector_pad_width,gi_host->sector_pad_width,N_THREADS_PER_SECTOR);
	
	griddingKernel<<<sector_count,block_dim>>>(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d);

	//compose total output from local blocks 
	composeOutput<<<1,block_dim>>>(temp_gdata_d,gdata_d,sector_centers_d);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed,start,stop);
	printf("Time elapsed: %3.1fms\n",elapsed);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (gridding_out == CONVOLUTION)
	{
		printf("stopping output after CONVOLUTION step\n");
		//get output
		copyFromDevice<CufftType>(gdata_d,gdata,gdata_cnt);
		freeTotalDeviceMemory(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,NULL);//NULL as stop token
		free(gi_host);
		return;
	}

	//Inverse fft plan and execution
	cufftHandle fft_plan;
	printf("creating cufft plan with %d,%d,%d dimensions\n",gi_host->width,gi_host->width,gi_host->width);
	cufftResult res = cufftPlan3d(&fft_plan, gi_host->width,gi_host->width,gi_host->width, CUFFT_C2C) ;

	if (res != CUFFT_SUCCESS) 
		printf("error on CUFFT Plan creation!!! %d\n",res);
	int err;
	
	if (err=cufftExecC2C(fft_plan, gdata_d, gdata_d, CUFFT_INVERSE) != CUFFT_SUCCESS)
	{
	      printf("cufft has failed with err %i \n",err);
      //return;
	}
	/* Destroy the cuFFT plan. */
	cufftDestroy(fft_plan);

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

	//get result
	copyFromDevice<CufftType>(gdata_d,gdata,gdata_cnt);
	
	freeTotalDeviceMemory(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,NULL);//NULL as stop
	free(gi_host);
}
