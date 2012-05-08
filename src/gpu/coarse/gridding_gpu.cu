#include "gridding_kernels.cu"
#include "cuda_utils.hpp"

//TODO inverse gridding from grid to k-space

//grid from k-space to grid
void gridding3D_gpu(DType* data, 
					int data_cnt,
					int n_coils,
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

	printf("allocate and copy data of size %d...\n",2*data_cnt*n_coils);
	allocateAndCopyToDeviceMem<DType>(&data_d,data,2*data_cnt*n_coils);

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
	
	//Inverse fft plan and execution
	cufftHandle fft_plan;
	printf("creating cufft plan with %d,%d,%d dimensions\n",gi_host->width,gi_host->width,gi_host->width);
	cufftResult res = cufftPlan3d(&fft_plan, gi_host->width,gi_host->width,gi_host->width, CUFFT_C2C) ;
	if (res != CUFFT_SUCCESS) 
		printf("error on CUFFT Plan creation!!! %d\n",res);
	int err;

	//iterate over coils and compute result
	for (int coil_it = 0; coil_it < n_coils; coil_it++)
	{
		int data_coil_offset = 2 * coil_it * data_cnt;
		int grid_coil_offset = coil_it * gdata_cnt;//gi_host->width_dim;
		//reset temp array
		cudaMemset(temp_gdata_d,0, sizeof(DType)*temp_grid_cnt);
		cudaMemset(gdata_d,0, sizeof(CufftType)*gdata_cnt);
		
		performConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,sector_count,block_dim,gi_host);

		//compose total output from local blocks 
		composeOutput(temp_gdata_d,gdata_d,sector_centers_d,1,block_dim);
	
		if (gridding_out == CONVOLUTION)
		{
			printf("stopping output after CONVOLUTION step\n");
			//get output
			copyFromDevice<CufftType>(gdata_d,gdata,gdata_cnt);
			printf("test value at point zero: %f\n",gdata[0].x);
			freeTotalDeviceMemory(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,NULL);//NULL as stop token
			free(gi_host);
			/* Destroy the cuFFT plan. */
			cufftDestroy(fft_plan);
			return;
		}

		//Inverse FFT
		if (err=cufftExecC2C(fft_plan, gdata_d, gdata_d, CUFFT_INVERSE) != CUFFT_SUCCESS)
		{
			  printf("cufft has failed with err %i \n",err);
		  //return;
		}
	
		if (gridding_out == FFT)
		{
			printf("stopping output after FFT step\n");
			//get output
			copyFromDevice<CufftType>(gdata_d,gdata,gdata_cnt);
			//free memory
			freeTotalDeviceMemory(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,NULL);//NULL as stop token
			free(gi_host);
			/* Destroy the cuFFT plan. */
			cufftDestroy(fft_plan);
			return;
		}

		performFFTShift(gdata_d,INVERSE,gi_host->width);

		dim3 block_dim_deapo(gi_host->width,gi_host->width,1);	
		performDeapodization(gdata_d,block_dim_deapo,gi_host->width,gi_host);

		//get result
		copyFromDevice<CufftType>(gdata_d,gdata+grid_coil_offset,gdata_cnt);
	}//iterate over coils

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed,start,stop);
	printf("Time elapsed: %3.1fms\n",elapsed);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* Destroy the cuFFT plan. */
	cufftDestroy(fft_plan);
	freeTotalDeviceMemory(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,NULL);//NULL as stop
	free(gi_host);
}
