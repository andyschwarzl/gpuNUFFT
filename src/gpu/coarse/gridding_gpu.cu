#include "gridding_kernels.cu"
#include "cuda_utils.hpp"

//TODO forward gridding from grid to k-space

//adjoint gridding from k-space to grid
void gridding3D_gpu(DType*		data,			//kspace data array 
					int			data_count,		//data count, samples per trajectory
					int			n_coils,		//number of coils 
					DType*		crds,			//
					CufftType*	gdata,			//
					int			gdata_count,	//			
					int			grid_width,		//
					DType*		kernel,			//
					int			kernel_count,	//
					int			kernel_width,	//
					int*		sectors,		//
					int			sector_count,	//
					int*		sector_centers,	//
					int			sector_width,	//
					int			im_width,		//
					DType		osr,			//
					const GriddingOutput gridding_out)
{
	assert(sectors != NULL);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
    
	//split and run sectors into blocks
	//and each data point to one thread inside this block 
	GriddingInfo* gi_host = initAndCopyGriddingInfo(sector_count,sector_width,kernel_width,kernel_count,grid_width,osr);
	
	DType* data_d, *crds_d, *kernel_d, *temp_gdata_d;
	CufftType *gdata_d;
	int* sector_centers_d, *sectors_d;

	printf("allocate and copy gdata of size %d...\n",gdata_count);
	allocateAndCopyToDeviceMem<CufftType>(&gdata_d,gdata,gdata_count);//Konvention!!!

	printf("allocate and copy data of size %d...\n",2*data_count*n_coils);
	allocateAndCopyToDeviceMem<DType>(&data_d,data,2*data_count*n_coils);

	int temp_grid_count = 2 * sector_count * gi_host->sector_dim;
	printf("allocate temp grid data of size %d...\n",temp_grid_count);
	allocateAndSetMem<DType>(&temp_gdata_d,temp_grid_count,0);
	
	printf("allocate and copy coords of size %d...\n",3*data_count);
	allocateAndCopyToDeviceMem<DType>(&crds_d,crds,3*data_count);
	
	printf("allocate and copy kernel of size %d...\n",kernel_count);
	allocateAndCopyToDeviceMem<DType>(&kernel_d,kernel,kernel_count);
	printf("allocate and copy sectors of size %d...\n",sector_count+1);
	allocateAndCopyToDeviceMem<int>(&sectors_d,sectors,sector_count+1);
	printf("allocate and copy sector_centers of size %d...\n",3*sector_count);
	allocateAndCopyToDeviceMem<int>(&sector_centers_d,sector_centers,3*sector_count);
	printf("sector pad width: %d\n",gi_host->sector_pad_width);
	
	//Inverse fft plan and execution
	cufftHandle fft_plan;
	printf("creating cufft plan with %d,%d,%d dimensions\n",gi_host->grid_width,gi_host->grid_width,gi_host->grid_width);
	cufftResult res = cufftPlan3d(&fft_plan, gi_host->grid_width,gi_host->grid_width,gi_host->grid_width, CUFFT_C2C) ;
	if (res != CUFFT_SUCCESS) 
		printf("error on CUFFT Plan creation!!! %d\n",res);
	int err;

	//iterate over coils and compute result
	for (int coil_it = 0; coil_it < n_coils; coil_it++)
	{
		int data_coil_offset = 2 * coil_it * data_count;
		int grid_coil_offset = coil_it * gdata_count;//gi_host->width_dim;
		//reset temp array
		cudaMemset(temp_gdata_d,0, sizeof(DType)*temp_grid_count);
		cudaMemset(gdata_d,0, sizeof(CufftType)*gdata_count);
		
		performConvolution(data_d+data_coil_offset,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,gi_host);

		//compose total output from local blocks 
		composeOutput(temp_gdata_d,gdata_d,sector_centers_d,gi_host);
	
		if (gridding_out == CONVOLUTION)
		{
			printf("stopping output after CONVOLUTION step\n");
			//get output
			copyFromDevice<CufftType>(gdata_d,gdata,gdata_count);
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
			copyFromDevice<CufftType>(gdata_d,gdata,gdata_count);
			//free memory
			freeTotalDeviceMemory(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,temp_gdata_d,NULL);//NULL as stop token
			free(gi_host);
			/* Destroy the cuFFT plan. */
			cufftDestroy(fft_plan);
			return;
		}

		performFFTShift(gdata_d,INVERSE,gi_host->grid_width);

		//TODO crop

		performDeapodization(gdata_d,gi_host);

		//get result
		copyFromDevice<CufftType>(gdata_d,gdata+grid_coil_offset,gdata_count);
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
