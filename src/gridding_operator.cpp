
#include "gridding_operator.hpp"

#include "gridding_gpu.hpp"
#include "gridding_kernels.hpp"
#include "cufft_config.hpp"
#include "cuda_utils.hpp"

#include <iostream>

template <typename T>
T* GriddingND::GriddingOperator::selectOrdered(GriddingND::Array<T>& dataArray,int offset)
{
	T* dataSorted = (T*) calloc(dataArray.count(),sizeof(T)); //2* re + im

	for (IndType i=0; i<dataIndices.count();i++)
	{
		for (IndType chn=0; chn<dataArray.dim.channels; chn++)
		{
			dataSorted[i+chn*offset] = dataArray.data[dataIndices.data[i]+chn*offset];
		}
	}
	return dataSorted;
}

template <typename T>
void GriddingND::GriddingOperator::writeOrdered(GriddingND::Array<T>& destArray, T* sortedArray, int offset)
{
	for (IndType i=0; i<dataIndices.count();i++)
	{
		for (IndType chn=0; chn<destArray.dim.channels; chn++)
		{
			destArray.data[dataIndices.data[i]+chn*offset] = sortedArray[i+chn*offset];
		}
	}
}

void GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingND::Array<CufftType>& imgData, GriddingOutput griddingOut)
{
	if (DEBUG)
	{
		std::cout << "performing gridding adjoint!!!" << std::endl;
		std::cout << "test " << imgData.dim.width << std::endl;
		std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
		std::cout << "imgCount: " << imgData.count() << " gridWidth: " << this->getGridWidth() << std::endl;
		std::cout << "apply density comp: " << this->applyDensComp() << std::endl;
		std::cout << "kernel: " << this->kernel.data[3] << std::endl;
		std::cout << (this->dens.data == NULL) << std::endl; 
	}
	
	// select data ordered
	
	DType2* dataSorted = selectOrdered<DType2>(kspaceData,this->kSpaceTraj.count());
	DType* densSorted = NULL;
	if (this->applyDensComp())
		densSorted = this->dens.data;//selectOrdered<DType>(this->dens);

    //gridding3D_gpu_adj(dataSorted,this->kSpaceTraj.count(),kspaceData.dim.channels,this->kSpaceTraj.data,
	//	               &imgData.data,this->imgDims.count(),this->getGridWidth(),this->kernel.data,this->kernel.count(),
	//				   this->kernelWidth,this->sectorDataCount.data,this->sectorDims.count(),
	//				   (IndType*)this->sectorCenters.data,this->sectorWidth, imgData.dim.width,
	//				   this->osf,this->applyDensComp(),densSorted,griddingOut);

	DType2*		data				= dataSorted;
	int			data_count          = this->kSpaceTraj.count();
	int			n_coils             = kspaceData.dim.channels;
	DType*		crds                = this->kSpaceTraj.data;
	CufftType**	imdata              = &imgData.data;
	IndType		imdata_count        = this->imgDims.count();
	int			grid_width          = this->getGridWidth();
	DType*		kernel              = this->kernel.data;
	int			kernel_count        = this->kernel.count();
	int			kernel_width        = this->kernelWidth;
	IndType*	sectors             = this->sectorDataCount.data;
	int			sector_count        = this->sectorDims.count();
	IndType*	sector_centers      = (IndType*)this->sectorCenters.data;
	int			sector_width        = this->sectorWidth;
	int			im_width            = imgData.dim.width;
	DType		osr                 = this->osf;
	bool		do_comp             = this->applyDensComp();
	DType*  density_comp            = densSorted;
	const GriddingOutput gridding_out = griddingOut;

	assert(sectors != NULL);
	
	showMemoryInfo();

	//split and run sectors into blocks
	//and each data point to one thread inside this block 
	GriddingInfo* gi_host = initAndCopyGriddingInfo(sector_count,sector_width,kernel_width,kernel_count,grid_width,im_width,osr,data_count);
	
	DType2* data_d;
	DType* crds_d, *density_comp_d, *deapo_d;
	CufftType *gdata_d, *imdata_d;
	IndType* sector_centers_d, *sectors_d;
	
	if (DEBUG)
		printf("allocate and copy imdata of size %d...\n",imdata_count);
	allocateAndCopyToDeviceMem<CufftType>(&imdata_d,*imdata,imdata_count);//Konvention!!!

	if (DEBUG)
		printf("allocate and copy gdata of size %d...\n",gi_host->grid_width_dim);
	allocateDeviceMem<CufftType>(&gdata_d,gi_host->grid_width_dim);

	if (DEBUG)
		printf("allocate and copy data of size %d...\n",data_count);
	allocateDeviceMem<DType2>(&data_d,data_count);

	if (DEBUG)
		printf("allocate and copy coords of size %d...\n",3*data_count);
	allocateAndCopyToDeviceMem<DType>(&crds_d,crds,3*data_count);
	
	if (DEBUG)
		printf("allocate and copy kernel in const memory of size %d...\n",kernel_count);
	
	initConstSymbol("KERNEL",(void*)kernel,kernel_count*sizeof(DType));

	//allocateAndCopyToDeviceMem<DType>(&kernel_d,kernel,kernel_count);
	if (DEBUG)
		printf("allocate and copy sectors of size %d...\n",sector_count+1);
	allocateAndCopyToDeviceMem<IndType>(&sectors_d,sectors,sector_count+1);
	if (DEBUG)
		printf("allocate and copy sector_centers of size %d...\n",3*sector_count);
	allocateAndCopyToDeviceMem<IndType>(&sector_centers_d,sector_centers,3*sector_count);
	
	if (do_comp == true)	
	{
		if (DEBUG)
			printf("allocate and copy density compensation of size %d...\n",data_count);
		allocateAndCopyToDeviceMem<DType>(&density_comp_d,density_comp,data_count);
	}
	
	if (n_coils > 1)
	{
		if (DEBUG)
			printf("allocate precompute deapofunction of size %d...\n",imdata_count);
		allocateDeviceMem<DType>(&deapo_d,imdata_count);
		precomputeDeapodization(deapo_d,gi_host);
	}
	if (DEBUG)
		printf("sector pad width: %d\n",gi_host->sector_pad_width);
	
	//Inverse fft plan and execution
	cufftHandle fft_plan;
	if (DEBUG)
		printf("creating cufft plan with %d,%d,%d dimensions\n",gi_host->grid_width,gi_host->grid_width,gi_host->grid_width);
	cufftResult res = cufftPlan3d(&fft_plan, gi_host->grid_width,gi_host->grid_width,gi_host->grid_width, CufftTransformType) ;
	if (res != CUFFT_SUCCESS) 
		fprintf(stderr,"error on CUFFT Plan creation!!! %d\n",res);
	int err;

	//iterate over coils and compute result
	for (int coil_it = 0; coil_it < n_coils; coil_it++)
	{
		int data_coil_offset = coil_it * data_count;
		int im_coil_offset = coil_it * imdata_count;//gi_host->width_dim;

		cudaMemset(gdata_d,0, sizeof(CufftType)*gi_host->grid_width_dim);
		copyToDevice(data + data_coil_offset, data_d,data_count);
	
		if (do_comp == true)
			performDensityCompensation(data_d,density_comp_d,gi_host);
		
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));
		performConvolution(data_d,crds_d,gdata_d,NULL,sectors_d,sector_centers_d,gi_host);
	
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			fprintf(stderr,"error at adj  thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));
		if (gridding_out == CONVOLUTION)
		{
			if (DEBUG)
				printf("stopping output after CONVOLUTION step\n");
			//get output
			copyFromDevice<CufftType>(gdata_d,*imdata,gi_host->grid_width_dim);
			if (DEBUG)
				printf("test value at point zero: %f\n",(*imdata)[0].x);
			freeTotalDeviceMemory(data_d,crds_d,imdata_d,gdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop token

			free(gi_host);
			// Destroy the cuFFT plan.
			cufftDestroy(fft_plan);
			return;
		}
		if ((cudaThreadSynchronize() != cudaSuccess))
			fprintf(stderr,"error at adj thread synchronization 3: %s\n",cudaGetErrorString(cudaGetLastError()));
		performFFTShift(gdata_d,INVERSE,gi_host->grid_width);
		
		//Inverse FFT
		if (err=pt2CufftExec(fft_plan, gdata_d, gdata_d, CUFFT_INVERSE) != CUFFT_SUCCESS)
		{
			fprintf(stderr,"cufft has failed at adj with err %i \n",err);
			showMemoryInfo(true,stderr);
		}
	  	if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			 fprintf(stderr,"error at adj thread synchronization 4: %s\n",cudaGetErrorString(cudaGetLastError()));
	
		if (gridding_out == FFT)
		{
			if (DEBUG)
				printf("stopping output after FFT step\n");
			//get output
			copyFromDevice<CufftType>(gdata_d,*imdata,gi_host->grid_width_dim);
			
			//free memory
			if (cufftDestroy(fft_plan) != CUFFT_SUCCESS)
				printf("error on destroying cufft plan\n");
			freeTotalDeviceMemory(data_d,crds_d,imdata_d,gdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop token
			free(gi_host);
			// Destroy the cuFFT plan.
			printf("last cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
			return;
		}
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 5: %s\n",cudaGetErrorString(cudaGetLastError()));
		performFFTShift(gdata_d,INVERSE,gi_host->grid_width);
			
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 6: %s\n",cudaGetErrorString(cudaGetLastError()));
		performCrop(gdata_d,imdata_d,gi_host);
		
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 7: %s\n",cudaGetErrorString(cudaGetLastError()));
		//check if precomputed deapo function can be used
		if (n_coils > 1 && deapo_d != NULL)
			performDeapodization(imdata_d,deapo_d,gi_host);
		else
		  performDeapodization(imdata_d,gi_host);
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 8: %s\n",cudaGetErrorString(cudaGetLastError()));
	
		performFFTScaling(imdata_d,gi_host->im_width_dim,gi_host);
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error: at adj  thread synchronization 9: %s\n",cudaGetErrorString(cudaGetLastError()));
	
		//get result
		copyFromDevice<CufftType>(imdata_d,*imdata+im_coil_offset,imdata_count);
	}//iterate over coils
	if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error: at adj  thread synchronization 10: %s\n",cudaGetErrorString(cudaGetLastError()));
	// Destroy the cuFFT plan.
	cufftDestroy(fft_plan);
	freeTotalDeviceMemory(data_d,crds_d,gdata_d,imdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop
	if (do_comp == true)
		cudaFree(density_comp_d);
	if (n_coils > 1)
		cudaFree(deapo_d);
	
	if ((cudaThreadSynchronize() != cudaSuccess))
		fprintf(stderr,"error in gridding3D_gpu_adj function: %s\n",cudaGetErrorString(cudaGetLastError()));
	free(gi_host);
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData, GriddingOutput griddingOut)
{
	// init result
	GriddingND::Array<CufftType> imgData;
	imgData.data = (CufftType*)calloc(imgDims.count(),sizeof(CufftType));
	imgData.dim = this->getImageDims();

	performGriddingAdj(kspaceData,imgData,griddingOut);

	return imgData;
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData)
{
	return performGriddingAdj(kspaceData,DEAPODIZATION);
}

void GriddingND::GriddingOperator::performForwardGridding(GriddingND::Array<DType2> imgData,GriddingND::Array<CufftType>& kspaceData, GriddingOutput griddingOut)
{
	if (DEBUG)
	{
		std::cout << "performing forward gridding!!!" << std::endl;
		std::cout << "test " << this->kSpaceTraj.dim.width << std::endl;
		std::cout << "dataCount: " << kspaceData.count() << " chnCount: " << kspaceData.dim.channels << std::endl;
		std::cout << "imgCount: " << imgData.count() << " gridWidth: " << this->getGridWidth() << std::endl;
	}

	CufftType* kspaceDataSorted = (CufftType*) calloc(kspaceData.count(),sizeof(CufftType));

	gridding3D_gpu(&kspaceDataSorted,this->kSpaceTraj.count(),kspaceData.dim.channels,this->kSpaceTraj.data,
		           imgData.data,this->imgDims.count(),this->getGridWidth(),this->kernel.data,this->kernel.count(),
				   this->kernelWidth,this->sectorDataCount.data,this->sectorDims.count(),
				   (IndType*)this->sectorCenters.data,this->sectorWidth, imgData.dim.width,this->osf,griddingOut);

	writeOrdered<CufftType>(kspaceData,kspaceDataSorted,this->kSpaceTraj.count());

	free(kspaceDataSorted);
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performForwardGridding(Array<DType2> imgData,GriddingOutput griddingOut)
{
	GriddingND::Array<CufftType> kspaceData;
	kspaceData.data = (CufftType*)calloc(this->kSpaceTraj.count()*imgData.dim.channels,sizeof(CufftType));
	kspaceData.dim = this->kSpaceTraj.dim;
	kspaceData.dim.channels = imgData.dim.channels;

	performForwardGridding(imgData,kspaceData,griddingOut);

	return kspaceData;
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performForwardGridding(Array<DType2> imgData)
{
	return performForwardGridding(imgData,CONVOLUTION);
}