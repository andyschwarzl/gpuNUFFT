
#include "gridding_operator.hpp"
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

void GriddingND::GriddingOperator::initKernel()
{
  IndType kernelSize = calculateGrid3KernelSize(osf, kernelWidth/2.0f);
  this->kernel.dim.length = kernelSize;
  this->kernel.data = (DType*) calloc(this->kernel.count(),sizeof(DType));
  load1DKernel(this->kernel.data,(int)kernelSize,(int)kernelWidth,osf);
}

GriddingND::GriddingInfo* GriddingND::GriddingOperator::initGriddingInfo()
{
  GriddingND::GriddingInfo* gi_host = (GriddingND::GriddingInfo*)malloc(sizeof(GriddingND::GriddingInfo));

  gi_host->data_count = (int)this->kSpaceTraj.count();
  gi_host->sector_count = (int)this->gridSectorDims.count();
  gi_host->sector_width = (int)sectorDims.width;

  gi_host->kernel_width = (int)this->kernelWidth; 
  gi_host->kernel_widthSquared = (int)(this->kernelWidth * this->kernelWidth);
  gi_host->kernel_count = (int)this->kernel.count();

  gi_host->grid_width_dim = (int)this->getGridDims().count();
  gi_host->grid_width_offset= (int)(floor(this->getGridDims().width / (DType)2.0));

  gi_host->im_width_dim = (int)imgDims.count();
  gi_host->im_width_offset.x = (int)(floor(imgDims.width / (DType)2.0));
  gi_host->im_width_offset.y = (int)(floor(imgDims.height / (DType)2.0));
  gi_host->im_width_offset.z = (int)(floor(imgDims.depth / (DType)2.0));

  gi_host->imgDims.x = imgDims.width;
  gi_host->imgDims.y = imgDims.height;
  gi_host->imgDims.z = imgDims.depth;
  gi_host->imgDims_count = imgDims.width*imgDims.height*DEFAULT_VALUE(imgDims.depth);//TODO check why not imgDims.count()

  gi_host->gridDims.x = this->getGridDims().width;
  gi_host->gridDims.y = this->getGridDims().height;
  gi_host->gridDims.z = this->getGridDims().depth;
  gi_host->gridDims_count = this->getGridDims().width*this->getGridDims().height*DEFAULT_VALUE(this->getGridDims().depth);//s.a.

  double kernel_radius = static_cast<double>(this->kernelWidth) / 2.0;
  double radius = kernel_radius / static_cast<double>(this->getGridDims().width);

  DType kernel_width_inv = (DType)1.0 / static_cast<DType>(this->kernelWidth);

  double radiusSquared = radius * radius;
  double kernelRadius_invSqr = 1.0 / radiusSquared;
  DType dist_multiplier = (DType)((this->kernel.count() - 1) * kernelRadius_invSqr);

  if (DEBUG)
    printf("radius rel. to grid width %f\n",radius);

  GriddingND::Dimensions sectorPadDims = sectorDims + 2*(int)(floor(this->kernelWidth / (DType)2.0));

  int sector_pad_width = (int)sectorPadDims.width;
  int sector_dim = (int)sectorPadDims.count();
  int sector_offset = (int)(floor(sector_pad_width / (DType)2.0));

  gi_host->grid_width_inv.x = (DType)1.0 / static_cast<DType>(this->getGridDims().width);
  gi_host->grid_width_inv.y = (DType)1.0 / static_cast<DType>(this->getGridDims().height);
  gi_host->grid_width_inv.z = (DType)1.0 / static_cast<DType>(this->getGridDims().depth);
  gi_host->kernel_widthInvSquared = kernel_width_inv * kernel_width_inv;
  gi_host->osr = this->osf;

  gi_host->kernel_radius = (DType)kernel_radius;
  gi_host->sector_pad_width = sector_pad_width;
  gi_host->sector_pad_max = sector_pad_width - 1;
  gi_host->sector_dim = sector_dim;
  gi_host->sector_offset = sector_offset;

  gi_host->aniso_z_scale = ((DType)this->getGridDims().depth/(DType)this->getGridDims().width);

  double sectorKdim = 1.0 / gridSectorDims.width;
  gi_host->aniso_z_shift = (DType)(((gridSectorDims.width-gridSectorDims.depth)/2.0)*sectorKdim);

  gi_host->radiusSquared = (DType)radiusSquared;
  gi_host->radiusSquared_inv = (DType)kernelRadius_invSqr;
  gi_host->dist_multiplier = dist_multiplier;

  gi_host->is2Dprocessing = this->is2DProcessing();
  return gi_host;
}

GriddingND::GriddingInfo* GriddingND::GriddingOperator::initAndCopyGriddingInfo()
{
  GriddingInfo* gi_host = initGriddingInfo();
    if (DEBUG)
    printf("copy Gridding Info to symbol memory... size = %ld \n",sizeof(GriddingND::GriddingInfo));

  initConstSymbol("GI",gi_host,sizeof(GriddingND::GriddingInfo));

  //free(gi_host);
  if (DEBUG)
    printf("...done!\n");
  return gi_host;
}

void GriddingND::GriddingOperator::adjConvolution(DType2* data_d, 
      DType* crds_d, 
      CufftType* gdata_d,
      DType* kernel_d, 
      IndType* sectors_d, 
      IndType* sector_centers_d,
  GriddingND::GriddingInfo* gi_host)
{
  performConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,gi_host);
}

void GriddingND::GriddingOperator::forwardConvolution(CufftType*		data_d, 
  DType*			crds_d, 
  CufftType*		gdata_d,
  DType*			kernel_d, 
  IndType*		sectors_d, 
  IndType*		sector_centers_d,
  GriddingND::GriddingInfo* gi_host)
{
  performForwardConvolution(data_d,crds_d,gdata_d,kernel_d,sectors_d,sector_centers_d,gi_host);
}

  void GriddingND::GriddingOperator::initLookupTable()
  {
    initConstSymbol("KERNEL",(void*)this->kernel.data,this->kernel.count()*sizeof(DType));
  }
  
  void GriddingND::GriddingOperator::freeLookupTable()
  {
    
  }

// ----------------------------------------------------------------------------
// performGriddingAdj: NUFFT^H
//
// Gridding implementation - interpolation from nonuniform k-space data onto 
//                           oversampled grid based on optimized gridding kernel
//                           with minimal oversampling ratio (see Beatty et al.)
//
// Basic steps: - density compensation
//              - convolution with interpolation function
//              - iFFT
//              - cropping due to oversampling ratio
//              - apodization correction
//
// parameters:
//	* data		     : input kspace data 
//  * data_count   : number of samples on trajectory
//  * n_coils      : number of channels or coils
//  * crds         : coordinate array on trajectory
//  * imdata       : output image data
//  * imdata_count : number of image data points
//  * grid_width   : size of grid 
//  * kernel       : precomputed convolution kernel as lookup table
//  * kernel_count : number of kernel lookup table entries
//  * sectors      : mapping of start and end points of each sector
//  * sector_count : number of sectors
//  * sector_centers: coordinates (x,y,z) of sector centers
//  * sector_width : width of sector 
//  * im_width     : dimension of image
//  * osr          : oversampling ratio
//  * do_comp      : true, if density compensation has to be done
//  * density_comp : densiy compensation array
//  * gridding_out : enum indicating how far gridding has to be processed
//  
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
  DType2* dataSorted = selectOrdered<DType2>(kspaceData,(int)this->kSpaceTraj.count());
  DType* densSorted = NULL;
  if (this->applyDensComp())
    densSorted = this->dens.data;//selectOrdered<DType>(this->dens);

  int			data_count          = (int)this->kSpaceTraj.count();
  int			n_coils             = (int)kspaceData.dim.channels;
  IndType		imdata_count        = this->imgDims.count();
  int			sector_count        = (int)this->gridSectorDims.count();

  DType*  density_comp            = densSorted;

  showMemoryInfo();

  //split and run sectors into blocks
  //and each data point to one thread inside this block 
  GriddingInfo* gi_host = initAndCopyGriddingInfo();//

  DType2* data_d;
  DType* crds_d, *density_comp_d, *deapo_d;
  CufftType *gdata_d, *imdata_d;
  IndType* sector_centers_d, *sectors_d;

  if (DEBUG)
    printf("allocate and copy imdata of size %d...\n",imdata_count);
  allocateAndCopyToDeviceMem<CufftType>(&imdata_d,imgData.data,imdata_count);//Konvention!!!

  if (DEBUG)
    printf("allocate and copy gdata of size %d...\n",gi_host->grid_width_dim);
  allocateDeviceMem<CufftType>(&gdata_d,gi_host->grid_width_dim);

  if (DEBUG)
    printf("allocate and copy data of size %d...\n",data_count);
  allocateDeviceMem<DType2>(&data_d,data_count);

  if (DEBUG)
    printf("allocate and copy coords of size %d...\n",getImageDimensionCount()*data_count);
  allocateAndCopyToDeviceMem<DType>(&crds_d,this->kSpaceTraj.data,getImageDimensionCount()*data_count);

  if (DEBUG)
    printf("allocate and copy kernel in const memory of size %d...\n",this->kernel.count());

  initLookupTable();

  //allocateAndCopyToDeviceMem<DType>(&kernel_d,kernel,kernel_count);
  if (DEBUG)
    printf("allocate and copy sectors of size %d...\n",sector_count+1);
  allocateAndCopyToDeviceMem<IndType>(&sectors_d,this->sectorDataCount.data,sector_count+1);
  if (DEBUG)
    printf("allocate and copy sector_centers of size %d...\n",getImageDimensionCount()*sector_count);
  allocateAndCopyToDeviceMem<IndType>(&sector_centers_d,(IndType*)this->getSectorCentersData(),getImageDimensionCount()*sector_count);

  if (this->applyDensComp())	
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
    printf("creating cufft plan with %d,%d,%d dimensions\n",DEFAULT_VALUE(gi_host->gridDims.z),gi_host->gridDims.y,gi_host->gridDims.x);
  cufftResult res = cufftPlan3d(&fft_plan, (int)DEFAULT_VALUE(gi_host->gridDims.z),(int)gi_host->gridDims.y,(int)gi_host->gridDims.x, CufftTransformType) ;
  if (res != CUFFT_SUCCESS) 
    fprintf(stderr,"error on CUFFT Plan creation!!! %d\n",res);
  int err;

  //iterate over coils and compute result
  for (int coil_it = 0; coil_it < n_coils; coil_it++)
  {
    int data_coil_offset = coil_it * data_count;
    int im_coil_offset = coil_it * (int)imdata_count;//gi_host->width_dim;

    cudaMemset(gdata_d,0, sizeof(CufftType)*gi_host->grid_width_dim);
    copyToDevice(dataSorted + data_coil_offset, data_d,data_count);

    if (this->applyDensComp())
      performDensityCompensation(data_d,density_comp_d,gi_host);

    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at adj thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));
    adjConvolution(data_d,crds_d,gdata_d,NULL,sectors_d,sector_centers_d,gi_host);

    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      fprintf(stderr,"error at adj  thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));
    if (griddingOut == CONVOLUTION)
    {
      if (DEBUG)
        printf("stopping output after CONVOLUTION step\n");
      //get output
      copyFromDevice<CufftType>(gdata_d,imgData.data,gi_host->grid_width_dim);
      if (DEBUG)
        printf("test value at point zero: %f\n",(imgData.data)[0].x);

      free(gi_host);
      // Destroy the cuFFT plan.
      cufftDestroy(fft_plan);
      freeLookupTable();
      cudaThreadSynchronize();
      freeTotalDeviceMemory(data_d,crds_d,imdata_d,gdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop token
      cudaThreadSynchronize();

      showMemoryInfo();
      return;
    }
    if ((cudaThreadSynchronize() != cudaSuccess))
      fprintf(stderr,"error at adj thread synchronization 3: %s\n",cudaGetErrorString(cudaGetLastError()));
    performFFTShift(gdata_d,INVERSE,getGridDims(),gi_host);

    //Inverse FFT
    if (err=pt2CufftExec(fft_plan, gdata_d, gdata_d, CUFFT_INVERSE) != CUFFT_SUCCESS)
    {
      fprintf(stderr,"cufft has failed at adj with err %i \n",err);
      showMemoryInfo(true,stderr);
    }
    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      fprintf(stderr,"error at adj thread synchronization 4: %s\n",cudaGetErrorString(cudaGetLastError()));

    if (griddingOut == FFT)
    {
      if (DEBUG)
        printf("stopping output after FFT step\n");
      //get output
      copyFromDevice<CufftType>(gdata_d,imgData.data,gi_host->grid_width_dim);

      //free memory
      if (cufftDestroy(fft_plan) != CUFFT_SUCCESS)
        printf("error on destroying cufft plan\n");

      free(gi_host);
      // Destroy the cuFFT plan.
      cufftDestroy(fft_plan);
      freeLookupTable();
      cudaThreadSynchronize();
      freeTotalDeviceMemory(data_d,crds_d,imdata_d,gdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop token
      cudaThreadSynchronize();
      printf("last cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
      return;
    }
    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at adj thread synchronization 5: %s\n",cudaGetErrorString(cudaGetLastError()));
    performFFTShift(gdata_d,INVERSE,getGridDims(),gi_host);

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
    copyFromDevice<CufftType>(imdata_d,imgData.data+im_coil_offset,imdata_count);
  }//iterate over coils
  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error: at adj  thread synchronization 10: %s\n",cudaGetErrorString(cudaGetLastError()));
  // Destroy the cuFFT plan.
  cufftDestroy(fft_plan);

  freeLookupTable();
  freeTotalDeviceMemory(data_d,crds_d,gdata_d,imdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop

  if (this->applyDensComp())
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

  if (griddingOut == GriddingND::CONVOLUTION)
  {
    imgData.dim = this->getGridDims();
    imgData.dim.channels = kspaceData.dim.channels;
    imgData.data = (CufftType*)calloc(imgData.count(),sizeof(CufftType));
  }
  else
  {
    imgData.dim = this->getImageDims();
    imgData.dim.channels = kspaceData.dim.channels;
    imgData.data = (CufftType*)calloc(imgData.count(),sizeof(CufftType));
  }
  performGriddingAdj(kspaceData,imgData,griddingOut);

  return imgData;
}

GriddingND::Array<CufftType> GriddingND::GriddingOperator::performGriddingAdj(GriddingND::Array<DType2> kspaceData)
{
  return performGriddingAdj(kspaceData,DEAPODIZATION);
}

// ----------------------------------------------------------------------------
// gridding3D_gpu: NUFFT
//
// Inverse gridding implementation - interpolation from uniform grid data onto 
//                                   nonuniform k-space data based on optimized 
//                                   gridding kernel with minimal oversampling
//                                   ratio (see Beatty et al.)
//
// Basic steps: - apodization correction
//              - zero padding with osf
//              - FFT
//							- convolution and resampling
//
// parameters:
//	* data		     : output kspace data 
//  * data_count   : number of samples on trajectory
//  * n_coils      : number of channels or coils
//  * crds         : coordinates on trajectory, passed as SoA
//  * imdata       : input image data
//  * imdata_count : number of image data points
//  * grid_width   : size of grid 
//  * kernel       : precomputed convolution kernel as lookup table
//  * kernel_count : number of kernel lookup table entries
//  * sectors      : mapping of data indices according to each sector
//  * sector_count : number of sectors
//  * sector_centers: coordinates (x,y,z) of sector centers
//  * sector_width : width of sector 
//  * im_width     : dimension of image
//  * osr          : oversampling ratio
//  * gridding_out : enum indicating how far gridding has to be processed
//  
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

  showMemoryInfo();

  int			data_count          = (int)this->kSpaceTraj.count();
  int			n_coils             = (int)kspaceData.dim.channels;
  IndType		imdata_count        = this->imgDims.count();
  int			sector_count        = (int)this->gridSectorDims.count();

  GriddingInfo* gi_host = initAndCopyGriddingInfo();

  //cuda mem allocation
  DType2 *imdata_d;
  DType *crds_d, *deapo_d;
  CufftType *gdata_d, *data_d;
  IndType* sector_centers_d, *sectors_d;

  if (DEBUG)
    printf("allocate and copy imdata of size %d...\n",imdata_count);
  allocateDeviceMem<DType2>(&imdata_d,imdata_count);

  if (DEBUG)
    printf("allocate and copy gdata of size %d...\n",gi_host->grid_width_dim );
  allocateAndSetMem<CufftType>(&gdata_d, gi_host->grid_width_dim,0);

  if (DEBUG)
    printf("allocate and copy data of size %d...\n",data_count);
  allocateDeviceMem<CufftType>(&data_d,data_count);

  if (DEBUG)
    printf("allocate and copy coords of size %d...\n",getImageDimensionCount()*data_count);
  allocateAndCopyToDeviceMem<DType>(&crds_d,this->kSpaceTraj.data,getImageDimensionCount()*data_count);

  if (DEBUG)
    printf("allocate and copy kernel in const memory of size %d...\n",this->kernel.count());

  initLookupTable();

  if (DEBUG)
    printf("allocate and copy sectors of size %d...\n",sector_count+1);
  allocateAndCopyToDeviceMem<IndType>(&sectors_d,this->sectorDataCount.data,sector_count+1);
  if (DEBUG)
    printf("allocate and copy sector_centers of size %d...\n",getImageDimensionCount()*sector_count);
  allocateAndCopyToDeviceMem<IndType>(&sector_centers_d,(IndType*)this->getSectorCentersData(),getImageDimensionCount()*sector_count);
  if (n_coils > 1)
  {
    if (DEBUG)
      printf("allocate and precompute deapofunction of size %d...\n",imdata_count);
    allocateDeviceMem<DType>(&deapo_d,imdata_count);
    precomputeDeapodization(deapo_d,gi_host);
  }

  if (DEBUG)
    printf("sector pad width: %d\n",gi_host->sector_pad_width);

  //Inverse fft plan and execution
  cufftHandle fft_plan;
  if (DEBUG)
    printf("creating cufft plan with %d,%d,%d dimensions\n",DEFAULT_VALUE(gi_host->gridDims.z),gi_host->gridDims.y,gi_host->gridDims.x);
  cufftResult res = cufftPlan3d(&fft_plan, (int)DEFAULT_VALUE(gi_host->gridDims.z),(int)gi_host->gridDims.y,(int)gi_host->gridDims.x, CufftTransformType) ;

  if (res != CUFFT_SUCCESS) 
    printf("error on CUFFT Plan creation!!! %d\n",res);
  int err;

  //iterate over coils and compute result
  for (int coil_it = 0; coil_it < n_coils; coil_it++)
  {
    int data_coil_offset = coil_it * data_count;
    int im_coil_offset = coil_it * (int)imdata_count;//gi_host->width_dim;
    //reset temp array
    copyToDevice(imgData.data + im_coil_offset,imdata_d,imdata_count);	
    cudaMemset(data_d,0, sizeof(CufftType)*data_count);
    cudaMemset(gdata_d,0, sizeof(CufftType)*gi_host->grid_width_dim);

    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));
    // apodization Correction
    if (n_coils > 1 && deapo_d != NULL)
      performForwardDeapodization(imdata_d,deapo_d,gi_host);
    else
      performForwardDeapodization(imdata_d,gi_host);

    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));
    // resize by oversampling factor and zero pad
    performPadding(imdata_d,gdata_d,gi_host);

    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at thread synchronization 3: %s\n",cudaGetErrorString(cudaGetLastError()));
    // shift image to get correct zero frequency position
    performFFTShift(gdata_d,INVERSE,getGridDims(),gi_host);

    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at thread synchronization 4: %s\n",cudaGetErrorString(cudaGetLastError()));
    // eventually free imdata_d
    // Forward FFT to kspace domain
    if (err=pt2CufftExec(fft_plan, gdata_d, gdata_d, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
      fprintf(stderr,"cufft has failed with err %i \n",err);
      showMemoryInfo(true,stderr);
    }

    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at thread synchronization 5: %s\n",cudaGetErrorString(cudaGetLastError()));
    performFFTShift(gdata_d,FORWARD,getGridDims(),gi_host);

    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at thread synchronization 6: %s\n",cudaGetErrorString(cudaGetLastError()));
    // convolution and resampling to non-standard trajectory
    forwardConvolution(data_d,crds_d,gdata_d,NULL,sectors_d,sector_centers_d,gi_host);
    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error at thread synchronization 7: %s\n",cudaGetErrorString(cudaGetLastError()));

    performFFTScaling(data_d,gi_host->data_count,gi_host);
    if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error: at adj  thread synchronization 8: %s\n",cudaGetErrorString(cudaGetLastError()));

    //get result
    copyFromDevice<CufftType>(data_d, kspaceDataSorted + data_coil_offset,data_count);
  }//iterate over coils
  cufftDestroy(fft_plan);
  // Destroy the cuFFT plan.
  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error at thread synchronization 9: %s\n",cudaGetErrorString(cudaGetLastError()));
  freeLookupTable();
  freeTotalDeviceMemory(data_d,crds_d,gdata_d,imdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop
  if (n_coils > 1 && deapo_d != NULL)
    cudaFree(deapo_d);

  if ((cudaThreadSynchronize() != cudaSuccess))
    fprintf(stderr,"error in gridding3D_gpu function: %s\n",cudaGetErrorString(cudaGetLastError()));
  free(gi_host);

  writeOrdered<CufftType>(kspaceData,kspaceDataSorted,(int)this->kSpaceTraj.count());

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
