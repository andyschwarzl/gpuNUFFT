#ifndef PRECOMP_KERNELS_CU
#define PRECOMP_KERNELS_CU

#include "precomp_kernels.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"

__global__ void assignSectorsKernel(DType* kSpaceTraj,
  IndType* assignedSectors,
  long coordCnt,
  bool is2DProcessing,
  gpuNUFFT::Dimensions gridSectorDims,
  gpuNUFFT::Dimensions gridDims,
  int sectorWidth)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;
  IndType sector;

  while (t < coordCnt) 
  {
    if (is2DProcessing)
    {
      DType2 coord;
      coord.x = kSpaceTraj[t];
      coord.y = kSpaceTraj[t + coordCnt];
      IndType2 mappedSector = computeSectorMapping(coord,gridDims,sectorWidth);
      //linearize mapped sector
      sector = computeInd22Lin(mappedSector,gridSectorDims);		
    }
    else
    {
      DType3 coord;
      coord.x = kSpaceTraj[t];
      coord.y = kSpaceTraj[t + coordCnt];
      coord.z = kSpaceTraj[t + 2*coordCnt];
      IndType3 mappedSector = computeSectorMapping(coord,gridDims,sectorWidth);
      //linearize mapped sector
      sector = computeInd32Lin(mappedSector,gridSectorDims);		
    }

    assignedSectors[t] = sector;

    t = t+ blockDim.x*gridDim.x;
  }
}

void assignSectorsGPU(gpuNUFFT::GpuNUFFTOperator* gpuNUFFTOp, gpuNUFFT::Array<DType>& kSpaceTraj, IndType* assignedSectors)
{
  IndType coordCnt = kSpaceTraj.count();

  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim((long)coordCnt,THREAD_BLOCK_SIZE));

  DType* kSpaceTraj_d;
  IndType* assignedSectors_d;

  if (DEBUG)
    printf("allocate and copy trajectory of size %d...\n",gpuNUFFTOp->getImageDimensionCount()*coordCnt);
  allocateAndCopyToDeviceMem<DType>(&kSpaceTraj_d,kSpaceTraj.data,gpuNUFFTOp->getImageDimensionCount()*coordCnt);

  if (DEBUG)
    printf("allocate and copy data of size %d...\n",coordCnt);
  allocateDeviceMem<IndType>(&assignedSectors_d,coordCnt);

  assignSectorsKernel<<<grid_dim,block_dim>>>(kSpaceTraj_d,
    assignedSectors_d,
    (long)coordCnt,
    gpuNUFFTOp->is2DProcessing(),
    gpuNUFFTOp->getGridSectorDims(),
    gpuNUFFTOp->getGridDims(),
    gpuNUFFTOp->getSectorWidth());

  if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
    printf("error: at assignSectors thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));

  //get result from device 
  copyFromDevice<IndType>(assignedSectors_d,assignedSectors,coordCnt);

  if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
    printf("error: at assignSectors thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));

  freeTotalDeviceMemory(kSpaceTraj_d,assignedSectors_d,NULL);//NULL as stop
}

__global__ void sortArraysKernel(gpuNUFFT::IndPair* assignedSectorsAndIndicesSorted,
  IndType* assignedSectors, 
  IndType* dataIndices,
  DType* kSpaceTraj,
  DType* trajSorted,
  DType* densCompData,
  DType* densData,
  bool is3DProcessing,
  long coordCnt)
{
  int t = threadIdx.x +  blockIdx.x *blockDim.x;

  while (t < coordCnt) 
  {
    trajSorted[t] = kSpaceTraj[assignedSectorsAndIndicesSorted[t].first];
    trajSorted[t + 1*coordCnt] = kSpaceTraj[assignedSectorsAndIndicesSorted[t].first + 1*coordCnt];
    if (is3DProcessing)
      trajSorted[t + 2*coordCnt] = kSpaceTraj[assignedSectorsAndIndicesSorted[t].first + 2*coordCnt];

    //sort density compensation
    if (densCompData != NULL)
      densData[t] = densCompData[assignedSectorsAndIndicesSorted[t].first];

    dataIndices[t] = assignedSectorsAndIndicesSorted[t].first;
    assignedSectors[t] = assignedSectorsAndIndicesSorted[t].second;		

    t = t+ blockDim.x*gridDim.x;
  }
}

void sortArrays(gpuNUFFT::GpuNUFFTOperator* gpuNUFFTOp, 
  std::vector<gpuNUFFT::IndPair> assignedSectorsAndIndicesSorted,
  IndType* assignedSectors, 
  IndType* dataIndices,
  gpuNUFFT::Array<DType>& kSpaceTraj,
  DType* trajSorted,
  DType* densCompData,
  DType* densData)
{
  IndType coordCnt = kSpaceTraj.count();
  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim((long)coordCnt,THREAD_BLOCK_SIZE));

  DType* kSpaceTraj_d;
  gpuNUFFT::IndPair* assignedSectorsAndIndicesSorted_d;
  IndType* assignedSectors_d;
  IndType* dataIndices_d;
  DType* trajSorted_d;
  DType* densCompData_d = NULL;
  DType* densData_d = NULL;

  //Trajectory and sorted result 
  allocateAndCopyToDeviceMem<DType>(&kSpaceTraj_d,kSpaceTraj.data,gpuNUFFTOp->getImageDimensionCount()*coordCnt);
  allocateDeviceMem<DType>(&trajSorted_d,gpuNUFFTOp->getImageDimensionCount()*coordCnt);

  //Assigned sorted sectors and data indices and result
  allocateAndCopyToDeviceMem<gpuNUFFT::IndPair>(&assignedSectorsAndIndicesSorted_d,&assignedSectorsAndIndicesSorted[0],coordCnt);
  allocateDeviceMem<IndType>(&assignedSectors_d,coordCnt);	 
  allocateDeviceMem<IndType>(&dataIndices_d,coordCnt);	 

  //Density compensation data and sorted result
  if (densCompData != NULL)
  {
    allocateAndCopyToDeviceMem<DType>(&densCompData_d,densCompData,coordCnt);
    allocateDeviceMem<DType>(&densData_d,coordCnt);
  }

  if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
    printf("error: at sortArrays thread synchronization 0: %s\n",cudaGetErrorString(cudaGetLastError()));

  sortArraysKernel<<<grid_dim,block_dim>>>( assignedSectorsAndIndicesSorted_d,
    assignedSectors_d, 
    dataIndices_d,
    kSpaceTraj_d,
    trajSorted_d,
    densCompData_d,
    densData_d,
    gpuNUFFTOp->is3DProcessing(),
    (long)coordCnt);
  if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
    printf("error: at sortArrays thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));

  copyFromDevice<IndType>(assignedSectors_d,assignedSectors,coordCnt);
  copyFromDevice<IndType>(dataIndices_d,dataIndices,coordCnt);
  copyFromDevice<DType>(trajSorted_d,trajSorted,gpuNUFFTOp->getImageDimensionCount()*coordCnt);
  if (densCompData != NULL)
    copyFromDevice<DType>(densData_d,densData,coordCnt);

  if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
    printf("error: at sortArrays thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));

  freeTotalDeviceMemory(kSpaceTraj_d,assignedSectorsAndIndicesSorted_d,assignedSectors_d,dataIndices_d,trajSorted_d,densCompData_d,densData_d,NULL);//NULL as stop
}

__global__ void selectOrderedGPUKernel(DType2* data, DType2* data_sorted, IndType* dataIndices, int N, int n_coils_cc)
{
  int t = threadIdx.x + blockIdx.x * blockDim.x;
  
  while (t < N) 
  {
    int c = 0;
    while (c < n_coils_cc)
    {
      data_sorted[t + c * N] = data[dataIndices[t] + c * N];
      c += 1;
    }

    t = t + blockDim.x * gridDim.x;
  }
}

void selectOrderedGPU(DType2* data_d, IndType* data_indices_d, DType2* data_sorted_d,int N, int n_coils_cc)
{
  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim(N,THREAD_BLOCK_SIZE)); 

  selectOrderedGPUKernel<<<grid_dim,block_dim>>>(data_d,data_sorted_d,data_indices_d,N,n_coils_cc);

  if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
    printf("error: at selectOrderedGPU thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));
}

__global__ void writeOrderedGPUKernel(DType2* data_sorted, CufftType* data, IndType* dataIndices, int N, int n_coils_cc)
{
  int t = threadIdx.x + blockIdx.x * blockDim.x;
  
  while (t < N) 
  {
    int c = 0;
    while (c < n_coils_cc)
    {
      data_sorted[dataIndices[t] + c * N] = data[t + c * N];
      c++;
    }

    t = t + blockDim.x * gridDim.x;
  }
}

void writeOrderedGPU( DType2* data_sorted_d, IndType* data_indices_d,CufftType* data_d, int N, int n_coils_cc)
{
  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim(N,THREAD_BLOCK_SIZE)); 
  
  writeOrderedGPUKernel<<<grid_dim,block_dim>>>(data_sorted_d,data_d,data_indices_d,N, n_coils_cc);

  if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
    printf("error: at writeOrderedGPU thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));
}

#endif
