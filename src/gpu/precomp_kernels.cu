#ifndef PRECOMP_KERNELS_CU
#define PRECOMP_KERNELS_CU

#include "precomp_kernels.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"

__global__ void assignSectorsKernel(DType* kSpaceTraj,
  IndType* assignedSectors,
  int coordCnt,
  bool is2DProcessing,
  GriddingND::Dimensions gridSectorDims)
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
      IndType2 mappedSector = computeSectorMapping(coord,gridSectorDims);
      //linearize mapped sector
      sector = computeInd22Lin(mappedSector,gridSectorDims);		
    }
    else
    {
      DType3 coord;
      coord.x = kSpaceTraj[t];
      coord.y = kSpaceTraj[t + coordCnt];
      coord.z = kSpaceTraj[t + 2*coordCnt];
      IndType3 mappedSector = computeSectorMapping(coord,gridSectorDims);
      //linearize mapped sector
      sector = computeInd32Lin(mappedSector,gridSectorDims);		
    }

    assignedSectors[t] = sector;

    t = t+ blockDim.x*gridDim.x;
  }
}

void assignSectorsGPU(GriddingND::GriddingOperator* griddingOp, GriddingND::Array<DType>& kSpaceTraj, IndType* assignedSectors)
{
  IndType coordCnt = kSpaceTraj.count();

  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim(coordCnt,THREAD_BLOCK_SIZE));

  DType* kSpaceTraj_d;
  IndType* assignedSectors_d;

  if (DEBUG)
    printf("allocate and copy trajectory of size %d...\n",griddingOp->getImageDimensionCount()*coordCnt);
  allocateAndCopyToDeviceMem<DType>(&kSpaceTraj_d,kSpaceTraj.data,griddingOp->getImageDimensionCount()*coordCnt);

  if (DEBUG)
    printf("allocate and copy data of size %d...\n",coordCnt);
  allocateDeviceMem<IndType>(&assignedSectors_d,coordCnt);

  assignSectorsKernel<<<grid_dim,block_dim>>>(kSpaceTraj_d,
    assignedSectors_d,
    coordCnt,
    griddingOp->is2DProcessing(),
    griddingOp->getGridSectorDims());

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error: at assignSectors thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));

  //get result from device 
  copyFromDevice<IndType>(assignedSectors_d,assignedSectors,coordCnt);

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error: at assignSectors thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));

  freeTotalDeviceMemory(kSpaceTraj_d,assignedSectors_d,NULL);//NULL as stop
}

__global__ void sortArraysKernel(GriddingND::IndPair* assignedSectorsAndIndicesSorted,
  IndType* assignedSectors, 
  IndType* dataIndices,
  DType* kSpaceTraj,
  DType* trajSorted,
  DType* densCompData,
  DType* densData,
  bool is3DProcessing,
  int coordCnt)
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

void sortArrays(GriddingND::GriddingOperator* griddingOp, 
  std::vector<GriddingND::IndPair> assignedSectorsAndIndicesSorted,
  IndType* assignedSectors, 
  IndType* dataIndices,
  GriddingND::Array<DType>& kSpaceTraj,
  DType* trajSorted,
  DType* densCompData,
  DType* densData)
{
  IndType coordCnt = kSpaceTraj.count();
  dim3 block_dim(THREAD_BLOCK_SIZE);
  dim3 grid_dim(getOptimalGridDim(coordCnt,THREAD_BLOCK_SIZE));

  DType* kSpaceTraj_d;
  GriddingND::IndPair* assignedSectorsAndIndicesSorted_d;
  IndType* assignedSectors_d;
  IndType* dataIndices_d;
  DType* trajSorted_d;
  DType* densCompData_d = NULL;
  DType* densData_d = NULL;

  //Trajectory and sorted result 
  allocateAndCopyToDeviceMem<DType>(&kSpaceTraj_d,kSpaceTraj.data,griddingOp->getImageDimensionCount()*coordCnt);
  allocateDeviceMem<DType>(&trajSorted_d,griddingOp->getImageDimensionCount()*coordCnt);

  //Assigned sorted sectors and data indices and result
  allocateAndCopyToDeviceMem<GriddingND::IndPair>(&assignedSectorsAndIndicesSorted_d,assignedSectorsAndIndicesSorted.data(),coordCnt);
  allocateDeviceMem<IndType>(&assignedSectors_d,coordCnt);	 
  allocateDeviceMem<IndType>(&dataIndices_d,coordCnt);	 

  //Density compensation data and sorted result
  if (densCompData != NULL)
  {
    allocateAndCopyToDeviceMem<DType>(&densCompData_d,densCompData,coordCnt);
    allocateDeviceMem<DType>(&densData_d,coordCnt);
  }

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error: at sortArrays thread synchronization 0: %s\n",cudaGetErrorString(cudaGetLastError()));

  sortArraysKernel<<<grid_dim,block_dim>>>( assignedSectorsAndIndicesSorted_d,
    assignedSectors_d, 
    dataIndices_d,
    kSpaceTraj_d,
    trajSorted_d,
    densCompData_d,
    densData_d,
    griddingOp->is3DProcessing(),
    coordCnt);
  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error: at sortArrays thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));

  copyFromDevice<IndType>(assignedSectors_d,assignedSectors,coordCnt);
  copyFromDevice<IndType>(dataIndices_d,dataIndices,coordCnt);
  copyFromDevice<DType>(trajSorted_d,trajSorted,griddingOp->getImageDimensionCount()*coordCnt);
  if (densCompData != NULL)
    copyFromDevice<DType>(densData_d,densData,coordCnt);

  if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
    printf("error: at sortArrays thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));

  freeTotalDeviceMemory(kSpaceTraj_d,assignedSectorsAndIndicesSorted_d,assignedSectors_d,dataIndices_d,trajSorted_d,densCompData_d,densData_d,NULL);//NULL as stop
}

#endif