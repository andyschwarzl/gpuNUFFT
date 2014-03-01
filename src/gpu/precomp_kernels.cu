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
  IndType sector;
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

#endif