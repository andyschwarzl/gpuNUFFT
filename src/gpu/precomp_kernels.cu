#ifndef PRECOMP_KERNELS_CU
#define PRECOMP_KERNELS_CU

#include "precomp_kernels.hpp"
#include "cuda_utils.cuh"

void assignSectorsGPU(GriddingND::GriddingOperator* griddingOp, GriddingND::Array<DType>& kSpaceTraj, IndType* assignedSectors)
{
  IndType sector;
  IndType coordCnt = kSpaceTraj.count();
	for (IndType cCnt = 0; cCnt < coordCnt; cCnt++)
	{
		if (griddingOp->is2DProcessing())
		{
			DType2 coord;
			coord.x = kSpaceTraj.data[cCnt];
			coord.y = kSpaceTraj.data[cCnt + coordCnt];
			IndType2 mappedSector = computeSectorMapping(coord,griddingOp->getGridSectorDims());
			//linearize mapped sector
			sector = computeInd22Lin(mappedSector,griddingOp->getGridSectorDims());		
		}
		else
		{
			DType3 coord;
			coord.x = kSpaceTraj.data[cCnt];
			coord.y = kSpaceTraj.data[cCnt + coordCnt];
			coord.z = kSpaceTraj.data[cCnt + 2*coordCnt];
			IndType3 mappedSector = computeSectorMapping(coord,griddingOp->getGridSectorDims());
			//linearize mapped sector
			sector = computeInd32Lin(mappedSector,griddingOp->getGridSectorDims());		
		}

		assignedSectors[cCnt] = sector;
  }
}

#endif