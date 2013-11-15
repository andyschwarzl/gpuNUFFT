
#include "gridding_operator_factory.hpp"
#include <iostream>

GriddingND::GriddingOperatorFactory* GriddingND::GriddingOperatorFactory::instance = NULL;

GriddingND::GriddingOperatorFactory* GriddingND::GriddingOperatorFactory::getInstance()
{
          if (instance == NULL)
            instance = new GriddingOperatorFactory();
          return instance;
}

size_t GriddingND::GriddingOperatorFactory::computeSectorMapping(DType coord, size_t sectorCount)
{
	size_t sector = (size_t)std::floor(static_cast<DType>(coord + 0.5) * sectorCount);
	if (sector >= sectorCount) 
		sector = sectorCount -1;
	if (sector < 0)
		sector = 0;
	return sector;
}

IndType3 GriddingND::GriddingOperatorFactory::computeSectorMapping(DType3 coord, GriddingND::Dimensions sectorDims)
{
	IndType3 sector;
	sector.x = computeSectorMapping(coord.x,sectorDims.width);
	sector.y  = computeSectorMapping(coord.y,sectorDims.height);
	sector.z  = computeSectorMapping(coord.z,sectorDims.depth);
	return sector;
}

IndType2 GriddingND::GriddingOperatorFactory::computeSectorMapping(DType2 coord, GriddingND::Dimensions sectorDims)
{
	IndType2 sector;
	sector.x = computeSectorMapping(coord.x,sectorDims.width);
	sector.y  = computeSectorMapping(coord.y,sectorDims.height);
	return sector;
}

size_t GriddingND::GriddingOperatorFactory::computeXYZ2Lin(size_t x, size_t y, size_t z, GriddingND::Dimensions dim)
{
	return x + dim.height * (y + dim.depth * z);
}

size_t GriddingND::GriddingOperatorFactory::computeInd32Lin(IndType3 sector, GriddingND::Dimensions dim)
{
	return sector.x + dim.height * (sector.y + dim.depth * sector.z);
}


size_t GriddingND::GriddingOperatorFactory::computeSectorCountPerDimension(size_t dim, size_t sectorWidth)
{
	return (size_t)std::ceil(static_cast<DType>(dim) / sectorWidth);
}

GriddingND::Dimensions GriddingND::GriddingOperatorFactory::computeSectorCountPerDimension(GriddingND::Dimensions dim, size_t sectorWidth)
{
	GriddingND::Dimensions sectorDims;
	sectorDims.width = computeSectorCountPerDimension(dim.width,sectorWidth);
	sectorDims.height = computeSectorCountPerDimension(dim.height,sectorWidth);
	sectorDims.depth = computeSectorCountPerDimension(dim.depth,sectorWidth);
	return sectorDims;
}

size_t GriddingND::GriddingOperatorFactory::computeTotalSectorCount(GriddingND::Dimensions dim, size_t sectorWidth)
{
	return computeSectorCountPerDimension(dim,sectorWidth).count();
}

std::vector<GriddingND::IndPair> GriddingND::GriddingOperatorFactory::sortVector(GriddingND::Array<size_t> assignedSectors)
{
	std::vector<IndPair> secVector;
	
	for (size_t i=0; i< assignedSectors.count(); i++)
	  secVector.push_back(IndPair(i,assignedSectors.data[i]));

	// using function as comp
	std::sort(secVector.begin(), secVector.end());

	return secVector;
}


GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::assignSectors(GriddingND::GriddingOperator* griddingOp, GriddingND::Array<DType>& kSpaceTraj)
{
	griddingOp->setSectorDims(computeSectorCountPerDimension(griddingOp->getGridDims(),griddingOp->getSectorWidth()));

	size_t coordCnt = kSpaceTraj.dim.count();

	GriddingND::Array<IndType> assignedSectors;
    assignedSectors.data = (IndType*)malloc(coordCnt * sizeof(IndType));
    assignedSectors.dim.length = coordCnt;

	for (int cCnt = 0; cCnt < coordCnt; cCnt++)
	{
		DType3 coord;
		coord.x = kSpaceTraj.data[cCnt];
		coord.y = kSpaceTraj.data[cCnt + coordCnt];
		coord.z = kSpaceTraj.data[cCnt + 2*coordCnt];

		IndType3 mappedSectors = computeSectorMapping(coord,griddingOp->getSectorDims());

		size_t sector = computeInd32Lin(mappedSectors,griddingOp->getSectorDims());
		assignedSectors.data[cCnt] = sector;
	}

	return assignedSectors;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::computeSectorDataCount(GriddingND::GriddingOperator *griddingOp,GriddingND::Array<IndType> assignedSectors)
{
	size_t cnt = 0;
	std::vector<IndType> dataCount;

	dataCount.push_back(0);
	for (int i=0; i<=griddingOp->getSectorDims().count(); i++)
	{	
		while (cnt < assignedSectors.count() && i == assignedSectors.data[cnt])
			cnt++;
		
		dataCount.push_back(cnt);
	}
	Array<IndType> sectorDataCount;
	sectorDataCount.data = (IndType*)malloc(dataCount.size()*sizeof(IndType));
	std::copy( dataCount.begin(), dataCount.end(), sectorDataCount.data );
	//sectorDataCount.data = &dataCount[0];
	sectorDataCount.dim.length = dataCount.size();
	return sectorDataCount;
}

inline IndType GriddingND::GriddingOperatorFactory::computeSectorCenter(IndType var, IndType sectorWidth)
{
	return (IndType)(var*sectorWidth + std::floor(static_cast<DType>(sectorWidth) / (DType)2.0));
}

GriddingND::Array<IndType3> GriddingND::GriddingOperatorFactory::computeSectorCenters(GriddingND::GriddingOperator *griddingOp)
{
	GriddingND::Array<IndType3> sectorCenters; 
	GriddingND::Dimensions sectorDims = griddingOp->getSectorDims();
	IndType sectorWidth = griddingOp->getSectorWidth();

	sectorCenters.data = (IndType3*)malloc(sectorDims.count() * sizeof(IndType3));
	sectorCenters.dim.length = sectorDims.count();

	for (size_t z=0;z<sectorDims.depth; z++)
		for (size_t y=0;y<sectorDims.height;y++)
			for (size_t x=0;x<sectorDims.width;x++)
			{
				IndType3 center;
				center.x = computeSectorCenter(x,sectorWidth);
				center.y = computeSectorCenter(y,sectorWidth);
				center.z = computeSectorCenter(z,sectorWidth);
				size_t index = computeXYZ2Lin(x,y,z,sectorDims);
				sectorCenters.data[index] = center;
			}
    return sectorCenters;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
    GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernelWidth,sectorWidth,osf);
	
	std::cout << "create gridding operator" << std::endl;
    
	griddingOp->setKSpaceTraj(kSpaceTraj);
	griddingOp->setImageDims(imgDims);

	GriddingND::Array<IndType> assignedSectors = assignSectors(griddingOp, kSpaceTraj);

	std::vector<IndPair> assignedSectorsAndIndicesSorted = sortVector(assignedSectors);
	
	size_t coordCnt = kSpaceTraj.dim.count();
	DType* coords_sorted = (DType*)malloc(coordCnt*3*sizeof(DType));

	GriddingND::Array<IndType> dataIndices;
	dataIndices.data = (IndType*)malloc(assignedSectors.count()*sizeof(IndType));
	dataIndices.dim.length = assignedSectors.count();

	//sort kspace data coords
	for (int i=0; i<coordCnt;i++)
	{
		coords_sorted[i] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first];
		coords_sorted[i + 1*coordCnt] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first + 1*coordCnt];
		coords_sorted[i + 2*coordCnt] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first + 2*coordCnt];
		
		//todo sort density compensation

		dataIndices.data[i] = assignedSectorsAndIndicesSorted[i].first;
		assignedSectors.data[i] = assignedSectorsAndIndicesSorted[i].second;		
	}

	//todo free mem
	//free(kSpaceTraj.data);
	kSpaceTraj.data = coords_sorted;

	griddingOp->setKSpaceTraj(kSpaceTraj);
	griddingOp->setDataIndices(dataIndices);

	griddingOp->setSectorDataCount(computeSectorDataCount(griddingOp,assignedSectors));
	griddingOp->setSectorCenters(computeSectorCenters(griddingOp));
	
	std::cout << "finished creation of gridding operator" << std::endl;
	return griddingOp;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<DType>& densCompData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
	GriddingOperator* op = createGriddingOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);
	op->setDens(densCompData);
	return op;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<DType>& densCompData, GriddingND::Array<DType>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
	return createGriddingOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);
}

