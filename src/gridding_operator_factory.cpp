
#include "gridding_operator_factory.hpp"
#include <iostream>

GriddingND::GriddingOperatorFactory GriddingND::GriddingOperatorFactory::instance;

GriddingND::GriddingOperatorFactory& GriddingND::GriddingOperatorFactory::getInstance()
{
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
	debug("in assign sectors\n");

	griddingOp->setSectorDims(computeSectorCountPerDimension(griddingOp->getGridDims(),griddingOp->getSectorWidth()));
	
	size_t coordCnt = kSpaceTraj.count();

	//create temporary array to store assigned values
	GriddingND::Array<IndType> assignedSectors;
    assignedSectors.data = (IndType*)malloc(coordCnt * sizeof(IndType));
    assignedSectors.dim.length = coordCnt;
	
	for (int cCnt = 0; cCnt < coordCnt; cCnt++)
	{
		DType3 coord;
		coord.x = kSpaceTraj.data[cCnt];
		coord.y = kSpaceTraj.data[cCnt + coordCnt];
		coord.z = kSpaceTraj.data[cCnt + 2*coordCnt];
		
		IndType3 mappedSector = computeSectorMapping(coord,griddingOp->getSectorDims());

		//linearize mapped sector
		size_t sector = computeInd32Lin(mappedSector,griddingOp->getSectorDims());
		assignedSectors.data[cCnt] = sector;
	}
	debug("finished assign sectors\n");
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
	Array<IndType> sectorDataCount = initSectorDataCount(griddingOp,dataCount.size());
	std::copy( dataCount.begin(), dataCount.end(), sectorDataCount.data );
	
	return sectorDataCount;
}

inline IndType GriddingND::GriddingOperatorFactory::computeSectorCenter(IndType var, IndType sectorWidth)
{
	return (IndType)(var*sectorWidth + std::floor(static_cast<DType>(sectorWidth) / (DType)2.0));
}

void GriddingND::GriddingOperatorFactory::debug(const std::string& message)
{
	std::cout << message << std::endl;
}

GriddingND::Array<IndType3> GriddingND::GriddingOperatorFactory::computeSectorCenters(GriddingND::GriddingOperator *griddingOp)
{
	
	GriddingND::Dimensions sectorDims = griddingOp->getSectorDims();
	IndType sectorWidth = griddingOp->getSectorWidth();

	GriddingND::Array<IndType3> sectorCenters = initSectorCenters(griddingOp,sectorDims.count());
	
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

//default implementation
GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::initDataIndices(GriddingND::GriddingOperator* griddingOp, size_t coordCnt)
{
	GriddingND::Array<IndType> dataIndices;
	dataIndices.data = (IndType*)malloc(coordCnt*sizeof(IndType));
	dataIndices.dim.length = coordCnt;
	return dataIndices;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::initSectorDataCount(GriddingND::GriddingOperator* griddingOp, size_t dataCount)
{
	Array<IndType> sectorDataCount;
	sectorDataCount.data = (IndType*)malloc(dataCount*sizeof(IndType));
	sectorDataCount.dim.length = dataCount;
	return sectorDataCount;
}

GriddingND::Array<DType> GriddingND::GriddingOperatorFactory::initDensData(GriddingND::GriddingOperator* griddingOp, size_t coordCnt)
{
	GriddingND::Array<DType> densData;

	return densData;
}

GriddingND::Array<DType> GriddingND::GriddingOperatorFactory::initCoordsData(GriddingND::GriddingOperator* griddingOp, size_t coordCnt)
{
	GriddingND::Array<DType> coordsData;
	coordsData.data = (DType*)malloc(coordCnt*3*sizeof(DType));
	coordsData.dim.length = coordCnt;
	return coordsData;
}

GriddingND::Array<IndType3> GriddingND::GriddingOperatorFactory::initSectorCenters(GriddingND::GriddingOperator* griddingOp, size_t sectorCnt)
{
	GriddingND::Array<IndType3> sectorCenters; 
	sectorCenters.data = (IndType3*)malloc(sectorCnt * sizeof(IndType3));
	sectorCenters.dim.length = sectorCnt;

	return sectorCenters;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
    //validate arguments
	if (kSpaceTraj.dim.channels > 1)
		throw std::invalid_argument("Trajectory dimension must not contain a channel size greater than 1!");
	
	if (imgDims.channels > 1)
		throw std::invalid_argument("Image dimensions must not contain a channel size greater than 1!");
	
	std::cout << "create gridding operator" << std::endl;
    GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernelWidth,sectorWidth,osf,imgDims);
	
	//assign Sectors
	GriddingND::Array<IndType> assignedSectors = assignSectors(griddingOp, kSpaceTraj);

	std::vector<IndPair> assignedSectorsAndIndicesSorted = sortVector(assignedSectors);
	
	size_t coordCnt = kSpaceTraj.dim.count();

	Array<DType>   trajSorted = initCoordsData(griddingOp,coordCnt);
	Array<IndType> dataIndices = initDataIndices(griddingOp,coordCnt);
	Array<DType>   densData;
	if (griddingOp->applyDensComp())
	 densData = initDensData(griddingOp,coordCnt);

	//sort kspace data coords
	for (int i=0; i<coordCnt;i++)
	{
		trajSorted.data[i] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first];
		trajSorted.data[i + 1*coordCnt] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first + 1*coordCnt];
		trajSorted.data[i + 2*coordCnt] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first + 2*coordCnt];
		
		//todo sort density compensation
		//densData.data[i] = dens.data[assignedSectorsAndIndicesSorted[i].first];

		dataIndices.data[i] = assignedSectorsAndIndicesSorted[i].first;
		assignedSectors.data[i] = assignedSectorsAndIndicesSorted[i].second;		
	}
	griddingOp->setDataIndices(dataIndices);
	//todo free mem ?
	//free(kSpaceTraj.data);

	griddingOp->setKSpaceTraj(trajSorted);

	griddingOp->setSectorDataCount(computeSectorDataCount(griddingOp,assignedSectors));
	
	griddingOp->setSectorCenters(computeSectorCenters(griddingOp));

	//free temporary array
	free(assignedSectors.data);
	std::cout << "finished creation of gridding operator" << std::endl;
	return griddingOp;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<DType>& densCompData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
	GriddingOperator* op = createGriddingOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);
	op->setDens(densCompData);
	return op;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<DType>& densCompData, GriddingND::Array<DType2>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
	return createGriddingOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);
}

