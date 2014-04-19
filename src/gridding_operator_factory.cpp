
#include "gridding_operator_factory.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "precomp_kernels.hpp"

GriddingND::GriddingOperatorFactory GriddingND::GriddingOperatorFactory::instance;

GriddingND::GriddingOperatorFactory& GriddingND::GriddingOperatorFactory::getInstance()
{
  return instance;
}

IndType GriddingND::GriddingOperatorFactory::computeSectorCountPerDimension(IndType dim, IndType sectorWidth)
{
  return (IndType)std::ceil(static_cast<DType>(dim) / sectorWidth);
}

template <typename T>
GriddingND::Array<T> GriddingND::GriddingOperatorFactory::initLinArray(IndType arrCount)
{
  GriddingND::Array<T> new_array;
  new_array.data = (T*)malloc(arrCount * sizeof(T));
  new_array.dim.length = arrCount;
  return new_array;
}

GriddingND::Dimensions GriddingND::GriddingOperatorFactory::computeSectorCountPerDimension(GriddingND::Dimensions dim, IndType sectorWidth)
{
  GriddingND::Dimensions sectorDims;
  sectorDims.width = computeSectorCountPerDimension(dim.width,sectorWidth);
  sectorDims.height = computeSectorCountPerDimension(dim.height,sectorWidth);
  sectorDims.depth = computeSectorCountPerDimension(dim.depth,sectorWidth);
  return sectorDims;
}

IndType GriddingND::GriddingOperatorFactory::computeTotalSectorCount(GriddingND::Dimensions dim, IndType sectorWidth)
{
  return computeSectorCountPerDimension(dim,sectorWidth).count();
}

template <typename T>
std::vector<GriddingND::IndPair> GriddingND::GriddingOperatorFactory::sortVector(GriddingND::Array<T> assignedSectors)
{
  std::vector<IndPair> secVector;

  for (IndType i=0; i< assignedSectors.count(); i++)
    secVector.push_back(IndPair(i,assignedSectors.data[i]));

  // using function as comp
  std::sort(secVector.begin(), secVector.end());

  return secVector;
}


GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::assignSectors(GriddingND::GriddingOperator* griddingOp, GriddingND::Array<DType>& kSpaceTraj)
{
  debug("in assign sectors\n");

  griddingOp->setGridSectorDims(computeSectorCountPerDimension(griddingOp->getGridDims(),griddingOp->getSectorWidth()));

  IndType coordCnt = kSpaceTraj.count();

  //create temporary array to store assigned values
  GriddingND::Array<IndType> assignedSectors;
  assignedSectors.data = (IndType*)malloc(coordCnt * sizeof(IndType));
  assignedSectors.dim.length = coordCnt;

  if (useGpu)
  {
    assignSectorsGPU(griddingOp, kSpaceTraj, assignedSectors.data);
  }
  else
  {
    IndType sector;
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

      assignedSectors.data[cCnt] = sector;
    }
  }
  debug("finished assign sectors\n");
  return assignedSectors;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::computeSectorDataCount(GriddingND::GriddingOperator *griddingOp,GriddingND::Array<IndType> assignedSectors)
{
  IndType cnt = 0;
  std::vector<IndType> dataCount;

  dataCount.push_back(0);
  for (IndType i=0; i<=griddingOp->getGridSectorDims().count(); i++)
  {	
    while (cnt < assignedSectors.count() && i == assignedSectors.data[cnt])
      cnt++;

    dataCount.push_back(cnt);
  }
  Array<IndType> sectorDataCount = initSectorDataCount(griddingOp,(IndType)dataCount.size());
   std::copy( dataCount.begin(), dataCount.end(), sectorDataCount.data );

  return sectorDataCount;
}

inline IndType GriddingND::GriddingOperatorFactory::computeSectorCenter(IndType var, IndType sectorWidth)
{
  return (IndType)(var*sectorWidth + std::floor(static_cast<DType>(sectorWidth) / (DType)2.0));
}

void GriddingND::GriddingOperatorFactory::debug(const std::string& message)
{
  if (DEBUG)
    std::cout << message << std::endl;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::computeSectorCenters2D(GriddingND::GriddingOperator *griddingOp)
{
  GriddingND::Dimensions sectorDims = griddingOp->getGridSectorDims();
  IndType sectorWidth = griddingOp->getSectorWidth();

  GriddingND::Array<IndType> sectorCenters = initSectorCenters(griddingOp,sectorDims.count());

  for (IndType y=0;y<sectorDims.height; y++)
    for (IndType x=0;x<sectorDims.width;x++)
    {
      IndType2 center;
      center.x = computeSectorCenter(x,sectorWidth);
      center.y = computeSectorCenter(y,sectorWidth);
      int index = computeXY2Lin((int)x,(int)y,sectorDims);
      sectorCenters.data[2*index] = center.x;
      sectorCenters.data[2*index+1] = center.y;
    }
    return sectorCenters;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::computeSectorCenters(GriddingND::GriddingOperator *griddingOp)
{
  GriddingND::Dimensions sectorDims = griddingOp->getGridSectorDims();
  IndType sectorWidth = griddingOp->getSectorWidth();

  GriddingND::Array<IndType> sectorCenters = initSectorCenters(griddingOp,sectorDims.count());

  for (IndType z=0;z<sectorDims.depth; z++)
    for (IndType y=0;y<sectorDims.height;y++)
      for (IndType x=0;x<sectorDims.width;x++)
      {
        IndType3 center;
        center.x = computeSectorCenter(x,sectorWidth);
        center.y = computeSectorCenter(y,sectorWidth);
        center.z = computeSectorCenter(z,sectorWidth);
        int index = computeXYZ2Lin((int)x,(int)y,(int)z,sectorDims);
        //necessary in order to avoid 2d or 3d typed array
        sectorCenters.data[3*index] = center.x;
        sectorCenters.data[3*index+1] = center.y;
        sectorCenters.data[3*index+2] = center.z;
      }
      return sectorCenters;
}

//default implementation
GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::initDataIndices(GriddingND::GriddingOperator* griddingOp, IndType coordCnt)
{
  return initLinArray<IndType>(coordCnt);
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::initSectorDataCount(GriddingND::GriddingOperator* griddingOp, IndType dataCount)
{
  return initLinArray<IndType>(dataCount);
}

GriddingND::Array<DType> GriddingND::GriddingOperatorFactory::initDensData(GriddingND::GriddingOperator* griddingOp, IndType coordCnt)
{
  return initLinArray<DType>(coordCnt);
}

GriddingND::Array<DType> GriddingND::GriddingOperatorFactory::initCoordsData(GriddingND::GriddingOperator* griddingOp, IndType coordCnt)
{
  GriddingND::Array<DType> coordsData = initLinArray<DType>(griddingOp->getImageDimensionCount()*coordCnt);
  coordsData.dim.length = coordCnt;
  return coordsData;
}

GriddingND::Array<IndType> GriddingND::GriddingOperatorFactory::initSectorCenters(GriddingND::GriddingOperator* griddingOp, IndType sectorCnt)
{
  //distinguish between 2d and 3d data
  return initLinArray<IndType>(griddingOp->getImageDimensionCount()*sectorCnt);
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<DType>& densCompData,GriddingND::Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
  //validate arguments
  if (kSpaceTraj.dim.channels > 1)
    throw std::invalid_argument("Trajectory dimension must not contain a channel size greater than 1!");

  if (imgDims.channels > 1)
    throw std::invalid_argument("Image dimensions must not contain a channel size greater than 1!");

  debug("create gridding operator...");

  GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernelWidth,sectorWidth,osf,imgDims);

  //assign Sectors
  GriddingND::Array<IndType> assignedSectors = assignSectors(griddingOp, kSpaceTraj);

  std::vector<IndPair> assignedSectorsAndIndicesSorted = sortVector<IndType>(assignedSectors);

  IndType coordCnt = kSpaceTraj.dim.count();

  Array<DType>   trajSorted = initCoordsData(griddingOp,coordCnt);
  Array<IndType> dataIndices = initDataIndices(griddingOp,coordCnt);

  Array<DType>   densData;
  if (densCompData.data != NULL)
    densData = initDensData(griddingOp,coordCnt);

  if (sensData.data != NULL)
    griddingOp->setSens(sensData);

  if (useGpu)
  {
    sortArrays(griddingOp,assignedSectorsAndIndicesSorted,assignedSectors.data, 
      dataIndices.data,kSpaceTraj,trajSorted.data,densCompData.data,densData.data);
  }
  else
  {
    //sort kspace data coords
    for (int i=0; i<coordCnt;i++)
    {
      trajSorted.data[i] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first];
      trajSorted.data[i + 1*coordCnt] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first + 1*coordCnt];
      if (griddingOp->is3DProcessing())
        trajSorted.data[i + 2*coordCnt] = kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first + 2*coordCnt];

      //sort density compensation
      if (densCompData.data != NULL)
        densData.data[i] = densCompData.data[assignedSectorsAndIndicesSorted[i].first];

      dataIndices.data[i] = assignedSectorsAndIndicesSorted[i].first;
      assignedSectors.data[i] = assignedSectorsAndIndicesSorted[i].second;		
    }
  }
  griddingOp->setDataIndices(dataIndices);

  griddingOp->setKSpaceTraj(trajSorted);

  griddingOp->setDens(densData);

  griddingOp->setSectorDataCount(computeSectorDataCount(griddingOp,assignedSectors));

  if (griddingOp->is3DProcessing())
    griddingOp->setSectorCenters(computeSectorCenters(griddingOp));
  else
    griddingOp->setSectorCenters(computeSectorCenters2D(griddingOp));

  //free temporary array
  free(assignedSectors.data);

  debug("finished creation of gridding operator\n");
  return griddingOp;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<DType>& densCompData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
  GriddingND::Array<DType2> sensData;
  return createGriddingOperator(kSpaceTraj,densCompData,sensData, kernelWidth, sectorWidth, osf, imgDims);
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::createGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
  GriddingND::Array<DType> densCompData;
  return createGriddingOperator(kSpaceTraj,densCompData, kernelWidth, sectorWidth, osf, imgDims);
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::loadPrecomputedGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<IndType>& dataIndices, GriddingND::Array<IndType>& sectorDataCount,GriddingND::Array<IndType>& sectorCenters, GriddingND::Array<DType>& densCompData, GriddingND::Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
	GriddingOperator* griddingOp = new GriddingOperator(kernelWidth,sectorWidth,osf,imgDims);
	griddingOp->setGridSectorDims(GriddingOperatorFactory::computeSectorCountPerDimension(griddingOp->getGridDims(),griddingOp->getSectorWidth()));
	
	griddingOp->setKSpaceTraj(kSpaceTraj);
	griddingOp->setDataIndices(dataIndices);
	griddingOp->setSectorDataCount(sectorDataCount);
	griddingOp->setSectorCenters(sectorCenters);
	griddingOp->setDens(densCompData);
	griddingOp->setSens(sensData);
	return griddingOp;
}

GriddingND::GriddingOperator* GriddingND::GriddingOperatorFactory::loadPrecomputedGriddingOperator(GriddingND::Array<DType>& kSpaceTraj, GriddingND::Array<IndType>& dataIndices, GriddingND::Array<IndType>& sectorDataCount,GriddingND::Array<IndType>& sectorCenters, GriddingND::Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, GriddingND::Dimensions& imgDims)
{
  GriddingOperator* griddingOp = new GriddingOperator(kernelWidth,sectorWidth,osf,imgDims);
	griddingOp->setGridSectorDims(GriddingOperatorFactory::computeSectorCountPerDimension(griddingOp->getGridDims(),griddingOp->getSectorWidth()));
	
	griddingOp->setKSpaceTraj(kSpaceTraj);
	griddingOp->setDataIndices(dataIndices);
	griddingOp->setSectorDataCount(sectorDataCount);
	griddingOp->setSectorCenters(sectorCenters);
	griddingOp->setSens(sensData);
	return griddingOp;
}
