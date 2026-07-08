
#include "gpuNUFFT_operator_factory.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <sstream>
#include "precomp_kernels.hpp"
#include <limits>
#include <cstring>

void gpuNUFFT::GpuNUFFTOperatorFactory::setBalanceWorkload(bool balanceWorkload)
{
  this->balanceWorkload = balanceWorkload;
}

IndType gpuNUFFT::GpuNUFFTOperatorFactory::computeSectorCountPerDimension(
    IndType dim, IndType sectorWidth)
{
  return (IndType)std::ceil(static_cast<DType>(dim) / sectorWidth);
}

template <typename T>
gpuNUFFT::Array<T>
gpuNUFFT::GpuNUFFTOperatorFactory::initLinArray(IndType arrCount)
{
  gpuNUFFT::Array<T> new_array;
  cudaMallocHost((void **)&new_array.data, arrCount * sizeof(T));
  new_array.dim.length = arrCount;
  return new_array;
}

gpuNUFFT::Dimensions
gpuNUFFT::GpuNUFFTOperatorFactory::computeSectorCountPerDimension(
    gpuNUFFT::Dimensions dim, IndType sectorWidth)
{
  gpuNUFFT::Dimensions sectorDims;
  sectorDims.width = computeSectorCountPerDimension(dim.width, sectorWidth);
  sectorDims.height = computeSectorCountPerDimension(dim.height, sectorWidth);
  sectorDims.depth = computeSectorCountPerDimension(dim.depth, sectorWidth);
  return sectorDims;
}

IndType gpuNUFFT::GpuNUFFTOperatorFactory::computeTotalSectorCount(
    gpuNUFFT::Dimensions dim, IndType sectorWidth)
{
  return computeSectorCountPerDimension(dim, sectorWidth).count();
}

template <typename T>
std::vector<gpuNUFFT::IndPair> gpuNUFFT::GpuNUFFTOperatorFactory::sortVector(
    gpuNUFFT::Array<T> assignedSectors, bool descending)
{
  std::vector<IndPair> secVector;

  for (IndType i = 0; i < assignedSectors.count(); i++)
    secVector.push_back(IndPair(i, assignedSectors.data[i]));

  // using function as comp
  if (descending)
    std::sort(secVector.begin(), secVector.end(), std::greater<IndPair>());
  else
    std::sort(secVector.begin(), secVector.end());

  return secVector;
}

void gpuNUFFT::GpuNUFFTOperatorFactory::computeProcessingOrder(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp)
{
  Array<IndType> sectorDataCount = gpuNUFFTOp->getSectorDataCount();
  std::vector<IndPair> countPerSector;

  for (unsigned i = 0; i < sectorDataCount.count() - 1; i++)
  {
    countPerSector.push_back(
        IndPair(i, sectorDataCount.data[i + 1] - sectorDataCount.data[i]));
  }

  std::sort(countPerSector.begin(), countPerSector.end(),
            std::greater<IndPair>());
  std::vector<IndType2> processingOrder;

  for (unsigned i = 0; i < countPerSector.size(); i++)
  {
    if (countPerSector[i].second > 0)
    {
      IndType2 tmp;
      tmp.x = countPerSector[i].first;
      tmp.y = 0;
      processingOrder.push_back(tmp);
      if (countPerSector[i].second > MAXIMUM_PAYLOAD)
      {
        int remaining = (int)countPerSector[i].second;
        IndType offset = 1;
        // split sector
        while ((remaining - MAXIMUM_PAYLOAD) > 0)
        {
          remaining -= MAXIMUM_PAYLOAD;
          IndType2 tmp_remain;
          tmp_remain.x = countPerSector[i].first;
          tmp_remain.y = (offset++) * MAXIMUM_PAYLOAD;
          processingOrder.push_back(tmp_remain);
        }
      }
    }
    else
      break;
  }

  Array<IndType2> sectorProcessingOrder =
      initSectorProcessingOrder(gpuNUFFTOp, (IndType)processingOrder.size());
  std::memcpy(sectorProcessingOrder.data, processingOrder.data(), processingOrder.size() * sizeof(IndType2));
  if (gpuNUFFTOp->getType() == gpuNUFFT::BALANCED)
    static_cast<BalancedGpuNUFFTOperator *>(gpuNUFFTOp)
        ->setSectorProcessingOrder(sectorProcessingOrder);
}

gpuNUFFT::Array<IndType> gpuNUFFT::GpuNUFFTOperatorFactory::assignSectors(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, gpuNUFFT::Array<DType> &kSpaceTraj)
{
  debug("in assign sectors\n");

  gpuNUFFTOp->setGridSectorDims(computeSectorCountPerDimension(
      gpuNUFFTOp->getGridDims(), gpuNUFFTOp->getSectorWidth()));

  IndType coordCnt = kSpaceTraj.count();

  // create temporary array to store assigned values
  gpuNUFFT::Array<IndType> assignedSectors;
  assignedSectors.data = (IndType *)malloc(coordCnt * sizeof(IndType));
  assignedSectors.dim.length = coordCnt;

  if (useGpu)
  {
    assignSectorsGPU(gpuNUFFTOp, kSpaceTraj, assignedSectors.data);
  }
  else
  {
    IndType sector;
    for (IndType cCnt = 0; cCnt < coordCnt; cCnt++)
    {
      if (gpuNUFFTOp->is2DProcessing())
      {
        DType2 coord;
        coord.x = kSpaceTraj.data[cCnt];
        coord.y = kSpaceTraj.data[cCnt + coordCnt];
        IndType2 mappedSector =
            computeSectorMapping(coord, gpuNUFFTOp->getGridDims(),
                                 (DType)gpuNUFFTOp->getSectorWidth());
        // linearize mapped sector
        sector = computeInd22Lin(mappedSector, gpuNUFFTOp->getGridSectorDims());
      }
      else
      {
        DType3 coord;
        coord.x = kSpaceTraj.data[cCnt];
        coord.y = kSpaceTraj.data[cCnt + coordCnt];
        coord.z = kSpaceTraj.data[cCnt + 2 * coordCnt];
        IndType3 mappedSector =
            computeSectorMapping(coord, gpuNUFFTOp->getGridDims(),
                                 (DType)gpuNUFFTOp->getSectorWidth());
        // linearize mapped sector
        sector = computeInd32Lin(mappedSector, gpuNUFFTOp->getGridSectorDims());
      }

      assignedSectors.data[cCnt] = sector;
    }
  }
  debug("finished assign sectors\n");
  return assignedSectors;
}

gpuNUFFT::Array<IndType>
gpuNUFFT::GpuNUFFTOperatorFactory::computeSectorDataCount(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp,
    gpuNUFFT::Array<IndType> assignedSectors,
    bool useLocalMemory)
{
  IndType cnt = 0;
  std::vector<IndType> dataCount;

  dataCount.push_back(0);
  for (IndType i = 0; i < gpuNUFFTOp->getGridSectorDims().count(); i++)
  {
    while (cnt < assignedSectors.count() && i == assignedSectors.data[cnt])
      cnt++;

    dataCount.push_back(cnt);
  }
  Array<IndType> sectorDataCount =
      useLocalMemory ? GpuNUFFTOperatorFactory::initSectorDataCount(gpuNUFFTOp, (IndType)dataCount.size()) : 
       initSectorDataCount(gpuNUFFTOp, (IndType)dataCount.size());
  std::memcpy(sectorDataCount.data, dataCount.data(), dataCount.size() * sizeof(IndType));
  return sectorDataCount;
}

inline IndType
gpuNUFFT::GpuNUFFTOperatorFactory::computeSectorCenter(IndType var,
                                                       IndType sectorWidth)
{
  return (IndType)(
      (int)var * (int)sectorWidth +
      (int)std::floor(static_cast<DType>(sectorWidth) / (DType)2.0));
}

void gpuNUFFT::GpuNUFFTOperatorFactory::debug(const std::string &message)
{
  if (DEBUG)
    std::cout << message << std::endl;
}

gpuNUFFT::Array<IndType>
gpuNUFFT::GpuNUFFTOperatorFactory::computeSectorCenters2D(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, bool useLocalMemory)
{
  gpuNUFFT::Dimensions sectorDims = gpuNUFFTOp->getGridSectorDims();
  IndType sectorWidth = gpuNUFFTOp->getSectorWidth();

  gpuNUFFT::Array<IndType> sectorCenters =
    useLocalMemory ? GpuNUFFTOperatorFactory::initSectorCenters(gpuNUFFTOp, sectorDims.count()) : initSectorCenters(gpuNUFFTOp, sectorDims.count());

  for (IndType y = 0; y < sectorDims.height; y++)
    for (IndType x = 0; x < sectorDims.width; x++)
    {
      IndType2 center;
      center.x = computeSectorCenter(x, sectorWidth);
      center.y = computeSectorCenter(y, sectorWidth);
      int index = computeXY2Lin((int)x, (int)y, sectorDims);
      sectorCenters.data[2 * index] = center.x;
      sectorCenters.data[2 * index + 1] = center.y;
    }
  return sectorCenters;
}

gpuNUFFT::Array<IndType>
gpuNUFFT::GpuNUFFTOperatorFactory::computeSectorCenters(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, bool useLocalMemory)
{
  gpuNUFFT::Dimensions sectorDims = gpuNUFFTOp->getGridSectorDims();
  IndType sectorWidth = gpuNUFFTOp->getSectorWidth();

  gpuNUFFT::Array<IndType> sectorCenters =
    useLocalMemory ? GpuNUFFTOperatorFactory::initSectorCenters(gpuNUFFTOp, sectorDims.count()) : initSectorCenters(gpuNUFFTOp, sectorDims.count());

  for (IndType z = 0; z < sectorDims.depth; z++)
    for (IndType y = 0; y < sectorDims.height; y++)
      for (IndType x = 0; x < sectorDims.width; x++)
      {
        IndType3 center;
        center.x = computeSectorCenter(x, sectorWidth);
        center.y = computeSectorCenter(y, sectorWidth);
        center.z = computeSectorCenter(z, sectorWidth);
        int index = computeXYZ2Lin((int)x, (int)y, (int)z, sectorDims);
        // necessary in order to avoid 2d or 3d typed array
        sectorCenters.data[3 * index] = center.x;
        sectorCenters.data[3 * index + 1] = center.y;
        sectorCenters.data[3 * index + 2] = center.z;
      }
  return sectorCenters;
}

// default implementation
gpuNUFFT::Array<IndType> gpuNUFFT::GpuNUFFTOperatorFactory::initDataIndices(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType coordCnt)
{
  return initLinArray<IndType>(coordCnt);
}

gpuNUFFT::Array<IndType> gpuNUFFT::GpuNUFFTOperatorFactory::initSectorDataCount(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType dataCount)
{
  return initLinArray<IndType>(dataCount);
}

gpuNUFFT::Array<IndType2>
gpuNUFFT::GpuNUFFTOperatorFactory::initSectorProcessingOrder(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType sectorCnt)
{
  return initLinArray<IndType2>(sectorCnt);
}

gpuNUFFT::Array<DType> gpuNUFFT::GpuNUFFTOperatorFactory::initDensData(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType coordCnt)
{
  return initLinArray<DType>(coordCnt);
}

gpuNUFFT::Array<DType> gpuNUFFT::GpuNUFFTOperatorFactory::initCoordsData(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType coordCnt)
{
  gpuNUFFT::Array<DType> coordsData =
      initLinArray<DType>(gpuNUFFTOp->getImageDimensionCount() * coordCnt);
  coordsData.dim.length = coordCnt;
  return coordsData;
}

gpuNUFFT::Array<IndType> gpuNUFFT::GpuNUFFTOperatorFactory::initSectorCenters(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType sectorCnt)
{
  // distinguish between 2d and 3d data
  return initLinArray<IndType>(gpuNUFFTOp->getImageDimensionCount() *
                               sectorCnt);
}

gpuNUFFT::Array<DType> gpuNUFFT::GpuNUFFTOperatorFactory::initDeapoData(
  IndType imgDimsCount)
{
  gpuNUFFT::Array<DType> deapoData =
    initLinArray<DType>(imgDimsCount);
  deapoData.dim.length = imgDimsCount;
  return deapoData;
}

gpuNUFFT::GpuNUFFTOperator *
gpuNUFFT::GpuNUFFTOperatorFactory::createNewGpuNUFFTOperator(
    IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims)
{
  if (balanceWorkload)
  {
    debug("creating Balanced GpuNUFFT Operator!\n");
    return new gpuNUFFT::BalancedGpuNUFFTOperator(kernelWidth, sectorWidth,
      osf, imgDims, this->matlabSharedMem);
  }

  debug("creating DEFAULT GpuNUFFT Operator!\n");
  return new gpuNUFFT::GpuNUFFTOperator(kernelWidth, sectorWidth, osf,
                                          imgDims, true, DEFAULT, true);
}

gpuNUFFT::Array<DType> gpuNUFFT::GpuNUFFTOperatorFactory::computeDeapodizationFunction(
  const IndType &kernelWidth, const DType &osf, gpuNUFFT::Dimensions &imgDims)
{
  debug("compute deapodization function\n");

  // Create simple gpuNUFFT Operator
  IndType sectorWidth = 8;
  gpuNUFFT::GpuNUFFTOperator *deapoGpuNUFFTOp;

  deapoGpuNUFFTOp = new gpuNUFFT::GpuNUFFTOperator(kernelWidth, sectorWidth, osf, imgDims);

  // Data
  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = (DType2*)calloc(1, sizeof(DType2)); // re + im
  dataArray.dim.length = 1;
  dataArray.data[0].x = 1;
  dataArray.data[0].y = 0;

  // Coord triplet (x,y,z)
  // should result in k-space center (0,0,0)
  gpuNUFFT::Array<DType> kSpaceTraj;
  if (deapoGpuNUFFTOp->is3DProcessing())
    kSpaceTraj.data = (DType*)calloc(3, sizeof(DType)); // x,y,z
  else
    kSpaceTraj.data = (DType*)calloc(2, sizeof(DType)); // x,y
  kSpaceTraj.dim.length = 1;
  deapoGpuNUFFTOp->setKSpaceTraj(kSpaceTraj);

  // assign according sector to k-Space position
  gpuNUFFT::Array<IndType> assignedSectors =
    assignSectors(deapoGpuNUFFTOp, kSpaceTraj);
  deapoGpuNUFFTOp->setSectorDataCount(
    computeSectorDataCount(deapoGpuNUFFTOp, assignedSectors, true));

  // only one data entry, data index = 0
  Array<IndType> dataIndices;
  dataIndices.data = (IndType*)calloc(1, sizeof(IndType));
  dataIndices.dim.length = 1;
  deapoGpuNUFFTOp->setDataIndices(dataIndices);

  // sector centers
  if (deapoGpuNUFFTOp->is3DProcessing())
    deapoGpuNUFFTOp->setSectorCenters(computeSectorCenters(deapoGpuNUFFTOp, true));
  else
    deapoGpuNUFFTOp->setSectorCenters(computeSectorCenters2D(deapoGpuNUFFTOp, true));
  debug("finished creation of gpuNUFFT operator for deapo computation\n");

  debug("compute deapodization\n");
  deapoGpuNUFFTOp->setDebugFunction(std::bind(&gpuNUFFT::GpuNUFFTOperatorFactory::debug, this, std::placeholders::_1));

  // Compute deapodization function by gridding of a single value positioned
  // in the center of k-space and by using the intended oversampling factor
  // and interpolation kernel width
  gpuNUFFT::Array<CufftType> deapoFunction =
    deapoGpuNUFFTOp->performGpuNUFFTAdj(dataArray,FFT);

  debug("finished deapo computation\n");

  // cleanup locally initialized arrays here
  free(dataArray.data);
  free(assignedSectors.data);

  // Compute abs values of deapo function and compensate
  // FFT scaling sqrt(N)
  Array<DType> deapoAbs = initDeapoData(deapoFunction.count());

  DType maxDeapoVal = 0;
  DType minDeapoVal = std::numeric_limits<DType>::max();
  double fft_scaling_factor = std::sqrt(deapoGpuNUFFTOp->getGridDims().count());

  for (unsigned cnt = 0; cnt < deapoFunction.count(); cnt++)
  {
    deapoFunction.data[cnt].x = static_cast<DType>(deapoFunction.data[cnt].x * fft_scaling_factor);
    deapoFunction.data[cnt].y = static_cast<DType>(deapoFunction.data[cnt].y * fft_scaling_factor);
    deapoAbs.data[cnt] = static_cast<DType>(1.0 / std::sqrt(std::pow(deapoFunction.data[cnt].x, 2.0) + std::pow(deapoFunction.data[cnt].y, 2.0)));
    if (deapoAbs.data[cnt] > maxDeapoVal)
      maxDeapoVal = deapoAbs.data[cnt];
    if (deapoAbs.data[cnt] < minDeapoVal)
      minDeapoVal = deapoAbs.data[cnt];
  }

  // cleanup
  delete deapoGpuNUFFTOp;
  free(deapoFunction.data);
  return deapoAbs;
}

gpuNUFFT::GpuNUFFTOperator *
gpuNUFFT::GpuNUFFTOperatorFactory::createGpuNUFFTOperator(
    gpuNUFFT::Array<DType> &kSpaceTraj, gpuNUFFT::Array<DType> &densCompData,
    gpuNUFFT::Array<DType2> &sensData, const IndType &kernelWidth,
    const IndType &sectorWidth, const DType &osf, gpuNUFFT::Dimensions &imgDims)
{
  // validate arguments
  checkMemoryConsumption(kSpaceTraj.dim, sectorWidth, osf, imgDims,
                         densCompData.dim, sensData.dim);

  if (imgDims.channels > 1)
    throw std::invalid_argument(
        "Image dimensions must not contain a channel size greater than 1!");

  debug("create gpuNUFFT operator...");

  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp =
      createNewGpuNUFFTOperator(kernelWidth, sectorWidth, osf, imgDims);

  // Set points and density compensation
  set_pts(gpuNUFFTOp, kSpaceTraj, densCompData);
  
  if (sensData.data != NULL)
    gpuNUFFTOp->setSens(sensData);
  
  gpuNUFFTOp->setDeapodizationFunction(
    this->computeDeapodizationFunction(kernelWidth, osf, imgDims));
    
  debug("finished creation of gpuNUFFT operator\n");
  
  return gpuNUFFTOp;
}


void gpuNUFFT::GpuNUFFTOperatorFactory::set_pts(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, gpuNUFFT::Array<DType> &kSpaceTraj,
    gpuNUFFT::Array<DType> &densCompData)
{
  if (kSpaceTraj.dim.channels > 1)
    throw std::invalid_argument(
        "Trajectory dimension must not contain a channel size greater than 1!");

  // assign according sector to k-Space position
  gpuNUFFT::Array<IndType> assignedSectors =
      assignSectors(gpuNUFFTOp, kSpaceTraj);

  // order the assigned sectors and memorize index
  std::vector<IndPair> assignedSectorsAndIndicesSorted =
      sortVector<IndType>(assignedSectors);

  IndType coordCnt = kSpaceTraj.dim.count();

  Array<DType> trajSorted = initCoordsData(gpuNUFFTOp, coordCnt);
  Array<IndType> dataIndices = initDataIndices(gpuNUFFTOp, coordCnt);

  Array<DType> densData;
  if (densCompData.data != NULL)
    densData = initDensData(gpuNUFFTOp, coordCnt);

  if (useGpu)
  {
    sortArrays(gpuNUFFTOp, assignedSectorsAndIndicesSorted,
               assignedSectors.data, dataIndices.data, kSpaceTraj,
               trajSorted.data, densCompData.data, densData.data);
  }
  else
  {
    // sort kspace data coords
    for (unsigned i = 0; i < coordCnt; i++)
    {
      trajSorted.data[i] =
          kSpaceTraj.data[assignedSectorsAndIndicesSorted[i].first];
      trajSorted.data[i + 1 * coordCnt] =
          kSpaceTraj
              .data[assignedSectorsAndIndicesSorted[i].first + 1 * coordCnt];
      if (gpuNUFFTOp->is3DProcessing())
        trajSorted.data[i + 2 * coordCnt] =
            kSpaceTraj
                .data[assignedSectorsAndIndicesSorted[i].first + 2 * coordCnt];

      // sort density compensation
      if (densCompData.data != NULL)
        densData.data[i] =
            densCompData.data[assignedSectorsAndIndicesSorted[i].first];

      dataIndices.data[i] = assignedSectorsAndIndicesSorted[i].first;
      assignedSectors.data[i] = assignedSectorsAndIndicesSorted[i].second;
    }
  }

  gpuNUFFTOp->setSectorDataCount(
      computeSectorDataCount(gpuNUFFTOp, assignedSectors));

  if (gpuNUFFTOp->getType() == gpuNUFFT::BALANCED)
  {
    computeProcessingOrder(gpuNUFFTOp);
  }

  gpuNUFFTOp->setDataIndices(dataIndices);

  gpuNUFFTOp->setKSpaceTraj(trajSorted);

  gpuNUFFTOp->setDens(densData);

  if (gpuNUFFTOp->is3DProcessing())
    gpuNUFFTOp->setSectorCenters(computeSectorCenters(gpuNUFFTOp));
  else
    gpuNUFFTOp->setSectorCenters(computeSectorCenters2D(gpuNUFFTOp));

  // free temporary array
  free(assignedSectors.data);
  assignedSectors.data = NULL;
}

gpuNUFFT::GpuNUFFTOperator *
gpuNUFFT::GpuNUFFTOperatorFactory::createGpuNUFFTOperator(
    gpuNUFFT::Array<DType> &kSpaceTraj, gpuNUFFT::Array<DType> &densCompData,
    const IndType &kernelWidth, const IndType &sectorWidth, const DType &osf,
    gpuNUFFT::Dimensions &imgDims)
{
  gpuNUFFT::Array<DType2> sensData;
  return createGpuNUFFTOperator(kSpaceTraj, densCompData, sensData, kernelWidth,
                                sectorWidth, osf, imgDims);
}

gpuNUFFT::GpuNUFFTOperator *
gpuNUFFT::GpuNUFFTOperatorFactory::createGpuNUFFTOperator(
    gpuNUFFT::Array<DType> &kSpaceTraj, const IndType &kernelWidth,
    const IndType &sectorWidth, const DType &osf, gpuNUFFT::Dimensions &imgDims)
{
  gpuNUFFT::Array<DType> densCompData;
  return createGpuNUFFTOperator(kSpaceTraj, densCompData, kernelWidth,
                                sectorWidth, osf, imgDims);
}

gpuNUFFT::GpuNUFFTOperator *
gpuNUFFT::GpuNUFFTOperatorFactory::loadPrecomputedGpuNUFFTOperator(
    gpuNUFFT::Array<DType> &kSpaceTraj, gpuNUFFT::Array<IndType> &dataIndices,
    gpuNUFFT::Array<IndType> &sectorDataCount,
    gpuNUFFT::Array<IndType2> &sectorProcessingOrder,
    gpuNUFFT::Array<IndType> &sectorCenters, gpuNUFFT::Array<DType2> &sensData,
    gpuNUFFT::Array<DType> &deapoData, const IndType &kernelWidth, 
    const IndType &sectorWidth, const DType &osf,
    gpuNUFFT::Dimensions &imgDims)
{
  GpuNUFFTOperator *gpuNUFFTOp =
      createNewGpuNUFFTOperator(kernelWidth, sectorWidth, osf, imgDims);
  gpuNUFFTOp->setGridSectorDims(
      GpuNUFFTOperatorFactory::computeSectorCountPerDimension(
          gpuNUFFTOp->getGridDims(), gpuNUFFTOp->getSectorWidth()));

  gpuNUFFTOp->setKSpaceTraj(kSpaceTraj);
  gpuNUFFTOp->setDataIndices(dataIndices);
  gpuNUFFTOp->setSectorDataCount(sectorDataCount);

  if (gpuNUFFTOp->getType() == gpuNUFFT::BALANCED)
    static_cast<BalancedGpuNUFFTOperator *>(gpuNUFFTOp)
        ->setSectorProcessingOrder(sectorProcessingOrder);

  gpuNUFFTOp->setSectorCenters(sectorCenters);
  gpuNUFFTOp->setSens(sensData);
  gpuNUFFTOp->setDeapodizationFunction(deapoData);
  return gpuNUFFTOp;
}

gpuNUFFT::GpuNUFFTOperator *
gpuNUFFT::GpuNUFFTOperatorFactory::loadPrecomputedGpuNUFFTOperator(
    gpuNUFFT::Array<DType> &kSpaceTraj, gpuNUFFT::Array<IndType> &dataIndices,
    gpuNUFFT::Array<IndType> &sectorDataCount,
    gpuNUFFT::Array<IndType2> &sectorProcessingOrder,
    gpuNUFFT::Array<IndType> &sectorCenters,
    gpuNUFFT::Array<DType> &densCompData, gpuNUFFT::Array<DType2> &sensData,
    gpuNUFFT::Array<DType> &deapoData, const IndType &kernelWidth,
    const IndType &sectorWidth, const DType &osf,
    gpuNUFFT::Dimensions &imgDims)
{
  GpuNUFFTOperator *gpuNUFFTOp = loadPrecomputedGpuNUFFTOperator(
      kSpaceTraj, dataIndices, sectorDataCount, sectorProcessingOrder,
      sectorCenters, sensData, deapoData, kernelWidth, sectorWidth, osf, imgDims);
  gpuNUFFTOp->setDens(densCompData);

  return gpuNUFFTOp;
}

void gpuNUFFT::GpuNUFFTOperatorFactory::checkMemoryConsumption(
    Dimensions &kSpaceDims, const IndType &sectorWidth, const DType &osf,
    Dimensions &imgDims, Dimensions &densDims, Dimensions &sensDims)
{
  size_t multiplier = sizeof(DType);
  size_t complexMultiplier = 2 * multiplier;

  size_t estMem = 2 * static_cast<size_t>(imgDims.count() * std::pow(osf, 3)) *
                  complexMultiplier;  //< oversampled grid, and fft
  estMem += kSpaceDims.count() *
            (3 * multiplier +
             complexMultiplier);  //< 3 components of trajectory + complex data

  estMem += densDims.count() * multiplier;  //<density compensation

  if (sensDims.count() > 0)
  {
    estMem += imgDims.count() *
              complexMultiplier;  //< only one coil is stored on the device
    estMem += imgDims.count() * complexMultiplier;  //< precomputed deapo
    estMem += imgDims.count() * complexMultiplier;  //< coil summation
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  size_t total = prop.totalGlobalMem;

  std::stringstream ss(
      "Required device memory too large for selected device!\n");
  ss << "Total available memory: " << total << std::endl;
  ss << "Required memory: " << estMem << std::endl;

  if (total < estMem)
    throw std::runtime_error(ss.str());
}
