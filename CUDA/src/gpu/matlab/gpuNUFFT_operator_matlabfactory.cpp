#include "gpuNUFFT_operator_matlabfactory.hpp"
#include <iostream>

#define INDTYPE_MATLAB                                                         \
  ((sizeof(IndType) == 8) ? mxUINT64_CLASS : mxUINT32_CLASS)

void gpuNUFFT::GpuNUFFTOperatorMatlabFactory::debug(const std::string &message)
{
  if (MATLAB_DEBUG)
    mexPrintf(message.c_str());
}

mxArray *createIndicesArray(const IndType arrSize)
{
  mwSize indSize[] = { 1, arrSize };
  return mxCreateNumericArray(2, indSize, INDTYPE_MATLAB, mxREAL);
}

mxArray *createSectorDataArray(const IndType arrSize)
{
  mwSize secSize[] = { 1, arrSize };
  return mxCreateNumericArray(2, secSize, INDTYPE_MATLAB, mxREAL);
}

mxArray *createDensArray(const IndType arrSize)
{
  mwSize densSize[] = { 1, arrSize };  // scaling factor
  return mxCreateNumericArray(2, densSize, mxSINGLE_CLASS, mxREAL);
}

mxArray *createCoordsArray(const IndType arrSize, const IndType nDim)
{
  mwSize coordSize[] = { nDim, arrSize };  // row major in matlab, SoA (DType3
                                           // or DType2)
  return mxCreateNumericArray(2, coordSize, mxSINGLE_CLASS, mxREAL);
}

mxArray *createSectorCentersArray(const IndType arrSize, const IndType nDim)
{
  mwSize secSize[] = { nDim, arrSize };  // IndType3 or IndType2
  return mxCreateNumericArray(2, secSize, INDTYPE_MATLAB, mxREAL);
}

mxArray *createDeapoArray(const IndType arrSize)
{
  mwSize deapoSize[] = { 1, arrSize };  // scaling factor
  return mxCreateNumericArray(2, deapoSize, mxSINGLE_CLASS, mxREAL);
}

gpuNUFFT::Array<IndType>
gpuNUFFT::GpuNUFFTOperatorMatlabFactory::initDataIndices(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType coordCnt)
{
  if (MATLAB_DEBUG)
    mexPrintf("init Data Index Output Array: %d\n", coordCnt);

  plhs[0] = createIndicesArray(coordCnt);

  gpuNUFFT::Array<IndType> dataIndices;
  dataIndices.data = (IndType *)mxGetData(plhs[0]);
  dataIndices.dim.length = coordCnt;

  if (dataIndices.data == NULL)
    mexErrMsgTxt("Could not create output mxArray.\n");

  return dataIndices;
}

gpuNUFFT::Array<DType> gpuNUFFT::GpuNUFFTOperatorMatlabFactory::initCoordsData(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType coordCnt)
{
  if (MATLAB_DEBUG)
    mexPrintf("init Coords Output Array: %d\n", coordCnt);

  plhs[3] = createCoordsArray(coordCnt, gpuNUFFTOp->getImageDimensionCount());

  gpuNUFFT::Array<DType> coordsData;
  coordsData.data = (DType *)mxGetData(plhs[3]);
  coordsData.dim.length = coordCnt;
  return coordsData;
}

gpuNUFFT::Array<IndType>
gpuNUFFT::GpuNUFFTOperatorMatlabFactory::initSectorCenters(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType sectorCnt)
{
  if (MATLAB_DEBUG)
    mexPrintf("init Sector Centers Output Array: %d\n", sectorCnt);
  plhs[4] =
      createSectorCentersArray(sectorCnt, gpuNUFFTOp->getImageDimensionCount());

  gpuNUFFT::Array<IndType> sectorCenters;
  sectorCenters.data = (IndType *)mxGetData(plhs[4]);
  sectorCenters.dim.length = sectorCnt;
  return sectorCenters;
}

gpuNUFFT::Array<IndType>
gpuNUFFT::GpuNUFFTOperatorMatlabFactory::initSectorDataCount(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType dataCount)
{
  if (MATLAB_DEBUG)
    mexPrintf("init SectorData Output Array: %d\n", dataCount);
  plhs[1] = createSectorDataArray(dataCount);

  Array<IndType> sectorDataCount;
  sectorDataCount.data = (IndType *)mxGetData(plhs[1]);
  sectorDataCount.dim.length = dataCount;
  return sectorDataCount;
}
gpuNUFFT::Array<IndType2>
gpuNUFFT::GpuNUFFTOperatorMatlabFactory::initSectorProcessingOrder(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType sectorCnt)
{
  if (MATLAB_DEBUG)
    mexPrintf("init SectorProcessingOrder Output Array: %d\n", sectorCnt);
  plhs[5] = createSectorCentersArray(sectorCnt, 2);

  Array<IndType2> sectorProcessingOrder;
  sectorProcessingOrder.data = (IndType2 *)mxGetData(plhs[5]);
  sectorProcessingOrder.dim.length = sectorCnt;
  return sectorProcessingOrder;
}

gpuNUFFT::Array<DType> gpuNUFFT::GpuNUFFTOperatorMatlabFactory::initDensData(
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp, IndType coordCnt)
{
  if (MATLAB_DEBUG)
    mexPrintf("init Dens Output Array: %d\n", coordCnt);
  plhs[2] = createDensArray(coordCnt);

  Array<DType> densData;
  densData.data = (DType *)mxGetData(plhs[2]);
  densData.dim.length = coordCnt;

  return densData;
}

gpuNUFFT::Array<DType> gpuNUFFT::GpuNUFFTOperatorMatlabFactory::initDeapoData(
  IndType imgDimsCount)
{
  if (MATLAB_DEBUG)
    mexPrintf("init Deapo Output Array: %d\n", imgDimsCount);
  plhs[6] = createDeapoArray(imgDimsCount);

  Array<DType> deapoData;
  deapoData.data = (DType *)mxGetData(plhs[6]);
  deapoData.dim.length = imgDimsCount;

  return deapoData;
}

gpuNUFFT::GpuNUFFTOperator *
gpuNUFFT::GpuNUFFTOperatorMatlabFactory::createGpuNUFFTOperator(
    gpuNUFFT::Array<DType> &kSpaceTraj, Array<DType> &densCompData,
    Array<DType2> &sensData, const IndType &kernelWidth,
    const IndType &sectorWidth, const DType &osf, gpuNUFFT::Dimensions &imgDims,
    mxArray *plhs[])
{
  if (MATLAB_DEBUG)
    mexPrintf("Start init of GpuNUFFT Operator\n");

  this->plhs = plhs;

  GpuNUFFTOperator *gpuNUFFTOp =
      GpuNUFFTOperatorFactory::createGpuNUFFTOperator(
          kSpaceTraj, densCompData, sensData, kernelWidth, sectorWidth, osf,
          imgDims);

  if (!gpuNUFFTOp->applyDensComp())
  {
    if (MATLAB_DEBUG)
      mexPrintf("returning empty dens array\n");
    plhs[2] = createDensArray(1);
  }
  if (gpuNUFFTOp->getType() != gpuNUFFT::BALANCED &&
      gpuNUFFTOp->getType() != gpuNUFFT::BALANCED_TEXTURE)
  {
    if (MATLAB_DEBUG)
      mexPrintf("returning empty processing order array\n");
    plhs[5] = createSectorDataArray(1);
  }

  return gpuNUFFTOp;
}
