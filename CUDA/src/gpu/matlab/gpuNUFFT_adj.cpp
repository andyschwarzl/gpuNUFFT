#include "matlab_helper.h"

#include "mex.h"
#include "matrix.h"

#include <complex>
#include <vector>

// GRIDDING 3D
#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h>
#include <cublas.h>

#include <stdio.h>
#include <string>
#include <iostream>

#include <string.h>

#include "gpuNUFFT_operator_matlabfactory.hpp"

/**
 * \brief MEX file cleanup to reset CUDA Device
**/
void cleanUp()
{
  cudaDeviceReset();
}

/**
  * @file
  * \brief MATLAB Wrapper for adjoint (NUFFT^H) Operation
  */

/**
  * \brief MATLAB Wrapper for adjoint (NUFFT^H) Operation
  *
  * Loads and converts all parameters arrays and passes
  * them to the MatlabGpuNUFFTOperatorFactory in order to
  * obtain an operator instance.
  *
  * From MATLAB doc:
  * Arguments
  * nlhs Number of expected output mxArrays
  * plhs Array of pointers to the expected output mxArrays
  * nrhs Number of input mxArrays
  * prhs Array of pointers to the input mxArrays. Do not modify any prhs values
  *in your MEX-file. Changing the data in these read-only mxArrays can produce
  *undesired side effects.
  */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (MATLAB_DEBUG)
    mexPrintf("Starting ADJOINT gpuNUFFT Function...\n");

  // get cuda context associated to MATLAB
  int cuDevice = 0;
  cudaGetDevice(&cuDevice);
  cudaSetDevice(cuDevice);

  mexAtExit(cleanUp);

  // fetch data from MATLAB
  int pcount = 0;  // param counter

  // Input: K-space Data
  DType2 *data = NULL;
  int data_count;
  int n_coils;
  readMatlabInputArray<DType2>(prhs, pcount++, 2, "data", &data, &data_count, 3,
                               &n_coils);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_count;
  dataArray.dim.channels = n_coils;

  // Data indices
  gpuNUFFT::Array<IndType> dataIndicesArray =
      readAndCreateArray<IndType>(prhs, pcount++, 0, "data-indices");

  // Coords
  gpuNUFFT::Array<DType> kSpaceTraj =
      readAndCreateArray<DType>(prhs, pcount++, 0, "coords");

  // SectorData Count
  gpuNUFFT::Array<IndType> sectorDataCountArray =
      readAndCreateArray<IndType>(prhs, pcount++, 0, "sector-data-count");

  // Sector Processing Order
  gpuNUFFT::Array<IndType2> sectorProcessingOrderArray =
      readAndCreateArray<IndType2>(prhs, pcount++, 0,
                                   "sector-processing-order");

  // Sector centers
  gpuNUFFT::Array<IndType> sectorCentersArray =
      readAndCreateArray<IndType>(prhs, pcount++, 0, "sector-centers");

  // Density compensation
  gpuNUFFT::Array<DType> density_compArray =
      readAndCreateArray<DType>(prhs, pcount++, 0, "density-comp");

  // Sens array	- same dimension as image data
  DType2 *sensData = NULL;
  int n_coils_sens, sens_count;
  readMatlabInputArray<DType2>(prhs, pcount++, 0, "sens-data", &sensData,
                               &sens_count, 3, &n_coils_sens);

  // Deapo function
  gpuNUFFT::Array<DType> deapoFunctionArray =
    readAndCreateArray<DType>(prhs, pcount++, 0, "deapo-function");

  // Parameters
  const mxArray *matParams = prhs[pcount++];

  if (!mxIsStruct(matParams))
    mexErrMsgTxt("expects struct containing parameters!");

  gpuNUFFT::Dimensions imgDims =
      getDimensionsFromParamField(matParams, "img_dims");
  DType osr = getParamField<DType>(matParams, "osr");
  int kernel_width = getParamField<int>(matParams, "kernel_width");
  int sector_width = getParamField<int>(matParams, "sector_width");
  int traj_length = getParamField<int>(matParams, "trajectory_length");
  bool use_textures = getParamField<bool>(matParams, "use_textures");
  bool balance_workload = getParamField<bool>(matParams, "balance_workload");

  if (MATLAB_DEBUG)
  {
    mexPrintf("data indices count: %d\n", dataIndicesArray.count());
    mexPrintf("coords count: %d\n", kSpaceTraj.count());
    mexPrintf("sector data count: %d\n", sectorDataCountArray.count());
    mexPrintf("centers count: %d\n", sectorCentersArray.count());
    mexPrintf("dens count: %d\n", density_compArray.count());
    mexPrintf("sens coils: %d, img coils: %d\n", n_coils_sens, n_coils);

    mexPrintf("passed Params, IM_WIDTH: [%d,%d,%d], OSR: %f, KERNEL_WIDTH: %d, "
              "SECTOR_WIDTH: %d dens_count: %d traj_len: %d n coils: %d, use "
              "textures: %d\n",
              imgDims.width, imgDims.height, imgDims.depth, osr, kernel_width,
              sector_width, density_compArray.count(), traj_length, n_coils,
              use_textures);
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    mexPrintf("memory usage on device, free: %lu total: %lu\n", free_mem,
              total_mem);
  }

  // complete Sens Array
  gpuNUFFT::Array<DType2> sensArray;
  sensArray.data = sensData;
  sensArray.dim = imgDims;
  sensArray.dim.channels = n_coils_sens;

  // Allocate Output: Image
  CufftType *imdata = NULL;
  const mwSize n_dims = 5;  // 2 * w * h * d * ncoils, 2 -> Re + Im

  mwSize dims_im[n_dims];
  mwSize n_dims_curr = 0;
  dims_im[n_dims_curr++] = (mwSize)2; /* complex */
  dims_im[n_dims_curr++] = (mwSize)imgDims.width;
  dims_im[n_dims_curr++] = (mwSize)imgDims.height;

  // check if 2d or 3d data has to be allocated
  if (imgDims.depth > 1)
    dims_im[n_dims_curr++] = (mwSize)imgDims.depth;

  // if sens data is present a summation over all
  // coils is performed automatically
  // thus only one coil has to be allocated
  bool applySensData = sens_count > 1;

  if (!applySensData)
    dims_im[n_dims_curr++] = (mwSize)n_coils;

  plhs[0] = mxCreateNumericArray(n_dims_curr, dims_im, mxSINGLE_CLASS, mxREAL);

  imdata = (CufftType *)mxGetData(plhs[0]);
  if (imdata == NULL)
    mexErrMsgTxt("Could not create output mxArray.\n");

  gpuNUFFT::Array<DType2> imdataArray;
  imdataArray.data = imdata;
  imdataArray.dim = imgDims;
  imdataArray.dim.channels = applySensData ? 1 : n_coils;

  if (MATLAB_DEBUG)
  {
    mexPrintf(" imdataArray dims %d,%d,%d,%d\n", imdataArray.dim.width,
              imdataArray.dim.height, imdataArray.dim.depth,
              imdataArray.dim.channels);
    mexPrintf(" data count: %d \n", data_count);
  }

  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = NULL;
  try
  {
    gpuNUFFT::GpuNUFFTOperatorMatlabFactory gpuNUFFTFactory(use_textures, true,
                                                            balance_workload);
    gpuNUFFTOp = gpuNUFFTFactory.loadPrecomputedGpuNUFFTOperator(
        kSpaceTraj, dataIndicesArray, sectorDataCountArray,
        sectorProcessingOrderArray, sectorCentersArray, density_compArray,
        sensArray, deapoFunctionArray, kernel_width, sector_width, osr, imgDims);

    if (MATLAB_DEBUG)
      mexPrintf("Creating gpuNUFFT Operator of Type: %d \n",
                gpuNUFFTOp->getType());

    gpuNUFFTOp->performGpuNUFFTAdj(dataArray, imdataArray);

    delete gpuNUFFTOp;
  }
  catch (std::exception &e)
  {
    delete gpuNUFFTOp;
    mexErrMsgIdAndTxt("gpuNUFFT:adjoint", "FAILURE in gpuNUFFT operation: %s\n",
                      e.what());
  }

  cudaDeviceSynchronize();
  if (MATLAB_DEBUG)
  {
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    mexPrintf("memory usage on device afterwards, free: %lu total: %lu\n",
              free_mem, total_mem);
  }
}

