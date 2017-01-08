#ifndef __MATLAB_HELPER_H
#define __MATLAB_HELPER_H

#include "mex.h"
#include "matrix.h"

#include "config.hpp"
#include "gpuNUFFT_operator.hpp"
#include "gpuNUFFT_types.hpp"

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

/**
* @file
* \brief Collection of helper functions to allow access to MATLAB data pointers.
*
*/

/** \brief Load MATLAB input array and check expected dimensions.
*
* Basic helper to convert MATLAB input array pointer to array of expected
*dimensions.
* If the input data pointer does not consist of the expected dimensions a mex
*error
* reported.
*
* @param prhs                MATLAB input array pointer
* @param input_index         Index to access inside prhs
* @param highest_varying_dim Highest varying dimension of data array, e.g. 2 for
*2xN array
* @param name                Name of parameter for debugging issues
* @param data                Output data pointer
* @param data_entries        Output amount of array elements
* @param max_nd              Maximum number of dimensions, 3 for MxNxO array
* @param n_coils             Output amount of coils
*/
template <typename TType>
void readMatlabInputArray(const mxArray *prhs[], int input_index,
                          int highest_varying_dim, const char *name,
                          TType **data, int *data_entries, int max_nd,
                          int *n_coils)
{
  int nd = (int)mxGetNumberOfDimensions(
      prhs[input_index]); /* get coordinate dimensions */

  if (MATLAB_DEBUG)
    mexPrintf("number of dims %d\n", nd);

  const mwSize *dims = mxGetDimensions(prhs[input_index]);
  *n_coils = 1;
  if (nd == 2)
  {
    if (highest_varying_dim > 0 &&
        dims[0] != (unsigned)highest_varying_dim)  // total: highest_varying_dim
                                                   // x N = 2
    {
      mexPrintf("dimensions of '%s' input array don't fit. Need to be %d x N "
                "but are %d x %d\n",
                name, highest_varying_dim, dims[0], dims[1]);
      mexErrMsgTxt("Error occured!\n");
    }
  }
  else if (max_nd == 3 && nd == 3)
  {
    // multiple coil data passed
    *n_coils = (int)dims[2];
    if (MATLAB_DEBUG)
      mexPrintf("number of coils %d\n", *n_coils);
  }
  else
  {
    mexPrintf(
        "dimensions of '%s' input array don't fit. Need to be %d x N but are ",
        name, highest_varying_dim);
    for (int i = 0; i < nd - 1; i++)
      mexPrintf(" %d x ", dims[i]);
    if (nd > 1)
      mexPrintf(" %d\n", dims[nd - 1]);
    else
      mexPrintf(" 0\n");
    mexErrMsgTxt("Error occured!\n");
  }

  if (MATLAB_DEBUG)
  {
    mexPrintf("%s dimensions: ", name);
    for (int i = 0; i < nd; i++)
      mexPrintf(" %d ", dims[i]);
    mexPrintf("\n");
  }

  *data_entries = (int)dims[1];

  const mxArray *matlabData;
  matlabData = prhs[input_index];
  bool is_int = false;

  if (mxIsInt32(matlabData) || mxIsUint32(matlabData) || mxIsUint64(matlabData))
  {
    is_int = true;
    *data = (TType *)mxGetData(matlabData);
  }
  else
  {
    *data = (TType *)mxGetPr(matlabData);
  }
  if (MATLAB_DEBUG)
  {
    for (int i = 0; i < MIN((highest_varying_dim * (*data_entries)), 100);
         i++)  // re, im
      // Debug out does not work with complex numbers
      if (is_int)
        mexPrintf("%s: %d, ", name, (*data)[i]);
      else
        mexPrintf("%s: %f, ", name, (*data)[i]);

    mexPrintf("\n");
  }
}

/** \brief Load MATLAB input array and check expected dimensions.
*
* Basic helper to convert MATLAB input array pointer to array of expected
*dimensions.
*
* @param prhs                MATLAB input array pointer
* @param input_index         Index to access inside prhs
* @param highest_varying_dim Highest varying dimension of data array, e.g. 2 for
*2xN array
* @param name                Name of parameter for debugging issues
* @param data                Output data pointer
* @param data_entries        Output amount of array elements
*/
template <typename TType>
void readMatlabInputArray(const mxArray *prhs[], int input_index,
                          int highest_varying_dim, const char *name,
                          TType **data, int *data_entries)
{
  int dummy;
  readMatlabInputArray<TType>(prhs, input_index, highest_varying_dim, name,
                              data, data_entries, 2, &dummy);
}

/** \brief Load MATLAB input array and return gpuNUFFT::Array
*
* The MATLAB data pointer is only checked by the highest varying dimension.
* The Dimensions of the returned Array are only defined by the total length of
* array data.
*
* @param prhs                MATLAB input array pointer
* @param input_index         Index to access inside prhs
* @param highest_varying_dim Highest varying dimension of data array, e.g. 2 for
*2xN array
* @param name                Name of parameter for debugging issues
* @return New created gpuNUFFT::Array referencing to the passed MATLAB data
*
* @see gpuNUFFT::Dimensions
*/
template <typename TType>
gpuNUFFT::Array<TType>
readAndCreateArray(const mxArray *prhs[], int input_index,
                   int highest_varying_dim, const char *name)
{
  TType *data = NULL;
  int data_count;
  readMatlabInputArray<TType>(prhs, input_index, highest_varying_dim, name,
                              &data, &data_count);

  gpuNUFFT::Array<TType> dataArray;
  dataArray.data = (TType *)data;
  dataArray.dim.length = data_count;
  return dataArray;
}

/** \brief Helper function to load a parameter from a MATLAB parameter struct
*
* @param params    Raw array pointer to the parameter struct array
* @param fieldname Name of the parameter that has to be read out
* @return value of parameter
*/
template <typename TType>
inline TType getParamField(const mxArray *params, const char *fieldname)
{
  const mxArray *data = mxGetField(params, 0, fieldname);
  if (mxIsInt32(data))
  {
    return (TType)(((TType *)mxGetData(data))[0]);
  }
  else
  {
    return (TType)(((TType *)mxGetPr(data))[0]);
  }
}

/** \brief Helper method to evaluate the size of a parameter field in a MATLAB
*struct
*
* @param params    Raw array pointer to the parameter struct array
* @param fieldname Name of the parameter that has to be read out
* @return gpuNUFFT::Dimensions of parameter field
*/
inline gpuNUFFT::Dimensions getDimensionsFromParamField(const mxArray *params,
                                                        const char *fieldname)
{
  const mxArray *data = mxGetField(params, 0, fieldname);
  gpuNUFFT::Dimensions dim;
  int *dims = (int *)mxGetData(data);
  dim.width = dims[0];
  dim.height = dims[1];
  dim.depth = dims[2];
  return dim;
}

/** \brief Helper method to map the Interpolation Type parameter from int to the
*gpuNUFFT::InterpolationType enum
*
* @param fieldname Name of the parameter that has to be read out
* @return gpuNUFFT::InterpolationType
*/
inline gpuNUFFT::InterpolationType getInterpolationTypeOf(int val)
{
  switch (val)
  {
  case 1:
    return gpuNUFFT::TEXTURE_LOOKUP;
  case 2:
    return gpuNUFFT::TEXTURE2D_LOOKUP;
  case 3:
    return gpuNUFFT::TEXTURE3D_LOOKUP;
  default:
    return gpuNUFFT::CONST_LOOKUP;
  }
}

#endif  // __MATLAB_HELPER_H
