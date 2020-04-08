#ifndef GPUNUFFT_TYPES_H_INCLUDED
#define GPUNUFFT_TYPES_H_INCLUDED

#include <cstdlib>
#include <iostream>
#include "config.hpp"

/**
 * @file
 * \brief Definition of gpuNUFFT defined types
 *
 */

/** \brief Default value of 1 if param is zero. */
#define DEFAULT_VALUE(a) ((a == 0) ? 1 : a)

/**
 * \brief Maximum allowed amount of data samples per sector if load balancing is
 *set to true
 *
 * @see gpuNUFFT::BalancedOperator
 */
#define MAXIMUM_PAYLOAD 256

/** \brief gpuNUFFT related classes
  */
namespace gpuNUFFT
{
/**\brief Pair implementation of IndType
 * Overloaded comparators in order to allow sorting
 * based on IndTypes.
*/
struct IndPair : std::pair<IndType, IndType>
{
  IndPair(IndType first, IndType second)
    : std::pair<IndType, IndType>(first, second)
  {
  }

  bool operator<(const IndPair &a) const
  {
    return this->second < a.second;
  }

  bool operator>(const IndPair &a) const
  {
    return this->second > a.second;
  }
};

/** \brief Data dimension struct
 *
 * Used as meta information of array data in gpuNUFFT::Array.
 *
 * Stores array length (1-d), width, height and depth (3-d),
 * channels (coils, 4-d) and frame (5-d) dimensions.
 *
 * Default values: width = 0,
 *                 height = 0,
 *                 depth = 0,
 *                 channels = 1,
 *                 frames = 1,
 *                 length = 0
 *
 * Warning: currently no distinction between one-dimensional (length)
 *          and higher-dimensional (width, height, depth) is made,
 *          thus ambiguity can occur in count() computation.
 */
struct Dimensions
{
  // TODO work on dimensions
  // avoid ambiguity between length (1D) and multidimensional case (2D/3D)

  Dimensions()
    : width(0), height(0), depth(0), channels(1), frames(1), length(0)
  {
  }

  Dimensions(IndType width, IndType height, IndType depth)
    : width(width), height(height), depth(depth), channels(1), frames(1),
      length(0)
  {
  }

  Dimensions(IndType width, IndType height)
    : width(width), height(height), depth(0), channels(1), frames(1), length(0)
  {
  }

  IndType width;
  IndType height;
  IndType depth;

  IndType channels;
  IndType frames;

  IndType length;  // 1D case

  /** \brief Compute total count of array length
   *
  */
  IndType count()
  {
    return DEFAULT_VALUE(length) * DEFAULT_VALUE(width) *
           DEFAULT_VALUE(height) * DEFAULT_VALUE(depth) *
           DEFAULT_VALUE(channels) * DEFAULT_VALUE(frames);
  }

  Dimensions operator*(const DType alpha)
  {
    Dimensions res;
    res.width = (IndType)((*this).width * alpha);
    res.height = (IndType)((*this).height * alpha);
    res.depth = (IndType)((*this).depth * alpha);
    res.length = (IndType)((*this).length * alpha);
    return res;
  }

  Dimensions operator+(const IndType alpha)
  {
    Dimensions res;
    if (this->width > 0)
      res.width = (IndType)((*this).width + alpha);
    if (this->height > 0)
      res.height = (IndType)((*this).height + alpha);
    if (this->depth > 0)
      res.depth = (IndType)((*this).depth + alpha);
    if (this->length > 0)
      res.length = (IndType)((*this).length + alpha);
    return res;
  }
};

/** \brief Array container consisting of raw data and gpuNUFFT::Dimensions
 *descriptor.
 *
 */
template <typename T> struct Array
{
  Array() : data(NULL)
  {
  }
  T *data;
  Dimensions dim;

  IndType count()
  {
    return dim.count();
  }
};

/** \brief Array container consisting of raw data, which is expected to be
 *allocated
 * on the GPU, and gpuNUFFT::Dimensions descriptor.
 *
 */
template <typename T> struct GpuArray : Array<T>
{
};

/** \brief Type used for texture interpolation on GPU
 *
 * The kernel used for gridding interpolation is loaded
 * in either a const array or in a 1-d, 2-d or 3-d texture.
 *
 */
enum InterpolationType
{
  /**\brief No texture lookup is performed */
  CONST_LOOKUP,
  /**\brief 1d texture fetch is performed */
  TEXTURE_LOOKUP,
  /**\brief 2d texture lookup is performed */
  TEXTURE2D_LOOKUP,
  /**\brief 3d texture lookup is performed */
  TEXTURE3D_LOOKUP
};

/** \brief Gridding step after which the processing is stopped.
 *
 * Neccessary for testing. Default value in the gridding adjoint operations
 * is DEAPODIZATION.
 */
enum GpuNUFFTOutput
{
  /** \brief Stop after convolution step. */
  CONVOLUTION,
  /** \brief Stop after iFFT step. */
  FFT,
  /** \brief Stop after deapodization step. */
  DEAPODIZATION,
  DENSITY_ESTIMATION,
};

/** \brief FFT shift direction enum. */
enum FFTShiftDir
{
  FORWARD,
  INVERSE
};

/** \brief Type enum of gpuNUFFT::GriddingOperator. */
enum OperatorType
{
  /** \brief Default Gridding Operator. Basic implementation. */
  DEFAULT,
  /** \brief Gridding Operator using Texture interpolation on GPU. */
  TEXTURE,
  /** \brief Gridding Operator using load balancing on GPU. */
  BALANCED,
  /** \brief Gridding Operator using load balancing and Texture interpolation on
     GPU. */
  BALANCED_TEXTURE
};

/** \brief Struct containing meta information of the current Gridding Problem.
  *Used in most GPU operations.
  *
  * Stores neccessary information required in most of the GPU gridding
  * and auxiliary processing steps.
  *
  * The GpuNUFFTInfo is loaded to a constant variable (GI) in the GPU
  * context.
  *
  * Initialization is performed in
  *gpuNUFFT::GpuNUFFTOperator::initGpuNUFFTInfo() function.
  *
  * @see gpuNUFFTOperator
  */
struct GpuNUFFTInfo
{
  /**\brief Total amount of data samples.*/
  int data_count;
  /**\brief Width in grid units of gridding kernel.*/
  int kernel_width;
  /**\brief Squared kernel_width.*/
  int kernel_widthSquared;
  /**\brief Reciprocal value of kernel_widthSquared.*/
  DType kernel_widthInvSquared;
  /**\brief Total amount of kernel entries.*/
  int kernel_count;
  /**\brief Radius of kernel relative to grid size.*/
  DType kernel_radius;

  /**\brief Width of oversampled grid.*/
  int grid_width_dim;
  /**\brief .*/
  int grid_width_offset;
  /**\brief Reciprocal value of grid_width_dim.*/
  DType3 grid_width_inv;

  /**\brief Total amount of image nodes.*/
  int im_width_dim;
  /**\brief Image offset (imgDims / 2).*/
  IndType3 im_width_offset;  // used in deapodization

  /**\brief Oversampling ratio.*/
  DType osr;

  /**\brief Total amount of sectors.*/
  int sector_count;
  /**\brief Amount of sectors per dimension.*/
  int sector_width;

  /**\brief Padded sector width (sector_width + kernel_width / 2).*/
  int sector_pad_width;
  /**\brief Maximum index per dimension of padded sector (sector_pad_width -
   * 1).*/
  int sector_pad_max;
  /**\brief Total amount of elements in one padded sector.*/
  int sector_dim;
  /**\brief Offset to zero position inside padded sector (sector_pad_width / 2).
   * Used in combination with the sector center in order to get to the starting
   * index (bottom left of the front slice)
   */
  int sector_offset;

  /**\brief Distance scale in x direction in case of anisotropic grids.*/
  DType aniso_x_scale;
  /**\brief Distance scale in y direction in case of anisotropic grids.*/
  DType aniso_y_scale;
  /**\brief Distance scale in z direction in case of anisotropic grids.*/
  DType aniso_z_scale;

  /**\brief Squared radius of kernel relative to the grid.*/
  DType radiusSquared;
  /**\brief Reciprocal radiusSquared.*/
  DType radiusSquared_inv;
  /**\brief Distance multiplier used for interpolation.*/
  DType dist_multiplier;

  /**\brief Image dimensions.*/
  IndType3 imgDims;
  /**\brief Total amount of image nodes.*/
  IndType imgDims_count;

  /**\brief Oversampled grid dimensions.*/
  IndType3 gridDims;
  /**\brief Total amount of grid nodes.*/
  IndType gridDims_count;

  /**\brief Flag to indicate whether 2-d or 3-d data is processed.*/
  bool is2Dprocessing;
  /**\brief Type used for texture interpolation.*/
  int interpolationType;
  /**\brief Total amount of sectors which have to be processed.
    * Depends on sector load balancing.*/
  int sectorsToProcess;
  /**\brief Number of coils processed concurrently */
  int n_coils_cc;
};
}

#endif  // GPUNUFFT_TYPES_H_INCLUDED
