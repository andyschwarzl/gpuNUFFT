#ifndef GPUNUFFT_KERNELS_H
#define GPUNUFFT_KERNELS_H
#include "gpuNUFFT_utils.hpp"
#include "cuda_utils.hpp"

/**
 * @file
 * \brief gpuNUFFT CUDA function prototypes
 */

// ADJOINT Operations

/**
 * \brief Adjoint gridding convolution implementation on GPU.
 *
 * Performs the adjoint gridding convolution step on the GPU, thus the
 *interpolation
 * from non-uniform sampled k-space data onto the uniform oversampled grid.
 *
 * The distance from each sample to its neighboring grid positions is computed
 *and the corresponding
 * data value is weighted by the kernel function according to the distance.
 *
 * CUDA function prototype.
 *
 *
 * @param data_d            Input k-space sample data value, complex, sorted due
 *to precomputation
 * @param crds_d            k-space sample coordinate (non-cartesian),
 *linearized array (x1,x2,x3,...,xn,y1,y2,y3,...,yn,z1,z2,z3,...zn)
 * @param gdata_d           Output k-space grid (cartesian)
 * @param kernel_d          precomputed interpolation kernel
 * @param sectors_d         precomputed data-sector mapping, defines the range
 *of data elements per sector, e.g. 0,3,4,4,10 -> maps data points 0..3 to
 *sector id 0, 3..4 to sector 1, no data point to sector 2, 4..10 to sector 3 an
 *so on
 * @param sector_centers_d  precomputed coordinates (x,y,z) of sector centers
 * @param gi_host           info struct with meta information
 */
void performConvolution(DType2 *data_d, DType *crds_d, CufftType *gdata_d,
                        DType *kernel_d, IndType *sectors_d,
                        IndType *sector_centers_d,
                        gpuNUFFT::GpuNUFFTInfo *gi_host);

/**
 * \brief Adjoint gridding convolution implementation on GPU using sector load
 *balancing.
 *
 * Performs the adjoint gridding convolution step on the GPU, thus the
 *interpolation
 * from non-uniform sampled k-space data onto the uniform oversampled grid.
 *
 * The distance from each sample to its neighboring grid positions is computed
 *and the corresponding
 * data value is weighted by the kernel function according to the distance.
 *
 * In order to balance the work load per thread block a sector processing order
 *is precomputed.
 *
 * CUDA function prototype.
 *
 *
 * @param data_d            Input k-space sample data value, complex, sorted due
 *to precomputation
 * @param crds_d            k-space sample coordinate (non-cartesian),
 *linearized array (x1,x2,x3,...,xn,y1,y2,y3,...,yn,z1,z2,z3,...zn)
 * @param gdata_d           Output k-space grid (cartesian)
 * @param kernel_d          precomputed interpolation kernel
 * @param sectors_d         precomputed data-sector mapping, defines the range
 *of data elements per sector, e.g. 0,3,4,4,10 -> maps data points 0..3 to
 *sector id 0, 3..4 to sector 1, no data point to sector 2, 4..10 to sector 3 an
 *so on
 * @param sector_processing_order_d precomputed sector processing order
 * @param sector_centers_d  precomputed coordinates (x,y,z) of sector centers
 * @param gi_host           info struct with meta information
 */
void performConvolution(DType2 *data_d, DType *crds_d, CufftType *gdata_d,
                        DType *kernel_d, IndType *sectors_d,
                        IndType2 *sector_processing_order_d,
                        IndType *sector_centers_d,
                        gpuNUFFT::GpuNUFFTInfo *gi_host);

/**
 * \brief Adjoint gridding convolution implementation on GPU using textures for
 *kernel lookup.
 *
 * Performs the adjoint gridding convolution step on the GPU, thus the
 *interpolation
 * from non-uniform sampled k-space data onto the uniform oversampled grid.
 *
 * The distance from each sample to its neighboring grid positions is computed
 *and the corresponding
 * data value is weighted by the kernel function according to the distance.
 *
 * The kernel lookup is performed by the use of gpu textures.
 *
 * CUDA function prototype.
 *
 *
 * @param data_d            Input k-space sample data value, complex, sorted due
 *to precomputation
 * @param crds_d            k-space sample coordinate (non-cartesian),
 *linearized array (x1,x2,x3,...,xn,y1,y2,y3,...,yn,z1,z2,z3,...zn)
 * @param gdata_d           Outpu k-space grid (cartesian)
 * @param kernel_d          precomputed interpolation kernel
 * @param sectors_d         precomputed data-sector mapping, defines the range
 *of data elements per sector, e.g. 0,3,4,4,10 -> maps data points 0..3 to
 *sector id 0, 3..4 to sector 1, no data point to sector 2, 4..10 to sector 3 an
 *so on
 * @param sector_processing_order_d precomputed sector processing order
 * @param sector_centers_d  precomputed coordinates (x,y,z) of sector centers
 * @param gi_host           info struct with meta information
 */
void performTextureConvolution(DType2 *data_d, DType *crds_d,
                               CufftType *gdata_d, DType *kernel_d,
                               IndType *sectors_d, IndType *sector_centers_d,
                               gpuNUFFT::GpuNUFFTInfo *gi_host);

/**
 * \brief Adjoint gridding convolution implementation on GPU using textures and
 *sector load balancing.
 *
 * Performs the adjoint gridding convolution step on the GPU, thus the
 *interpolation
 * from non-uniform sampled k-space data onto the uniform oversampled grid.
 *
 * The distance from each sample to its neighboring grid positions is computed
 *and the corresponding
 * data value is weighted by the kernel function according to the distance.
 *
 * The kernel lookup is performed by the use of gpu textures and the workload is
 *balanced.
 *
 * CUDA function prototype.
 *
 * @param data_d            Input k-space sample data value, complex, sorted due
 *to precomputation
 * @param crds_d            k-space sample coordinate (non-cartesian),
 *linearized array (x1,x2,x3,...,xn,y1,y2,y3,...,yn,z1,z2,z3,...zn)
 * @param gdata_d           Output k-space grid (cartesian)
 * @param kernel_d          precomputed interpolation kernel
 * @param sectors_d         precomputed data-sector mapping, defines the range
 *of data elements per sector, e.g. 0,3,4,4,10 -> maps data points 0..3 to
 *sector id 0, 3..4 to sector 1, no data point to sector 2, 4..10 to sector 3 an
 *so on
 * @param sector_processing_order_d precomputed sector processing order
 * @param sector_centers_d  precomputed coordinates (x,y,z) of sector centers
 * @param gi_host           info struct with meta information
 */
void performTextureConvolution(DType2 *data_d, DType *crds_d,
                               CufftType *gdata_d, DType *kernel_d,
                               IndType *sectors_d,
                               IndType2 *sector_processing_order_d,
                               IndType *sector_centers_d,
                               gpuNUFFT::GpuNUFFTInfo *gi_host);

// FORWARD Operations

/**
 * \brief Forward gridding convolution implementation on GPU.
 *
 * Performs the forward gridding convolution step on the GPU, thus the
 *interpolation
 * from uniform oversampled grid positions to non-uniform sampled k-space data
 *points.
 *
 * The distance from each sample to its neighboring grid positions is computed
 *and the corresponding
 * data value is weighted by the kernel function according to the distance.
 *
 * CUDA function prototype.
 *
 * @param data_d            Output k-space sample data value, complex, sorted
 *due to precomputation
 * @param crds_d            k-space sample coordinate (non-cartesian),
 *linearized array (x1,x2,x3,...,xn,y1,y2,y3,...,yn,z1,z2,z3,...zn)
 * @param gdata_d           Input k-space grid (cartesian)
 * @param kernel_d          precomputed interpolation kernel
 * @param sectors_d         precomputed data-sector mapping, defines the range
 *of data elements per sector, e.g. 0,3,4,4,10 -> maps data points 0..3 to
 *sector id 0, 3..4 to sector 1, no data point to sector 2, 4..10 to sector 3 an
 *so on
 * @param sector_processing_order_d precomputed sector processing order
 * @param sector_centers_d  precomputed coordinates (x,y,z) of sector centers
 * @param gi_host           info struct with meta information
 */
void performForwardConvolution(CufftType *data_d, DType *crds_d,
                               CufftType *gdata_d, DType *kernel_d,
                               IndType *sectors_d, IndType *sector_centers_d,
                               gpuNUFFT::GpuNUFFTInfo *gi_host);

/**
 * \brief Forward gridding convolution implementation on GPU using sector load
 *balancing.
 *
 * Performs the forward gridding convolution step on the GPU, thus the
 *interpolation
 * from uniform oversampled grid positions to non-uniform sampled k-space data
 *points.
 *
 * The distance from each sample to its neighboring grid positions is computed
 *and the corresponding
 * data value is weighted by the kernel function according to the distance.
 *
 * In order to balance the work load per thread block a sector processing order
 *is precomputed.
 *
 * CUDA function prototype.
 *
 * @param data_d            Output k-space sample data value, complex, sorted
 *due to precomputation
 * @param crds_d            k-space sample coordinate (non-cartesian),
 *linearized array (x1,x2,x3,...,xn,y1,y2,y3,...,yn,z1,z2,z3,...zn)
 * @param gdata_d           Input k-space grid (cartesian)
 * @param kernel_d          precomputed interpolation kernel
 * @param sectors_d         precomputed data-sector mapping, defines the range
 *of data elements per sector, e.g. 0,3,4,4,10 -> maps data points 0..3 to
 *sector id 0, 3..4 to sector 1, no data point to sector 2, 4..10 to sector 3 an
 *so on
 * @param sector_processing_order_d precomputed sector processing order
 * @param sector_centers_d  precomputed coordinates (x,y,z) of sector centers
 * @param gi_host           info struct with meta information
 */
void performForwardConvolution(CufftType *data_d, DType *crds_d,
                               CufftType *gdata_d, DType *kernel_d,
                               IndType *sectors_d,
                               IndType2 *sector_processing_order_d,
                               IndType *sector_centers_d,
                               gpuNUFFT::GpuNUFFTInfo *gi_host);

/**
 * \brief Forward gridding convolution implementation on GPU using textures .
 *
 * Performs the forward gridding convolution step on the GPU, thus the
 *interpolation
 * from uniform oversampled grid positions to non-uniform sampled k-space data
 *points.
 *
 * The distance from each sample to its neighboring grid positions is computed
 *and the corresponding
 * data value is weighted by the kernel function according to the distance.
 *
 * The kernel lookup is performed by the use of gpu textures.
 *
 * CUDA function prototype.
 *
 * @param data_d            Output k-space sample data value, complex, sorted
 *due to precomputation
 * @param crds_d            k-space sample coordinate (non-cartesian),
 *linearized array (x1,x2,x3,...,xn,y1,y2,y3,...,yn,z1,z2,z3,...zn)
 * @param gdata_d           Input k-space grid (cartesian)
 * @param kernel_d          precomputed interpolation kernel
 * @param sectors_d         precomputed data-sector mapping, defines the range
 *of data elements per sector, e.g. 0,3,4,4,10 -> maps data points 0..3 to
 *sector id 0, 3..4 to sector 1, no data point to sector 2, 4..10 to sector 3 an
 *so on
 * @param sector_centers_d  precomputed coordinates (x,y,z) of sector centers
 * @param gi_host           info struct with meta information
 */
void performTextureForwardConvolution(CufftType *data_d, DType *crds_d,
                                      CufftType *gdata_d, DType *kernel_d,
                                      IndType *sectors_d,
                                      IndType *sector_centers_d,
                                      gpuNUFFT::GpuNUFFTInfo *gi_host);

/**
 * \brief Forward gridding convolution implementation on GPU using sector load
 *balancing and textures.
 *
 * Performs the forward gridding convolution step on the GPU, thus the
 *interpolation
 * from uniform oversampled grid positions to non-uniform sampled k-space data
 *points.
 *
 * The distance from each sample to its neighboring grid positions is computed
 *and the corresponding
 * data value is weighted by the kernel function according to the distance.
 *
 * The kernel lookup is performed by the use of gpu textures.
 * In order to balance the work load per thread block a sector processing order
 *is precomputed.
 *
 * CUDA function prototype.
 *
 * @param data_d            Output k-space sample data value, complex, sorted
 *due to precomputation
 * @param crds_d            k-space sample coordinate (non-cartesian),
 *linearized array (x1,x2,x3,...,xn,y1,y2,y3,...,yn,z1,z2,z3,...zn)
 * @param gdata_d           Input k-space grid (cartesian)
 * @param kernel_d          precomputed interpolation kernel
 * @param sectors_d         precomputed data-sector mapping, defines the range
 *of data elements per sector, e.g. 0,3,4,4,10 -> maps data points 0..3 to
 *sector id 0, 3..4 to sector 1, no data point to sector 2, 4..10 to sector 3 an
 *so on
 * @param sector_processing_order_d precomputed sector processing order
 * @param sector_centers_d  precomputed coordinates (x,y,z) of sector centers
 * @param gi_host           info struct with meta information
 */
void performTextureForwardConvolution(CufftType *data_d, DType *crds_d,
                                      CufftType *gdata_d, DType *kernel_d,
                                      IndType *sectors_d,
                                      IndType2 *sector_processing_order_d,
                                      IndType *sector_centers_d,
                                      gpuNUFFT::GpuNUFFTInfo *gi_host);

// UTIL Functions
/** \brief Scale each element by the total number of elements.
  *
  * Used after fft steps.
  * @param data FFT data
  * @param N    Problem size N
  * @param gi_host Info struct with meta information
  */
void performFFTScaling(CufftType *data, long int N, gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Scale each element of the input data by the value of the density
  *compensation function for the corresponding sample point.
  *
  * @param data kspace data
  * @param density_comp density compensation function
  * @param gi_host Info struct with meta information
  */
void performDensityCompensation(DType2 *data, DType *density_comp,
                                gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Multiply each image element with the coil sensitivity
  *
  * @param imdata_d image data
  * @param sens_d complex coil sensitivity
  * @param gi_host Info struct with meta information
  * @param conjugate Flag whether to apply complex conjugate of sensitivity or
  *not
  */
void performSensMul(CufftType *imdata_d, DType2 *sens_d,
                    gpuNUFFT::GpuNUFFTInfo *gi_host, bool conjugate);

/** \brief Perform a element-wise summation over multiple coil images
  *
  * @param imdata_d multi-coil image data
  * @param imdata_sum_d element-wise summated image
  * @param gi_host Info struct with meta information
  */
void performSensSum(CufftType *imdata_d, CufftType *imdata_sum_d,
                    gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief FFT shift the grid data
  *
  * @param gdata_d Grid data
  * @param shift_dir gpuNUFFT::FFTShiftDir FORWARD or INVERSE
  * @param gridDims Grid dimensions
  * @param gi_host Info struct with meta information
  */
void performFFTShift(CufftType *gdata_d, gpuNUFFT::FFTShiftDir shift_dir,
                     gpuNUFFT::Dimensions gridDims,
                     gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Crop the oversampled grid according to the oversampling factor
  *
  * @param gdata_d Ovesampled grid data
  * @param imdata_d Cropped image
  * @param gi_host Info struct with meta information
  */
void performCrop(CufftType *gdata_d, CufftType *imdata_d,
                 gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Apply the deapodization function to the image
  *
  * @param imdata_d Image data
  * @param gi_host Info struct with meta information
  */
void performDeapodization(CufftType *imdata_d, gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Apply the deapodization function to the image using a precomputed
  *deapodization array
  *
  * @param imdata_d Image data
  * @param deapo_d Precomputed deapodization
  * @param gi_host Info struct with meta information
  */
void performDeapodization(CufftType *imdata_d, DType *deapo_d,
                          gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Apply the deapodization function to the image in the forward
  *transformation
  *
  * @param imdata_d Image data
  * @param gi_host Info struct with meta information
  */
void performForwardDeapodization(DType2 *imdata_d,
                                 gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Apply the deapodization function to the image in the forward
  *transformation using a precomputed deapodization array
  *
  * @param imdata_d Image data
  * @param deapo_d Precomputed deapodization
  * @param gi_host Info struct with meta information
  */
void performForwardDeapodization(DType2 *imdata_d, DType *deapo_d,
                                 gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Perform the padding of the input image to the oversampled grid
  *
  * @param imdata_d Image data
  * @param gdata_d Oversampled grid containing the zero padded image
  * @param gi_host Info struct with meta information
  */
void performPadding(DType2 *imdata_d, CufftType *gdata_d,
                    gpuNUFFT::GpuNUFFTInfo *gi_host);

/** \brief Perform the precomputation of the deapodization function
  *
  * @param deapo_d Precomputed deapodization function
  * @param gi_host Info struct with meta information
  */
void precomputeDeapodization(DType *deapo_d, gpuNUFFT::GpuNUFFTInfo *gi_host);

#endif
