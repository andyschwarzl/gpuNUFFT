#ifndef GPUNUFFT_OPERATOR_H_INCLUDED
#define GPUNUFFT_OPERATOR_H_INCLUDED

#pragma once 

#include "gpuNUFFT_types.hpp"
#include "gpuNUFFT_kernels.hpp"
#include "config.hpp"
#include <cstdlib>
#include <iostream>
#include <functional>

namespace gpuNUFFT
{
  using DebugFnType = std::function<void(const std::string&)>;

  static void defaultDebug(const std::string& message) {
    if (DEBUG) {
      printf(message.c_str());
    }
  }

/**
 * \brief Main "Operator" used for gridding operations
 *
 * The GpuNUFFTOperator is basically instantiated by the
 *gpuNUFFT::GpuNUFFTOperatorFactory.
 * The Operator is provided with all the information needed in order to perform
 *the gridding operations.
 *
 * The two main functions are:
 *
 * - performGpuNUFFTAdj for adjoint gridding operation and
 * - performForwardGpuNUFFT for the forward gridding operation
 *
 * @see GpuNUFFTOperatorFactory
 * @see BalancedGpuNUFFTOperator
 * @see TextureGpuNUFFTOperator
 */
class GpuNUFFTOperator
{
 public:
  /** \brief GpuNUFFTOperator ctor
    *
    * @param kernelWidth  kernel width in grid units
    * @param sectorWidth  sector width in grid units
    * @param osf          oversampling factor
    * @param imgDims      image dimensions of problem
    * @param loadKernel   Flag to determine whether the default interpolation
    *kernel has to be loaded or not, false for texture interpolation
    * @param operatorType Overwrite default operator type
    */
  GpuNUFFTOperator(IndType kernelWidth, IndType sectorWidth, DType osf,
                   Dimensions imgDims, bool loadKernel = true,
                   OperatorType operatorType = DEFAULT,
                   bool matlabSharedMem = false, bool grad_mode = false)
    : operatorType(operatorType), osf(osf), kernelWidth(kernelWidth),
      sectorWidth(sectorWidth), imgDims(imgDims), gpuMemAllocated(false),
      debugTiming(DEBUG), sens_d(NULL), crds_d(NULL), density_comp_d(NULL),
      deapo_d(NULL), gdata_d(NULL), sector_centers_d(NULL), sectors_d(NULL),
      data_indices_d(NULL), data_sorted_d(NULL), allocatedCoils(0),
      matlabSharedMem(matlabSharedMem), grad_mode(grad_mode)
  {
    if (loadKernel)
      initKernel();

    sectorDims.width = sectorWidth;
    sectorDims.height = sectorWidth;
    if (imgDims.depth > 0)
      sectorDims.depth = sectorWidth;
  }

  virtual ~GpuNUFFTOperator()
  {
    freeDeviceMemory();
    freeLocalMemberArray(this->kernel.data);
    freeLocalMemberArray(this->dens.data);
    freeLocalMemberArray(this->sens.data);
    freeLocalMemberArray(this->deapo.data);
    freeLocalMemberArray(this->kSpaceTraj.data);
    freeLocalMemberArray(this->sectorCenters.data);
    freeLocalMemberArray(this->dataIndices.data);
    freeLocalMemberArray(this->sectorDataCount.data);
  }

  friend class GpuNUFFTOperatorFactory;

  // SETTER
  void setOsf(DType osf)
  {
    this->osf = osf;
  }

  void setKSpaceTraj(Array<DType> kSpaceTraj)
  {
    if (this->kSpaceTraj.data != NULL)
      freeLocalMemberArray(this->kSpaceTraj.data);
    this->kSpaceTraj = kSpaceTraj;
  }
  void setSectorCenters(Array<IndType> sectorCenters)
  {
    if(this->sectorCenters.data != NULL)
      freeLocalMemberArray(this->sectorCenters.data);
    this->sectorCenters = sectorCenters;
  }
  void setSectorDataCount(Array<IndType> sectorDataCount)
  {
    if(this->sectorDataCount.data != NULL)
      freeLocalMemberArray(this->sectorDataCount.data);

    this->sectorDataCount = sectorDataCount;
  }
  void setDataIndices(Array<IndType> dataIndices)
  {
    if (this->dataIndices.data != NULL)
      freeLocalMemberArray(this->dataIndices.data);
    this->dataIndices = dataIndices;
  }
  void setSens(Array<DType2> sens)
  {
    this->sens = sens;
  }
  void setDens(Array<DType> dens)
  {
    if (this->dens.data != NULL)
      freeLocalMemberArray(this->dens.data);
    this->dens = dens;
  }
  void setDeapodizationFunction(Array<DType> deapo)
  {
    if (this->deapo.data != NULL)
      freeLocalMemberArray(this->deapo.data);
    this->deapo= deapo;
  }

  void setImageDims(Dimensions dims)
  {
    this->imgDims = dims;
  }
  void setGridSectorDims(Dimensions dims)
  {
    this->gridSectorDims = dims;
  }

  // GETTER
  DType getOsf()
  {
    return this->osf;
  }
  Array<DType> getKSpaceTraj()
  {
    return this->kSpaceTraj;
  }

  Array<DType2> getSens()
  {
    return this->sens;
  }
  Array<DType> getDens()
  {
    return this->dens;
  }
  Array<DType> getKernel()
  {
    return this->kernel;
  }
  Array<IndType> getSectorDataCount()
  {
    return this->sectorDataCount;
  }

  IndType getKernelWidth()
  {
    return this->kernelWidth;
  }
  IndType getSectorWidth()
  {
    return this->sectorWidth;
  }

  Dimensions getImageDims()
  {
    return this->imgDims;
  }
  Dimensions getGridDims()
  {
    return this->imgDims * osf;
  }

  Dimensions getGridSectorDims()
  {
    return this->gridSectorDims;
  }
  Dimensions getSectorDims()
  {
    return this->sectorDims;
  }

  Array<IndType> getSectorCenters()
  {
    return this->sectorCenters;
  }
  IndType *getSectorCentersData()
  {
    return reinterpret_cast<IndType *>(this->sectorCenters.data);
  }

  Array<IndType> getDataIndices()
  {
    return this->dataIndices;
  }

  bool is2DProcessing()
  {
    return this->imgDims.depth == 0;
  }
  bool is3DProcessing()
  {
    return !is2DProcessing();
  }

  /** \brief Return dimension of problem. Either 2-d or 3-d. */
  int getImageDimensionCount()
  {
    return (is2DProcessing() ? 2 : 3);
  }

  // OPERATIONS

  // adjoint gpuNUFFT

  /** \brief Perform Adjoint gridding operation on given kspaceData
    *
    * Gridding implementation - interpolation from nonuniform k-space data onto
    *                           oversampled grid based on optimized gridding
    *kernel
    *                           with minimal oversampling ratio (see Beatty et
    *al.)
    *
    * Basic steps: - density compensation
    *              - convolution with interpolation function
    *              - iFFT
    *              - cropping due to oversampling ratio
    *              - apodization correction
    *
    * The memory for the output array is allocated automatically but has to be
    *freed
    * manually.
    *
    * @param k-space data
    * @return Regridded Image
    */
  Array<CufftType> performGpuNUFFTAdj(Array<DType2> kspaceData);

  /** \brief Perform Adjoint gridding operation on given kspaceData
    *
    * @param k-space data
    * @param preallocated image data array
    * @param Stop gridding operation after gpuNUFFT::GpuNUFFTOutput
    * @return Regridded Image
    */
  virtual void performGpuNUFFTAdj(Array<DType2> kspaceData,
                                  Array<CufftType> &imgData,
                                  GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);

  /** \brief Perform Adjoint gridding operation on kspaceData already residing
   *in GPU memory
   *
   * This may be the case in iterative reconstructions, when k-Space and image
   * data is already residing on the GPU.
   *
   *
   * @param k-space data
   * @param preallocated image data array
   * @param Stop gridding operation after gpuNUFFT::GpuNUFFTOutput
   * @return Regridded Image
   */
  virtual void performGpuNUFFTAdj(GpuArray<DType2> kspaceData_gpu,
                                  GpuArray<CufftType> &imgData_gpu,
                                  GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);

  /** \brief Perform Adjoint gridding operation on given kspaceData
    *
    * @param k-space data
    * @param Stop gridding operation after gpuNUFFT::GpuNUFFTOutput
    * @return Regridded Image
    */
  Array<CufftType> performGpuNUFFTAdj(Array<DType2> kspaceData,
                                      GpuNUFFTOutput gpuNUFFTOut);

  // forward gpuNUFFT

  /** \brief Perform forward gridding operation on given kspaceData
   *
   * Inverse gpuNUFFT implementation - interpolation from uniform grid data onto
   *                                   nonuniform k-space data based on
   *optimized
   *                                   gpuNUFFT kernel with minimal oversampling
   *                                   ratio (see Beatty et al.)
   *
   * Basic steps: - apodization correction
   *              - zero padding with osf
   *              - FFT
   *               - convolution and resampling
   *
   * The memory for the output array is allocated automatically but has to be
   *freed
   * manually.
   *
   * @param image data
   * @return k-space data
   */
  Array<CufftType> performForwardGpuNUFFT(Array<DType2> imgData);

  /** \brief Perform forward gridding operation on given kspaceData
    *
    * The memory for the output array is allocated automatically but has to be
    *freed
    * manually.
    *
    * @param image data
    * @param Stop gridding operation after gpuNUFFT::GpuNUFFTOutput
    * @return k-space data
    */
  virtual void
  performForwardGpuNUFFT(Array<DType2> imgData, Array<CufftType> &kspaceData,
                         GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);

  /** \brief Perform forward gridding operation on kspaceData already residing
   *in GPU memory
   *
   * This may be the case in iterative reconstructions, when k-Space and image
   * data is already residing on the GPU.
   *
   * @param image data
   * @param Stop gridding operation after gpuNUFFT::GpuNUFFTOutput
   * @return k-space data
   */
  virtual void
  performForwardGpuNUFFT(GpuArray<DType2> imgData_gpu,
                         GpuArray<CufftType> &kspaceData_gpu,
                         GpuNUFFTOutput gpuNUFFTOut = DEAPODIZATION);

  /** \brief Perform forward gridding operation on given kspaceData
    *
    * The memory for the output array is allocated automatically but has to be
    *freed
    * manually.
    *
    * @param image data
    * @param Stop gridding operation after gpuNUFFT::GpuNUFFTOutput
    * @return k-space data
    */
  Array<CufftType> performForwardGpuNUFFT(Array<DType2> imgData,
                                          GpuNUFFTOutput gpuNUFFTOut);

  void clean_memory();

  void setGradMode(bool grad_mode) {
    this->grad_mode = grad_mode;
  }

  bool getGradMode() {
    return this->grad_mode;
  }
  /** \brief Check if density compensation data is available. */
  bool applyDensComp()
  {
    return (this->dens.data != NULL && this->dens.count() > 1);
  }

  /** \brief Check if coil sensitivity data is available. */
  bool applySensData()
  {
    return (this->sens.data != NULL && this->sens.count() > 1);
  }

  /** \brief Return type of GriddingOperator. */
  virtual OperatorType getType()
  {
    return operatorType;
  }

  /** \brief Function to allocate all neccessary device memory used by the
   * GriddingOperator. */
  void initDeviceMemory(int n_coils, int n_coils_cc = 1);

  /** \brief Function to free the neccessary device memory used by the
   * GriddingOperator. */
  void freeDeviceMemory();
  
  /** \brief Gridding Info meta information.
  *
  * @see gpuNUFFT::GpuNUFFTOperator::initGpuNUFFTInfo
  */
  GpuNUFFTInfo *gi_host;

 protected:

   template<typename T>
   void freeLocalMemberArray(T* dataPointer)
   {
     if (dataPointer != NULL) {
       cudaFreeHost(dataPointer);
       dataPointer = NULL;
     }
   }

  /** \brief gpuNUFFT::OperatorType classifier. Value according to sub-class
   * implementation. */
  OperatorType operatorType;

  /** \brief Precomputed interpolation kernel lookup table. */
  Array<DType> kernel;

  /** \brief k space trajectory. Array of coordinates.
   *
   * simple array
   * dimensions: n dimensions * dataCount
   */
  Array<DType> kSpaceTraj;

  /** \brief Array of coil sensitivities.
    * complex array
    * dimensions: imgDim * chnCount
    */
  Array<DType2> sens;

  /** \brief Density compensation array.
    * density compensation
    * dimensions: dataCount
    */
  Array<DType> dens;

  /** \brief Deapodization function array
  * 
  * dimensions: image dimensions
  */
  Array<DType> deapo;

  /** \brief Sector centers array.
    * sector centers
    * dimensions: sectorCount * nDimensions (3: x,y,z)
    */
  Array<IndType> sectorCenters;

  /** \brief Boundaries of assigned dataIndices per sector element.
    *
    *  Defines the range of data elements per sector,
    *  e.g. [0,3,4,4,10]
    *   -> maps data points 0..3 to sector 0,
    *                       3..4 to sector 1,
    *                       no data point to sector 2,
    *                       4..10 to sector 3 an so on
    */
  Array<IndType> sectorDataCount;

  /** \brief Array of indices defining the sort order/mapping of input k-space
    *data to sectors
    *
    * Mapping/assignment of data index to according sector is defined by the
    *dataIndices array
    * and the sectorDataCount array
    *
    */
  Array<IndType> dataIndices;

  /** \brief Oversampling factor */
  DType osf;

  /** \brief Width of kernel in grid units */
  IndType kernelWidth;

  /** \brief Sector size in grid units */
  IndType sectorWidth;

  /** \brief Image Dimensions */
  Dimensions imgDims;

  /** \brief Amount of sectors per grid direction */
  Dimensions gridSectorDims;

  /** \brief Size of one sector in grid units*/
  Dimensions sectorDims;

  /** \brief Flag which indicates if data pointers are allocated with Matlab 
  */
  bool matlabSharedMem;

  /** \brief Flag for changing the isign, mainly used for gradients 
  */
  bool grad_mode;

  /** \brief Return Grid Width (ImageWidth * osf) */
  IndType getGridWidth()
  {
    return (IndType)(this->getGridDims().width);
  }

  /** \brief Select data array in ordered manner. */
  template <typename T> T *selectOrdered(Array<T> &dataArray, int offset = 0);

  /** \brief Select data array in ordered manner and write it to output array.
   */
  template <typename T>
  void writeOrdered(Array<T> &destArray, T *sortedArray, int offset = 0);

  /** \brief Precompute interpolation kernel lookup table. */
  virtual void initKernel();

  /** \brief Compute all neccessary meta information used in the gridding steps.
    *
    * @see gpuNUFFT::GpuNUFFTInfo
    */
  GpuNUFFTInfo *initGpuNUFFTInfo(int n_coils_cc = 1);

  /** \brief Virtual method to perform precomputation of all neccessary meta
  *information used in the gridding steps.
  *
  * @see gpuNUFFT::GpuNUFFTInfo
  */
  virtual GpuNUFFTInfo *initAndCopyGpuNUFFTInfo(int n_coils_cc = 1);

  /** \brief Virtual adjoint convolution call, which can be used by sub-classes
   *to add behaviour to the gridding steps
   *
   */
  virtual void adjConvolution(DType2 *data_d, DType *crds_d, CufftType *gdata_d,
                              DType *kernel_d, IndType *sectors_d,
                              IndType *sector_centers_d,
                              gpuNUFFT::GpuNUFFTInfo *gi_host);

  /** \brief Virtual forward convolution call, which can be used by sub-classes
   *to add behaviour to the gridding steps
   *
   */
  virtual void forwardConvolution(CufftType *data_d, DType *crds_d,
                                  CufftType *gdata_d, DType *kernel_d,
                                  IndType *sectors_d, IndType *sector_centers_d,
                                  gpuNUFFT::GpuNUFFTInfo *gi_host);

  /** \brief Virtual method to allow different methods of generation of the
   *lookup table
   *
   * @see TextureGpuNUFFTOperator
   */
  virtual void initLookupTable();

  /** \brief Virtual method to allow custom free of lookup table */
  virtual void freeLookupTable();

  /** \brief Pass debug function pointer
  */
  virtual void setDebugFunction(DebugFnType debugFn) {
    this->debug = debugFn;
  }

 private:
  /** \brief Flag to remember if gpu device memory has already been allocated */
  bool gpuMemAllocated;

  /** \brief Flag to determine debugging of GPU kernel execution times */
  bool debugTiming;

  // GPU Device Members

  /** \brief GPU Device pointer to sensitivity array elements. */
  DType2 *sens_d;
  /** \brief GPU Device pointer to coordinates array elements. */
  DType *crds_d;
  /** \brief GPU Device pointer to density compensation function array elements.
   */
  DType *density_comp_d;
  /** \brief GPU Device pointer to deapodization function array elements. */
  DType *deapo_d;
  /** \brief GPU Device pointer to grid data array elements. */
  CufftType *gdata_d;
  /** \brief GPU Device pointer to sector centers array elements. */
  IndType *sector_centers_d;
  /** \brief GPU Device pointer to sector data mapping array elements. */
  IndType *sectors_d;
  /** \brief GPU Device pointer to data indices array elements. */
  IndType *data_indices_d;
  /** \brief GPU Device pointer to sorted data array elements. */
  DType2 *data_sorted_d;

  int allocatedCoils;

  /** \brief GPU CUFFT plan. */
  cufftHandle fft_plan;

  /** \brief CUDA events for timing tests. */
  cudaEvent_t start, stop;

  /** \brief Function to start CUDA timing test. */
  void startTiming();
  /** \brief Function to stop CUDA timing test. */
  float stopTiming();

  /** \brief Update amount of concurrently computed coils
   */
  void updateConcurrentCoilCount(int coil_it, int n_coils, int &n_coils_cc, cudaStream_t stream=0);

  /** \brief Compute amount of coils which can be computed at once.
   *
   * Depends primary on available GPU memory
   *
   */
  int computePossibleConcurrentCoilCount(int n_coils,
                                         gpuNUFFT::Dimensions kSpaceDataDim);

  DebugFnType debug = &defaultDebug;
};
}

#endif  // GPUNUFFT_OPERATOR_H_INCLUDED
