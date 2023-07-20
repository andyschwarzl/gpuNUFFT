#ifndef GPUNUFFT_OPERATOR_FACTORY_H_INCLUDED
#define GPUNUFFT_OPERATOR_FACTORY_H_INCLUDED

#include "config.hpp"
#include "gpuNUFFT_operator.hpp"
#include "balanced_gpuNUFFT_operator.hpp"
#include "texture_gpuNUFFT_operator.hpp"
#include "balanced_texture_gpuNUFFT_operator.hpp"
#include <algorithm>  // std::sort
#include <vector>     // std::vector
#include <string>
#include <cmath>

#include "cuda_utils.hpp"
#include "precomp_utils.hpp"

namespace gpuNUFFT
{
/** \brief Manages the initialization of the GpuNUFFT Operator and its sub
 *classes.
 *
 * Distinguishes between two cases:
 *
 * - new computation of "data - sector" mapping, sorting etc.
 *
 * - reuse of previously calculated mapping, i.e. loading of an already created
 *   operator like from subsequent matlab calls
 *
 * The factory defines how the operator is going to process (load balancing
 *and/or
 * texture interpolation).
 *
 * Sector mapping:
 *
 * The grid is split into multiple subunits (sectors) in order to enable a
 *parallel
 * processing strategy. The sector size in each dimension is defined by the
 *parameter
 * 'sectorWidth', e.g. a grid of (256,256,256) is split into (32,32,32) sectors
 *by
 * a sector width of 8.
 *
 * Due to this strategy it is neccessary to perform a precomputation step, which
 * assigns each data sample to its surrounding sector. This is the main step
 * performed by the gpuNUFFTOperatorFactory.
 *
 */
class GpuNUFFTOperatorFactory
{
 public:

  /** \brief Constructor overload
    *
    * @param useTextures Flag to indicate texture interpolation
    * @param useGpu Flag to indicat&GpuNUFFTPythonOperator::adj_op);e gpu usage for precomputation
    * @param balanceWorkload Flag to indicate load balancing
    */
  GpuNUFFTOperatorFactory(const bool useTextures = false, const bool useGpu = true,
                          bool balanceWorkload = true, bool matlabSharedMem = false)
    : useTextures(useTextures), useGpu(useGpu), balanceWorkload(balanceWorkload),
    matlabSharedMem(matlabSharedMem)
  {
  }

  ~GpuNUFFTOperatorFactory()
  {
  }

  /** \brief Create GpuNUFFT Operator.
    *
    * Based on the k-space trajectory (x,y,(z))  and the given gridding
    *parameters the
    * GpuNUFFTOperator is precomputed.
    *
    * @param kSpaceTraj     coordinate array of sample locations
    * @param kernelWidth    interpolation kernel size in grid units
    * @param sectorWidth    sector width
    * @param osf            grid oversampling ratio
    * @param imgDims        image dimensions (problem size)
   */
  GpuNUFFTOperator *createGpuNUFFTOperator(Array<DType> &kSpaceTraj,
                                           const IndType &kernelWidth,
                                           const IndType &sectorWidth,
                                           const DType &osf,
                                           Dimensions &imgDims);

  /** \brief Create GpuNUFFT Operator.
    *
    * Based on the k-space trajectory (x,y,(z))  and the given gridding
    *parameters the
    * GpuNUFFTOperator is precomputed.
    *
    * @param kSpaceTraj     coordinate array of sample locations
    * @param densCompData   data for density compensation
    * @param kernelWidth    interpolation kernel size in grid units
    * @param sectorWidth    sector width
    * @param osf            grid oversampling ratio
    * @param imgDims        image dimensions (problem size)
   */
  GpuNUFFTOperator *
  createGpuNUFFTOperator(Array<DType> &kSpaceTraj, Array<DType> &densCompData,
                         const IndType &kernelWidth, const IndType &sectorWidth,
                         const DType &osf, Dimensions &imgDims);

  /** \brief Create GpuNUFFT Operator.
    *
    * Based on the k-space trajectory (x,y,(z))  and the given gridding
    *parameters the
    * GpuNUFFTOperator is precomputed.
    *
    * @param kSpaceTraj     coordinate array of sample locations
    * @param densCompData   data for density compensation
    * @param sensData       coil sensitivity data
    * @param kernelWidth    interpolation kernel size in grid units
    * @param sectorWidth    sector width
    * @param osf            grid oversampling ratio
    * @param imgDims        image dimensions (problem size)
   */
  GpuNUFFTOperator *
  createGpuNUFFTOperator(Array<DType> &kSpaceTraj, Array<DType> &densCompData,
                         Array<DType2> &sensData, const IndType &kernelWidth,
                         const IndType &sectorWidth, const DType &osf,
                         Dimensions &imgDims);

  /** \brief Load GpuNUFFT Operator from previously computed mappings.
    *
    * Based on a previously performed mapping the GpuNUFFTOperator can be
    *loaded.
    *
    * @param kSpaceTraj             coordinate array of sample locations
    * @param dataIndices            precomputed data indices array
    * @param sectorDataCount        precomputed sector data count array
    * @param sectorProcessingOrder  precomputed sector processing order array
    * @param sectorCenters          precomputed sector centers array
    * @param densCompData           data for density compensation
    * @param sensData               coil sensitivity data
    * @param kernelWidth            interpolation kernel size in grid units
    * @param sectorWidth            sector width
    * @param osf                    grid oversampling ratio
    * @param imgDims                image dimensions (problem size)
   */
  GpuNUFFTOperator *loadPrecomputedGpuNUFFTOperator(
      Array<DType> &kSpaceTraj, Array<IndType> &dataIndices,
      Array<IndType> &sectorDataCount,
      gpuNUFFT::Array<IndType2> &sectorProcessingOrder,
      Array<IndType> &sectorCenters, Array<DType> &densCompData,
      Array<DType2> &sensData, Array<DType> &deapoData, const IndType &kernelWidth,
      const IndType &sectorWidth, const DType &osf, Dimensions &imgDims);

  /** \brief Load GpuNUFFT Operator from previously computed mappings.
    *
    * Based on a previously performed mapping the GpuNUFFTOperator can be
    *loaded.
    *
    * @param kSpaceTraj             coordinate array of sample locations
    * @param dataIndices            precomputed data indices array
    * @param sectorDataCount        precomputed sector data count array
    * @param sectorProcessingOrder  precomputed sector processing order array
    * @param sectorCenters          precomputed sector centers array
    * @param sensData               coil sensitivity data
    * @param kernelWidth            interpolation kernel size in grid units
    * @param sectorWidth            sector width
    * @param osf                    grid oversampling ratio
    * @param imgDims                image dimensions (problem size)
   */
  GpuNUFFTOperator *loadPrecomputedGpuNUFFTOperator(
      Array<DType> &kSpaceTraj, Array<IndType> &dataIndices,
      Array<IndType> &sectorDataCount,
      gpuNUFFT::Array<IndType2> &sectorProcessingOrder,
      Array<IndType> &sectorCenters, Array<DType2> &sensData,
      Array<DType> &deapoData, const IndType &kernelWidth, const IndType &sectorWidth, 
      const DType &osf, Dimensions &imgDims);

  void setUseTextures(bool useTextures);

  void setBalanceWorkload(bool balanceWorkload);

 protected:
  template<typename T>
   void freeLocalMemberArray(T* dataPointer)
   {
     if (dataPointer != NULL) {
       cudaFreeHost(dataPointer);
       dataPointer = NULL;
     }
   }
   /** \brief Assign the samples on the k-space trajectory to its corresponding
    *sector
    *
    * @return array of indices of the assigned sector
   */
  Array<IndType> assignSectors(GpuNUFFTOperator *gpuNUFFTOp,
                               Array<DType> &kSpaceTraj);

  /** \brief Init a linear array of size arrCount */
  template <typename T> Array<T> initLinArray(IndType arrCount);

  /** \brief Initialization method for the data indices array */
  virtual Array<IndType> initDataIndices(GpuNUFFTOperator *gpuNUFFTOp,
                                         IndType coordCnt);

  /** \brief Initialization method for the sector data count array */
  virtual Array<IndType> initSectorDataCount(GpuNUFFTOperator *gpuNUFFTOp,
                                             IndType coordCnt);

  /** \brief Initialization method for the sector processing order array */
  virtual Array<IndType2>
  initSectorProcessingOrder(GpuNUFFTOperator *gpuNUFFTOp, IndType sectorCnt);

  /** \brief Initialization method for the density compensation array */
  virtual Array<DType> initDensData(GpuNUFFTOperator *gpuNUFFTOp,
                                    IndType coordCnt);

  /** \brief Initialization method for the coords array */
  virtual Array<DType> initCoordsData(GpuNUFFTOperator *gpuNUFFTOp,
                                      IndType coordCnt);

  /** \brief Initialization method for the sector centers array */
  virtual Array<IndType> initSectorCenters(GpuNUFFTOperator *gpuNUFFTOp,
                                           IndType sectorCnt);

  /** \brief Initialization method for the deapodization function array */
  virtual Array<DType> initDeapoData(IndType imgDimsCount);

  /** \brief Debug message */
  virtual void debug(const std::string &message);

  /** \brief Compute the amount of sectors per dimension */
  IndType computeSectorCountPerDimension(IndType dim, IndType sectorWidth);

  /** \brief Compute the amount of sectors per each dimension */
  Dimensions computeSectorCountPerDimension(Dimensions dim,
                                            IndType sectorWidth);

  /** \brief Compute the total count of sectors. */
  IndType computeTotalSectorCount(Dimensions dim, IndType sectorWidth);

  /** \brief Sort array of values and keep index information for further
    *processing.
    *
    * @return Vector of index-value pairs
    */
  template <typename T>
  std::vector<IndPair> sortVector(Array<T> assignedSectors,
                                  bool descending = false);

  /** \brief Compute the boundaries of the assigned dataIndices per sector
    *element.
    *
    *  Defines the range of data elements per sector,
    *  e.g. if the ordered assigned sector array is like [0,0,0,1,1,2]
    *  (thus the first three elements are assigned to sector 0, and so on)
    *
    *  The boundaries are computed as subsequently [start,end) pairs, like
    *[0,3,5,6]
    *  So:
    *     - sector 0 accesses the array elements 0...3 (excl. 3)
    *     - sector 1 accesses the array elements 3...5 (excl. 5)
    *     - sector 2 accesses the array elements 5...6 (excl. 6)
    *
   */
  Array<IndType>
  computeSectorDataCount(gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp,
                         gpuNUFFT::Array<IndType> assignedSectors, bool useLocalMemory = false);

  /** \brief Method to compute the sector processing order.
    *
    * This method iterates over the previously performed sector mapping
    * and splits any sectors which contain more samples than
    * defined in MAXIMUM_PAYLOAD.
    *
    */
  void computeProcessingOrder(GpuNUFFTOperator *gpuNUFFTOp);

  /** \brief Compute sector centers array */
  Array<IndType> computeSectorCenters(GpuNUFFTOperator *gpuNUFFTOp, bool useLocalMemory = false);
  /** \brief Compute 2-d sector centers array */
  Array<IndType> computeSectorCenters2D(GpuNUFFTOperator *gpuNUFFTOp, bool useLocalMemory = false);

  /** \brief Method to compute the sector center for the given sector index. */
  IndType computeSectorCenter(IndType var, IndType sectorWidth);

  /** \brief Method to create a new GpuNUFFTOperator or a sub class depending on
    *the parameters.
    *
    * Depending on the initializiation of the factory different
    *GpuNUFFTOperators may be
    * allocated:
    *
    * - default: GpuNUFFTOperator
    * - balanceWorkload = true: BalancedGpuNUFFTOperator
    * - useTextures = true: TextureGpuNUFFTOperator
    * - balanceWorkload + useTextures = true: BalancedTextureGpuNUFFTOperator
    *
    * @return New allocated GpuNUFFTOperator or sub class
    */
  GpuNUFFTOperator *createNewGpuNUFFTOperator(IndType kernelWidth,
                                              IndType sectorWidth, DType osf,
                                              Dimensions imgDims);

  /**
   * \brief Function to check if the problem will fit into device memory
   *
   * @throws Exception in case of too much required memory
   */
  void checkMemoryConsumption(Dimensions &kSpaceDims,
                              const IndType &sectorWidth, const DType &osf,
                              Dimensions &imgDims, Dimensions &densDims,
                              Dimensions &sensDims);

  /**
  * \brief Computation of the deapodization function
  * 
  * @returns scalar array in image dimensions (imgDims)
  */
  gpuNUFFT::Array<DType> computeDeapodizationFunction(const IndType &kernelWidth,
    const DType &osf, gpuNUFFT::Dimensions &imgDims);

 private:
  /** \brief Flag to indicate texture interpolation */
  bool useTextures;

  /** \brief Flag to indicate gpu usage for precomputation */
  bool useGpu;

  /** \brief Flag to indicate load balancing */
  bool balanceWorkload;

  /** \brief Flag to indicate shared memory usage with Matlab */
  bool matlabSharedMem;
};
}

#endif  // GPUNUFFT_OPERATOR_FACTORY_H_INCLUDED
