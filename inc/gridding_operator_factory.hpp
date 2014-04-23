#ifndef GRIDDING_OPERATOR_FACTORY_H_INCLUDED
#define GRIDDING_OPERATOR_FACTORY_H_INCLUDED

#include "config.hpp"
#include "gridding_operator.hpp"
#include "balanced_gridding_operator.hpp"
#include "texture_gridding_operator.hpp"
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <string>
#include <cmath>

#include "cuda_utils.hpp"
#include "precomp_utils.hpp"

namespace GriddingND
{
  // GriddingOperatorFactory
  // 
  // Manages the initialization of the Gridding Operator.
  // Distinguishes between two cases:
  //
  // * new calculation of "data - sector" mapping, sorting etc.
  //  
  // * reuse of previously calculated mapping
  // 
  class GriddingOperatorFactory
  {
  public:

    GriddingOperatorFactory(const InterpolationType interpolationType,const bool useGpu) 
      : interpolationType(interpolationType), useGpu(useGpu)
    {
    }

    GriddingOperatorFactory(const InterpolationType interpolationType) 
      : interpolationType(interpolationType), useGpu(true)
    {
    }

    GriddingOperatorFactory() 
      : interpolationType(CONST_LOOKUP),useGpu(true)
    {
    }

    ~GriddingOperatorFactory()
    {
    }

    GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    GriddingOperator* loadPrecomputedGriddingOperator(Array<DType>& kSpaceTraj, Array<IndType>& dataIndices, Array<IndType>& sectorDataCount,GriddingND::Array<IndType>& sectorProcessingOrder,Array<IndType>& sectorCenters, Array<DType>& densCompData, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    GriddingOperator* loadPrecomputedGriddingOperator(Array<DType>& kSpaceTraj, Array<IndType>& dataIndices, Array<IndType>& sectorDataCount,GriddingND::Array<IndType>& sectorProcessingOrder,Array<IndType>& sectorCenters, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

    void setInterpolationType(InterpolationType interpolationType);
    
  protected:
    Array<IndType> assignSectors(GriddingOperator* griddingOp, Array<DType>& kSpaceTraj);

    template <typename T> 
    Array<T> initLinArray(IndType arrCount);

    virtual Array<IndType> initDataIndices(GriddingOperator* griddingOp, IndType coordCnt);
    virtual Array<IndType> initSectorDataCount(GriddingOperator* griddingOp, IndType coordCnt);
    virtual Array<IndType> initSectorProcessingOrder(GriddingOperator* griddingOp, IndType sectorCnt);
    virtual Array<DType> initDensData(GriddingOperator* griddingOp, IndType coordCnt);
    virtual Array<DType> initCoordsData(GriddingOperator* griddingOp, IndType coordCnt);
    virtual Array<IndType> initSectorCenters(GriddingOperator* griddingOp, IndType sectorCnt);
    virtual void debug(const std::string& message);

    IndType computeSectorCountPerDimension(IndType dim, IndType sectorWidth);

    Dimensions computeSectorCountPerDimension(Dimensions dim, IndType sectorWidth);

    IndType computeTotalSectorCount(Dimensions dim, IndType sectorWidth);

    template <typename T>
    std::vector<IndPair> sortVector(Array<T> assignedSectors);

    Array<IndType> computeSectorDataCount(GriddingND::GriddingOperator *griddingOp,GriddingND::Array<IndType> assignedSectors);
    void computeProcessingOrder(BalancedGriddingOperator* griddingOp);

    Array<IndType> computeSectorCenters(GriddingOperator *griddingOp);
    Array<IndType> computeSectorCenters2D(GriddingOperator *griddingOp);

    IndType computeSectorCenter(IndType var, IndType sectorWidth);

    GriddingOperator* createNewGriddingOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims);

  private:
    InterpolationType interpolationType;

    bool useGpu;
  };

}

#endif //GRIDDING_OPERATOR_FACTORY_H_INCLUDED
