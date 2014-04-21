#ifndef GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED
#define GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED

#include "config.hpp"
#include "gridding_operator_factory.hpp"
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include "matlab_helper.h"
#include <string>

#include "mex.h"
#include "matrix.h"

namespace GriddingND
{
  // GriddingOperatorMatlabFactory
  // decorates GriddingOperatorFactory
  //
  // Manages the initialization of the Gridding Operator.
  // Distinguishes between two cases:
  //
  // * new calculation of "data - sector" mapping, sorting etc.
  //  
  // * reuse of previously calculated mapping
  // 
  class GriddingOperatorMatlabFactory : public GriddingOperatorFactory
  {
  public:

    GriddingOperatorMatlabFactory():
        GriddingOperatorFactory()
    {
    }

    GriddingOperatorMatlabFactory(const InterpolationType interpolationType,const bool useGpu) 
      : GriddingOperatorFactory(interpolationType,useGpu)
    {
    }

    GriddingOperatorMatlabFactory(const InterpolationType interpolationType) 
      : GriddingOperatorFactory(interpolationType)
    {
    }

    ~GriddingOperatorMatlabFactory()
    {
    }

    GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims,mxArray *plhs[]);

  protected:

    //abstract methods from GriddingOperatorFactory
    Array<IndType> initDataIndices(GriddingOperator* griddingOp, IndType coordCnt);
    Array<DType> initCoordsData(GriddingOperator* griddingOp, IndType coordCnt);
    Array<IndType> initSectorCenters(GriddingOperator* griddingOp, IndType sectorCnt);
    Array<IndType> initSectorDataCount(GriddingOperator* griddingOp, IndType dataCount);
    Array<IndType> initSectorProcessingOrder(GriddingOperator* griddingOp, IndType sectorCnt);
    Array<DType> initDensData(GriddingOperator* griddingOp, IndType coordCnt);

    void debug(const std::string& message);

  private:
    mxArray **plhs;

  };

}

#endif //GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED
