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
    protected:
      GriddingOperatorMatlabFactory()
      {
      }

	  
	  //abstract methods from GriddingOperatorFactory
	  Array<IndType> initDataIndices(GriddingOperator* griddingOp, size_t coordCnt);
	  Array<DType> initCoordsData(GriddingOperator* griddingOp, size_t coordCnt);
	  Array<IndType3> initSectorCenters(GriddingOperator* griddingOp, size_t sectorCnt);
	  Array<IndType> initSectorDataCount(GriddingOperator* griddingOp, size_t dataCount);
	  Array<DType> initDensData(GriddingOperator* griddingOp, size_t coordCnt);

	  void debug(const std::string& message);

    public:
	
		  ~GriddingOperatorMatlabFactory()
		  {
			  std::cout << "GOMF destruct " << std::endl;
		  }

		// SETTER 
		
		// GETTER
		
		// OPERATIONS

        static GriddingOperatorMatlabFactory& getInstance();
        
		GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType2>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, Dimensions& imgDims,mxArray *plhs[]);
		GriddingOperator* loadPrecomputedGriddingOperator(Array<DType>& kSpaceTraj, Array<IndType>& dataIndices, Array<IndType>& sectorDataCount,Array<IndType3>& sectorCenters, Array<DType>& densCompData, Array<DType2>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, Dimensions& imgDims);
		GriddingOperator* loadPrecomputedGriddingOperator(Array<DType>& kSpaceTraj, Array<IndType>& dataIndices, Array<IndType>& sectorDataCount,Array<IndType3>& sectorCenters, Array<DType2>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, Dimensions& imgDims);
	private:
		static GriddingOperatorMatlabFactory instance;

		mxArray **plhs;
		
    };

}

#endif //GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED
