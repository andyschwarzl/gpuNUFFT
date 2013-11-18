#ifndef GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED
#define GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED

#include "config.hpp"
#include "gridding_operator_factory.hpp"
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include "matlab_helper.h"

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

    public:
	
		  ~GriddingOperatorMatlabFactory()
		  {
			  std::cout << "GOMF destruct " << std::endl;
		  }

		// SETTER 
		
		// GETTER
		
		// OPERATIONS

        static GriddingOperatorMatlabFactory& getInstance();
        
		GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, Dimensions& imgDims,mxArray *plhs[]);

	private:
		static GriddingOperatorMatlabFactory instance;
		
    };

}

#endif //GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED
