#ifndef GRIDDING_OPERATOR_FACTORY_H_INCLUDED
#define GRIDDING_OPERATOR_FACTORY_H_INCLUDED

#include "config.hpp"
#include "gridding_operator.hpp"

namespace GriddingND
{
	// GriddingOperatorFactory
	// 
	// Manages the initialization of the Gridding Operator.
	// Distinguishes between two cases:
	//
	// * new calculation of "data - sector" mapping
	//  
	// * reuse of previously calculated mapping
	// 
	// T defines if the Factory works on 
	class GriddingOperatorFactory
	{
    protected:
      GriddingOperatorFactory()
      {
      }

      ~GriddingOperatorFactory()
      {
        delete instance;
      }
    public:

        GriddingOperator* createGriddingOperator(Array<DType> kSpaceData, size_t kernelWidth, size_t sectorWidth, DType osf);

		// SETTER 
		
		// GETTER
		
		// OPERATIONS

        static GriddingOperatorFactory* getInstance();
        
	private:
        static GriddingOperatorFactory* instance;
		
    };

}

#endif //GRIDDING_OPERATOR_FACTORY_H_INCLUDED
