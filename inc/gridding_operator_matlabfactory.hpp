#ifndef GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED
#define GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED

#include "config.hpp"
#include "gridding_operator_factory.hpp"
#include <algorithm>    // std::sort
#include <vector>       // std::vector

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

      ~GriddingOperatorMatlabFactory()
      {
		  std::cout << "GOMF destruct " << std::endl;
      }

    public:
				
		// SETTER 
		
		// GETTER
		
		// OPERATIONS

        static GriddingOperatorMatlabFactory& getInstance();
        

	private:
		static GriddingOperatorMatlabFactory instance;
		
    };

}

#endif //GRIDDING_OPERATOR_MATLABFACTORY_H_INCLUDED
