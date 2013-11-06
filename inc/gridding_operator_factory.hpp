#ifndef GRIDDING_OPERATOR_FACTORY_H_INCLUDED
#define GRIDDING_OPERATOR_FACTORY_H_INCLUDED

#include "config.hpp"
#include "gridding_operator.hpp"
#include <algorithm>    // std::sort
#include <vector>       // std::vector

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

	  Array<IndType> assignSectors(GriddingOperator* griddingOp, Array<DType> kSpaceCoords);
    public:

        GriddingOperator* createGriddingOperator(Array<DType> kSpaceData, size_t kernelWidth, size_t sectorWidth, DType osf, Dimensions imgDims);

		// SETTER 
		
		// GETTER
		
		// OPERATIONS

        static GriddingOperatorFactory* getInstance();
        
	private:
		static GriddingOperatorFactory* instance;
		
		size_t computeSectorMapping(DType coord, size_t sectorCount);
		IndType3 computeSectorMapping(DType3 coord, Dimensions sectorDims);
		IndType2 computeSectorMapping(DType2 coord, Dimensions sectorDims);
		size_t computeXYZ2Lin(size_t x, size_t y, size_t z, Dimensions dim);
		size_t computeInd32Lin(IndType3 sector, Dimensions dim);
		size_t computeSectorCountPerDimension(size_t dim, size_t sectorWidth);
		Dimensions computeSectorCountPerDimension(Dimensions dim, size_t sectorWidth);
		size_t computeTotalSectorCount(Dimensions dim, size_t sectorWidth);
		std::vector<IndPair> sortVector(Array<size_t> assignedSectors);
		Array<IndType> computeSectorDataCount(GriddingND::GriddingOperator *griddingOp,GriddingND::Array<IndType> assignedSectors);
		Array<IndType3> computeSectorCenters(GriddingOperator *griddingOp);
    };

}

#endif //GRIDDING_OPERATOR_FACTORY_H_INCLUDED
