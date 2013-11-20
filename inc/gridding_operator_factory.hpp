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
	// * new calculation of "data - sector" mapping, sorting etc.
	//  
	// * reuse of previously calculated mapping
	// 
	class GriddingOperatorFactory
	{
    protected:
      GriddingOperatorFactory()
      {
      }

      ~GriddingOperatorFactory()
      {
		std::cout << "GOF destruct " << std::endl;
      }

	  Array<IndType> assignSectors(GriddingOperator* griddingOp, Array<DType>& kSpaceTraj);

	  virtual Array<IndType> initDataIndices(GriddingOperator* griddingOp, size_t coordCnt);
	  virtual Array<IndType> initSectorDataCount(GriddingOperator* griddingOp, size_t coordCnt);
	  virtual Array<IndType> initDensData(GriddingOperator* griddingOp, size_t coordCnt);
	  virtual Array<DType> initCoordsData(GriddingOperator* griddingOp, size_t coordCnt);
	  virtual Array<IndType3> initSectorCenters(GriddingOperator* griddingOp, size_t sectorCnt);

    public:

        GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, Dimensions& imgDims);

		GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, Dimensions& imgDims);

		GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType>& sensData, const size_t& kernelWidth, const size_t& sectorWidth, const DType& osf, Dimensions& imgDims);

        static GriddingOperatorFactory& getInstance();
        
	private:
		static GriddingOperatorFactory instance;
		
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

		IndType computeSectorCenter(IndType var, IndType sectorWidth);
	};

}

#endif //GRIDDING_OPERATOR_FACTORY_H_INCLUDED
