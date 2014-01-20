#ifndef GRIDDING_OPERATOR_FACTORY_H_INCLUDED
#define GRIDDING_OPERATOR_FACTORY_H_INCLUDED

#include "config.hpp"
#include "gridding_operator.hpp"
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <string>

#include "cuda_utils.hpp"

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

		template <typename T> 
		Array<T> initLinArray(IndType arrCount);

		virtual Array<IndType> initDataIndices(GriddingOperator* griddingOp, IndType coordCnt);
		virtual Array<IndType> initSectorDataCount(GriddingOperator* griddingOp, IndType coordCnt);
		virtual Array<DType> initDensData(GriddingOperator* griddingOp, IndType coordCnt);
		virtual Array<DType> initCoordsData(GriddingOperator* griddingOp, IndType coordCnt);
		virtual Array<IndType3> initSectorCenters(GriddingOperator* griddingOp, IndType sectorCnt);
		virtual Array<IndType2> initSectorCenters2D(GriddingOperator* griddingOp, IndType sectorCnt);
		virtual void debug(const std::string& message);

		IndType computeSectorMapping(DType coord, IndType sectorCount);
		
		IndType3 computeSectorMapping(DType3 coord, Dimensions sectorDims);

		IndType2 computeSectorMapping(DType2 coord, Dimensions sectorDims);

		IndType computeXYZ2Lin(IndType x, IndType y, IndType z, Dimensions dim);
		
		IndType computeXY2Lin(IndType x, IndType y, Dimensions dim);

		IndType computeInd32Lin(IndType3 sector, Dimensions dim);

		IndType computeInd22Lin(IndType2 sector, Dimensions dim);

		IndType computeSectorCountPerDimension(IndType dim, IndType sectorWidth);

		Dimensions computeSectorCountPerDimension(Dimensions dim, IndType sectorWidth);

		IndType computeTotalSectorCount(Dimensions dim, IndType sectorWidth);

		template <typename T>
		std::vector<IndPair> sortVector(Array<T> assignedSectors);

		Array<IndType> computeSectorDataCount(GriddingND::GriddingOperator *griddingOp,GriddingND::Array<IndType> assignedSectors);

		Array<IndType3> computeSectorCenters(GriddingOperator *griddingOp);
		Array<IndType2> computeSectorCenters2D(GriddingOperator *griddingOp);

		IndType computeSectorCenter(IndType var, IndType sectorWidth);

    public:

        GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

		GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

		GriddingOperator* createGriddingOperator(Array<DType>& kSpaceTraj, Array<DType>& densCompData, Array<DType2>& sensData, const IndType& kernelWidth, const IndType& sectorWidth, const DType& osf, Dimensions& imgDims);

        static GriddingOperatorFactory& getInstance();
        
	private:
		static GriddingOperatorFactory instance;
	};

}

#endif //GRIDDING_OPERATOR_FACTORY_H_INCLUDED
