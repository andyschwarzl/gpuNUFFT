#include <limits.h>
#include "gridding_cpu.hpp"

#include "gtest/gtest.h"
#include "gridding_operator_factory.hpp"

// sort algorithm example
#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <map>

#include <cmath>
#include <stdexcept>

#define EPS 0.0001

TEST(OperatorFactoryTest,TestInit)
{
	IndType imageWidth = 16; 
	DType osf = 1.5;
	IndType sectorWidth = 8;
	IndType kernelWidth = 3;

	const IndType coordCnt = 6;
	
	// Coords as StructureOfArrays
	// i.e. first x-vals, then y-vals and z-vals
	DType coords[coordCnt*3] = {-0.5,-0.3,-0.1, 0.1, 0.3, 0.5,//x
	                            -0.5,-0.5,   0,   0, 0.5, 0.45,//y
	                            -0.33,-0.16666,   0,   0, -0.23, 0.45};//z

	GriddingND::Array<DType> kSpaceTraj;
    kSpaceTraj.data = coords;
    kSpaceTraj.dim.length = coordCnt;

	GriddingND::Dimensions imgDims;
	imgDims.width = imageWidth;
	imgDims.height = imageWidth;
	imgDims.depth = imageWidth;
  GriddingND::GriddingOperatorFactory factory;
	GriddingND::GriddingOperator *griddingOp = factory.createGriddingOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);

	EXPECT_TRUE(griddingOp != NULL);

	GriddingND::Dimensions gridDims = griddingOp->getGridDims();
	EXPECT_EQ(gridDims.width,imgDims.width * osf);

	GriddingND::Array<IndType> sectorDataCount = griddingOp->getSectorDataCount();
	IndType sectorDataCountExpected[29] = {0,1,1,1,1,1,1,1,1,2,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6};
	for (int i=0; i<sectorDataCount.count(); i++)
	{	
		EXPECT_EQ(sectorDataCountExpected[i],sectorDataCount.data[i]);
	}

	GriddingND::Dimensions sectorDims = griddingOp->getGridSectorDims();
	IndType expected = 3*3*3;
	EXPECT_EQ(expected,sectorDims.count());	

	GriddingND::Array<IndType> sectorCenters = griddingOp->getSectorCenters();
	EXPECT_EQ(4,sectorCenters.data[0]);
	EXPECT_EQ(4,sectorCenters.data[1]);
	EXPECT_EQ(4,sectorCenters.data[2]);

	GriddingND::Array<IndType> dataIndices = griddingOp->getDataIndices();
	IndType expectedSecIndexSorted[6] = {0,4,1,2,3,5};
	
	for (int i=0; i<dataIndices.count(); i++)
	{	
		EXPECT_EQ(expectedSecIndexSorted[i],dataIndices.data[i]);
	}

	GriddingND::Array<DType> sortedCoords = griddingOp->getKSpaceTraj();
	EXPECT_NEAR(-0.5,sortedCoords.data[0],EPS);
	EXPECT_NEAR(-0.5,sortedCoords.data[6],EPS);
	EXPECT_NEAR(-0.33,sortedCoords.data[12],EPS);
	delete griddingOp;
}

TEST(OperatorFactoryTest,TestInvalidArgumentInit)
{
	IndType imageWidth = 16; 
	DType osf = 1.5;
	IndType sectorWidth = 8;
	IndType kernelWidth = 3;

	const IndType coordCnt = 6;
	
	// Coords as StructureOfArrays
	// i.e. first x-vals, then y-vals and z-vals
	DType coords[coordCnt*3] = {-0.5,-0.3,-0.1, 0.1, 0.3, 0.5,//x
	                            -0.5,-0.5,   0,   0, 0.5, 0.45,//y
	                            -0.33,-0.16666,   0,   0, -0.23, 0.45};//z

	GriddingND::Array<DType> kSpaceTraj;
    kSpaceTraj.data = coords;
    kSpaceTraj.dim.length = coordCnt;
	kSpaceTraj.dim.channels = 3;

	GriddingND::Dimensions imgDims;
	imgDims.width = imageWidth;
	imgDims.height = imageWidth;
	imgDims.depth = imageWidth;

	EXPECT_THROW({
    GriddingND::GriddingOperatorFactory factory;
	GriddingND::GriddingOperator *griddingOp = factory.createGriddingOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);
	},std::invalid_argument);
}

TEST(OperatorFactoryTest,Test2DInit)
{
	IndType imageWidth = 16; 
	DType osf = 1.5;
	IndType sectorWidth = 8;
	IndType kernelWidth = 3;

	const IndType coordCnt = 6;
	
	// Coords as StructureOfArrays
	// i.e. first x-vals, then y-vals and z-vals
	DType coords[coordCnt*2] = {-0.5,-0.3,-0.1, 0.1, 0.3, 0.5,//x
								-0.5,-0.5,   0,   0, 0.5, 0.45};//y}

	GriddingND::Array<DType> kSpaceTraj;
    kSpaceTraj.data = coords;
    kSpaceTraj.dim.length = coordCnt;

	GriddingND::Dimensions imgDims(imageWidth,imageWidth);
  GriddingND::GriddingOperatorFactory factory;
	GriddingND::GriddingOperator *griddingOp = factory.createGriddingOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);

	EXPECT_TRUE(griddingOp != NULL);

	GriddingND::Dimensions gridDims = griddingOp->getGridDims();
	EXPECT_EQ(gridDims.width,imgDims.width * osf);
	EXPECT_EQ(gridDims.count(),imgDims.width * osf * imgDims.width * osf);

	GriddingND::Array<IndType> sectorDataCount = griddingOp->getSectorDataCount();
	IndType sectorDataCountExpected[11] = {0,2,2,2,2,4,4,4,4,6,6};
	for (int i=0; i<sectorDataCount.count(); i++)
	{	
		EXPECT_EQ(sectorDataCountExpected[i],sectorDataCount.data[i]);
	}
	
	GriddingND::Dimensions sectorDims = griddingOp->getGridSectorDims();
	IndType expected = 3*3;
	EXPECT_EQ(expected,sectorDims.count());	
	
	GriddingND::Array<IndType> sectorCenters = griddingOp->getSectorCenters();
	EXPECT_EQ(4,sectorCenters.data[0]);
	EXPECT_EQ(4,sectorCenters.data[1]);
	/*
	GriddingND::Array<IndType> dataIndices = griddingOp->getDataIndices();
	IndType expectedSecIndexSorted[6] = {0,4,1,2,3,5};
	
	for (int i=0; i<dataIndices.count(); i++)
	{	
		EXPECT_EQ(expectedSecIndexSorted[i],dataIndices.data[i]);
	}

	GriddingND::Array<DType> sortedCoords = griddingOp->getKSpaceTraj();
	EXPECT_NEAR(-0.5,sortedCoords.data[0],EPS);
	EXPECT_NEAR(-0.5,sortedCoords.data[6],EPS);
	EXPECT_NEAR(-0.33,sortedCoords.data[12],EPS);*/
	delete griddingOp;
}
