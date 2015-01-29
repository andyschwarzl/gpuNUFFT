#include <limits.h>
#include "gpuNUFFT_cpu.hpp"

#include "gtest/gtest.h"
#include "gpuNUFFT_operator_factory.hpp"

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
	DType coords[coordCnt*3] = {(DType)-0.5,(DType)-0.3,(DType)-0.1, (DType)0.1, (DType)0.3, (DType)0.5,//x
	                            (DType)-0.5,(DType)-0.5,   0,   0, (DType)0.5, (DType)0.45,//y
	                            (DType)-0.33,(DType)-0.16666,   0,   0, (DType)-0.23, (DType)0.45};//z

	gpuNUFFT::Array<DType> kSpaceTraj;
    kSpaceTraj.data = coords;
    kSpaceTraj.dim.length = coordCnt;

	gpuNUFFT::Dimensions imgDims;
	imgDims.width = imageWidth;
	imgDims.height = imageWidth;
	imgDims.depth = imageWidth;
  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,false,false);
	gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);

	EXPECT_TRUE(gpuNUFFTOp != NULL);

	gpuNUFFT::Dimensions gridDims = gpuNUFFTOp->getGridDims();
	EXPECT_EQ(gridDims.width,imgDims.width * osf);

	gpuNUFFT::Array<IndType> sectorDataCount = gpuNUFFTOp->getSectorDataCount();
	IndType sectorDataCountExpected[29] = {0,1,1,1,1,1,1,1,1,2,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6};
	for (unsigned i=0; i<sectorDataCount.count(); i++)
	{	
		EXPECT_EQ(sectorDataCountExpected[i],sectorDataCount.data[i]);
	}

	gpuNUFFT::Dimensions sectorDims = gpuNUFFTOp->getGridSectorDims();
	IndType expected = 3*3*3;
	EXPECT_EQ(expected,sectorDims.count());	

	gpuNUFFT::Array<IndType> sectorCenters = gpuNUFFTOp->getSectorCenters();
	EXPECT_EQ(4u,sectorCenters.data[0]);
	EXPECT_EQ(4u,sectorCenters.data[1]);
	EXPECT_EQ(4u,sectorCenters.data[2]);

	gpuNUFFT::Array<IndType> dataIndices = gpuNUFFTOp->getDataIndices();
	IndType expectedSecIndexSorted[6] = {0,4,1,2,3,5};
	
	for (unsigned i=0; i<dataIndices.count(); i++)
	{	
		EXPECT_EQ(expectedSecIndexSorted[i],dataIndices.data[i]);
	}

	gpuNUFFT::Array<DType> sortedCoords = gpuNUFFTOp->getKSpaceTraj();
	EXPECT_NEAR(-0.5,sortedCoords.data[0],EPS);
	EXPECT_NEAR(-0.5,sortedCoords.data[6],EPS);
	EXPECT_NEAR(-0.33,sortedCoords.data[12],EPS);
	delete gpuNUFFTOp;
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
	DType coords[coordCnt*3] = {(DType)-0.5,(DType)-0.3,(DType)-0.1, (DType)0.1, (DType)0.3, (DType)0.5,//x
	                            (DType)-0.5,(DType)-0.5,   0,   0, (DType)0.5, (DType)0.45,//y
	                            (DType)-0.33,(DType)-0.16666,   0,   0, (DType)-0.23, (DType)0.45};//z

	gpuNUFFT::Array<DType> kSpaceTraj;
    kSpaceTraj.data = coords;
    kSpaceTraj.dim.length = coordCnt;
	kSpaceTraj.dim.channels = 3;

	gpuNUFFT::Dimensions imgDims;
	imgDims.width = imageWidth;
	imgDims.height = imageWidth;
	imgDims.depth = imageWidth;

	EXPECT_THROW({
    gpuNUFFT::GpuNUFFTOperatorFactory factory(false,false,false);
	factory.createGpuNUFFTOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);
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
	DType coords[coordCnt*2] = {(DType)-0.5,(DType)-0.3,(DType)-0.1, (DType)0.1, (DType)0.3, (DType)0.5,//x
								(DType)-0.5,(DType)-0.5,   0,   0, (DType)0.5, (DType)0.45};//y}

	gpuNUFFT::Array<DType> kSpaceTraj;
    kSpaceTraj.data = coords;
    kSpaceTraj.dim.length = coordCnt;

	gpuNUFFT::Dimensions imgDims(imageWidth,imageWidth);
  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,false,false);
	gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceTraj, kernelWidth, sectorWidth, osf, imgDims);

	EXPECT_TRUE(gpuNUFFTOp != NULL);

	gpuNUFFT::Dimensions gridDims = gpuNUFFTOp->getGridDims();
	EXPECT_EQ(gridDims.width,imgDims.width * osf);
	EXPECT_EQ(gridDims.count(),imgDims.width * osf * imgDims.width * osf);

	gpuNUFFT::Array<IndType> sectorDataCount = gpuNUFFTOp->getSectorDataCount();
	IndType sectorDataCountExpected[11] = {0,2,2,2,2,4,4,4,4,6,6};
	for (unsigned i=0; i<sectorDataCount.count(); i++)
	{	
		EXPECT_EQ(sectorDataCountExpected[i],sectorDataCount.data[i]);
	}
	
	gpuNUFFT::Dimensions sectorDims = gpuNUFFTOp->getGridSectorDims();
	IndType expected = 3*3;
	EXPECT_EQ(expected,sectorDims.count());	
	
	gpuNUFFT::Array<IndType> sectorCenters = gpuNUFFTOp->getSectorCenters();
	EXPECT_EQ(4u,sectorCenters.data[0]);
	EXPECT_EQ(4u,sectorCenters.data[1]);
	/*
	gpuNUFFT::Array<IndType> dataIndices = gpuNUFFTOp->getDataIndices();
	IndType expectedSecIndexSorted[6] = {0,4,1,2,3,5};
	
	for (int i=0; i<dataIndices.count(); i++)
	{	
		EXPECT_EQ(expectedSecIndexSorted[i],dataIndices.data[i]);
	}

	gpuNUFFT::Array<DType> sortedCoords = gpuNUFFTOp->getKSpaceTraj();
	EXPECT_NEAR(-0.5,sortedCoords.data[0],EPS);
	EXPECT_NEAR(-0.5,sortedCoords.data[6],EPS);
	EXPECT_NEAR(-0.33,sortedCoords.data[12],EPS);*/
	delete gpuNUFFTOp;
}
