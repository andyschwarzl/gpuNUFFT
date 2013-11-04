#include <limits.h>
#include "gridding_cpu.hpp"

#include "gtest/gtest.h"
#include "gridding_operator.hpp"

// sort algorithm example
#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <map>

#define EPS 0.0001

TEST(PrecomputationTest, ComputeIsotropicSectorCount) {
	size_t imageWidth = 128; 
	size_t sectorWidth = 8;

	size_t sectors = std::ceil(static_cast<float>(imageWidth) / sectorWidth);
	EXPECT_EQ(16,sectors);

	imageWidth = 124; 
	sectorWidth = 8;
    sectors = std::ceil(static_cast<float>(imageWidth) / sectorWidth);
	EXPECT_EQ(16,sectors);

	imageWidth = 7; 
	sectorWidth = 8;
    sectors = std::ceil(static_cast<float>(imageWidth) / sectorWidth);
	EXPECT_EQ(1,sectors);

	imageWidth = 120; 
	sectorWidth = 8;
    sectors = std::ceil(static_cast<float>(imageWidth) / sectorWidth);
	EXPECT_EQ(15,sectors);
}

TEST(PrecomputationTest, ComputeIsotropicSectorDim) {
	size_t imageWidth = 128; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

    GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)(imageWidth * osr);

	size_t sectors = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	size_t sectorDim = std::ceil(static_cast<float>(gridDim.width) / sectorWidth) * 
		               std::ceil(static_cast<float>(gridDim.height) / sectorWidth) *
	                   std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);
	EXPECT_EQ(16*osr,sectors);
	
	size_t expected = 16*16*16*osr*osr*osr;
	EXPECT_EQ(expected,sectorDim);	

	GriddingND::Dimensions sectorDims;
	sectorDims.width = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	sectorDims.height = std::ceil(static_cast<float>(gridDim.height) / sectorWidth);
	sectorDims.depth = std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);

	EXPECT_EQ(expected,sectorDims.count());
}

TEST(PrecomputationTest, ComputeAnisotropicSectorDim) {
	size_t imageWidth = 128; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

    GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)((imageWidth-16) * osr);

	size_t sectorDim = std::ceil(static_cast<float>(gridDim.width) / sectorWidth) * 
		               std::ceil(static_cast<float>(gridDim.height) / sectorWidth) *
	                   std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);
	 
	size_t expected = 16*16*14*osr*osr*osr;
	EXPECT_EQ(expected,sectorDim);	

	GriddingND::Dimensions sectorDims;
	sectorDims.width = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	sectorDims.height = std::ceil(static_cast<float>(gridDim.height) / sectorWidth);
	sectorDims.depth = std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);

	EXPECT_EQ(expected,sectorDims.count());
}

TEST(PrecomputationTest, ComputeSectorRanges) {
	size_t imageWidth = 128; 
	DType osr = 1.0;
	size_t sectorWidth = 8;

	GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)(imageWidth * osr);

	GriddingND::Dimensions sectorDims;
	sectorDims.width = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	sectorDims.height = std::ceil(static_cast<float>(gridDim.height) / sectorWidth);
	sectorDims.depth = std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);

	DType expected[17] = {-0.5000,-0.4375,-0.3750,-0.3125,-0.2500,-0.1875,-0.1250,-0.0625,0,0.0625,0.1250,0.1875,0.2500,0.3125,0.3750,0.4375,0.5000};

	DType* sectorRange = (DType*)malloc((sectorDims.width +1) * sizeof(DType));
	//linspace in range from -0.5 to 0.5
	for (int i=0; i <= sectorDims.width; i++)
	{
		sectorRange[i] = -0.5 + i*(static_cast<DType>(1.0) / (sectorDims.width));
		printf("%5.4f ",sectorRange[i]);
		EXPECT_NEAR(sectorRange[i],expected[i],EPS);
	}
	std::cout << std::endl;

	free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors1D) {
	size_t imageWidth = 16; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

	const size_t coordCnt = 6;
	
	// Coords as StructureOfArrays
	// i.e. first x-vals, then y-vals and z-vals
	DType coords[coordCnt] = {-0.5,-0.3,-0.1, 0.1, 0.3, 0.5};//x

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = coordCnt;

	GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)(imageWidth * osr);

	GriddingND::Dimensions sectorDims;
	sectorDims.width = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	sectorDims.height = std::ceil(static_cast<float>(gridDim.height) / sectorWidth);
	sectorDims.depth = std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);

	DType expected[4] = {-0.5000,-0.16666,0.16666,0.5000};

	DType* sectorRange = (DType*)malloc((sectorDims.width +1) * sizeof(DType));
	//linspace in range from -0.5 to 0.5
	for (int i=0; i <= sectorDims.width; i++)
	{
		sectorRange[i] = -0.5 + i*(static_cast<DType>(1.0) / (sectorDims.width));
		printf("%5.4f ",sectorRange[i]);
		EXPECT_NEAR(sectorRange[i],expected[i],EPS);
	}
	std::cout << std::endl;
	
	size_t expectedSec[6] = {0,0,1,1,2,2};

	for (int cCnt = 0; cCnt < coordCnt; cCnt++)
	{
		DType x = kSpaceData.data[cCnt];
		std::cout << "processing x var: " << x << std::endl;
		size_t sector = std::floor(static_cast<float>(x + 0.5) * sectorDims.width);
		if (sector == sectorDims.width) 
			sector--;
		std::cout << "into sector : " << sector << std::endl;
		EXPECT_EQ(expectedSec[cCnt],sector);
	}

	free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors1DOnBorders) {
	size_t imageWidth = 16; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

	const size_t coordCnt = 4;
	
	// Coords as StructureOfArrays
	// i.e. first x-vals, then y-vals and z-vals
	DType coords[coordCnt] = {-0.5,-0.1666,0.1666,0.5};//x

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = coordCnt;

	GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)(imageWidth * osr);

	GriddingND::Dimensions sectorDims;
	sectorDims.width = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	sectorDims.height = std::ceil(static_cast<float>(gridDim.height) / sectorWidth);
	sectorDims.depth = std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);

	DType expected[4] = {-0.5000,-0.16666,0.16666,0.5000};

	DType* sectorRange = (DType*)malloc((sectorDims.width +1) * sizeof(DType));
	//linspace in range from -0.5 to 0.5
	for (int i=0; i <= sectorDims.width; i++)
	{
		sectorRange[i] = -0.5 + i*(static_cast<DType>(1.0) / (sectorDims.width));
		printf("%5.4f ",sectorRange[i]);
		EXPECT_NEAR(sectorRange[i],expected[i],EPS);
	}
	std::cout << std::endl;
	
	size_t expectedSec[4] = {0,1,2,2};

	for (int cCnt = 0; cCnt < coordCnt; cCnt++)
	{
		DType x = kSpaceData.data[cCnt];
		std::cout << "processing x var: " << x << std::endl;
		size_t sector = std::round(static_cast<float>(x + 0.5) * sectorDims.width);
		if (sector == sectorDims.width) 
			sector--;
		std::cout << "into sector : " << sector << std::endl;
		EXPECT_EQ(expectedSec[cCnt],sector);
	}

	free(sectorRange);
}


TEST(PrecomputationTest, AssignSectors2D) {
	size_t imageWidth = 16; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

	const size_t coordCnt = 6;
	
	// Coords as StructureOfArrays
	// i.e. first x-vals, then y-vals and z-vals
	DType coords[coordCnt*2] = {-0.5,-0.3,-0.1, 0.1, 0.3, 0.5,//x
	                            -0.5,-0.5,   0,   0, 0.5, 0.45};//y

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = coordCnt;

	GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)(imageWidth * osr);

	GriddingND::Dimensions sectorDims;
	sectorDims.width = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	sectorDims.height = std::ceil(static_cast<float>(gridDim.height) / sectorWidth);
	sectorDims.depth = std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);

	DType expected[4] = {-0.5000,-0.16666,0.16666,0.5000};

	DType* sectorRange = (DType*)malloc((sectorDims.width +1) * sizeof(DType));
	//linspace in range from -0.5 to 0.5
	for (int i=0; i <= sectorDims.width; i++)
	{
		sectorRange[i] = -0.5 + i*(static_cast<DType>(1.0) / (sectorDims.width));
		EXPECT_NEAR(sectorRange[i],expected[i],EPS);
	}
	std::cout << std::endl;
	
	size_t expectedSec[6] = {0,0,4,4,8,8};

	for (int cCnt = 0; cCnt < coordCnt; cCnt++)
	{
		DType2 coord;
		coord.x = kSpaceData.data[cCnt];
		coord.y = kSpaceData.data[cCnt + kSpaceData.count()];
		
		std::cout << "processing x var: " << coord.x << " y: " << coord.y << std::endl;

		size_t x_sector = std::floor(static_cast<float>(coord.x + 0.5) * sectorDims.width);
		if (x_sector == sectorDims.width) 
			x_sector--;

		size_t y_sector = std::floor(static_cast<float>(coord.y + 0.5) * sectorDims.height);
		if (y_sector == sectorDims.height) 
			y_sector--;

		std::cout << "into sector x: " << x_sector << " y: " << y_sector << std::endl;
		EXPECT_EQ(expectedSec[cCnt],x_sector + y_sector * sectorDims.height);
	}

	free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors3D) {
	size_t imageWidth = 16; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

	const size_t coordCnt = 6;
	
	// Coords as StructureOfArrays
	// i.e. first x-vals, then y-vals and z-vals
	DType coords[coordCnt*3] = {-0.5,-0.3,-0.1, 0.1, 0.3, 0.5,//x
	                            -0.5,-0.5,   0,   0, 0.5, 0.45,//y
	                            -0.33,-0.16666,   0,   0, -0.23, 0.45};//z

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = coordCnt;

	GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)(imageWidth * osr);

	GriddingND::Dimensions sectorDims;
	sectorDims.width = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	sectorDims.height = std::ceil(static_cast<float>(gridDim.height) / sectorWidth);
	sectorDims.depth = std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);

	DType expected[4] = {-0.5000,-0.16666,0.16666,0.5000};

	DType* sectorRange = (DType*)malloc((sectorDims.width +1) * sizeof(DType));
	//linspace in range from -0.5 to 0.5
	for (int i=0; i <= sectorDims.width; i++)
	{
		sectorRange[i] = -0.5 + i*(static_cast<DType>(1.0) / (sectorDims.width));
		EXPECT_NEAR(sectorRange[i],expected[i],EPS);
	}
	std::cout << std::endl;
	
	size_t expectedSec[6] = {0,9,13,13,8,26};

	for (int cCnt = 0; cCnt < coordCnt; cCnt++)
	{
		DType3 coord;
		coord.x = kSpaceData.data[cCnt];
		coord.y = kSpaceData.data[cCnt + kSpaceData.count()];
		coord.z = kSpaceData.data[cCnt + 2*kSpaceData.count()];
		
		std::cout << "processing x var: " << coord.x << " y: " << coord.y << " z: " << coord.z  << std::endl;

		size_t x_sector = std::floor(static_cast<float>(coord.x + 0.5) * sectorDims.width);
		if (x_sector == sectorDims.width) 
			x_sector--;

		size_t y_sector = std::floor(static_cast<float>(coord.y + 0.5) * sectorDims.height);
		if (y_sector == sectorDims.height) 
			y_sector--;

		size_t z_sector = std::floor(static_cast<float>(coord.z + 0.5) * sectorDims.depth);
		if (z_sector == sectorDims.depth) 
			z_sector--;

		std::cout << "into sector x: " << x_sector << " y: " << y_sector << " z: " << z_sector << std::endl;
		EXPECT_EQ(expectedSec[cCnt],x_sector + sectorDims.height * (y_sector + sectorDims.depth * z_sector));
	}

	free(sectorRange);
}

bool pairComp (std::pair<size_t,size_t> i,std::pair<size_t,size_t> j) 
{ 
	return (i.second < j.second); 
}

std::vector<std::pair<size_t,size_t>> sortVector(GriddingND::Array<size_t> assignedSectors)
{
	std::vector<std::pair<size_t,size_t>> secVector;
	
	for (size_t i=0; i< assignedSectors.count(); i++)
	  secVector.push_back(std::pair<size_t,size_t>(i,assignedSectors.data[i]));

	// using function as comp
	std::sort (secVector.begin(), secVector.end(), pairComp);

	return secVector;
}

TEST(PrecomputationTest, TestIndexSorting) 
{
  size_t assSectors[6] = {0,9,13,13,8,26};
  size_t expectedSectors[6] = {0,8,9,13,13,26};

  GriddingND::Array<size_t> assignedSectors;
  assignedSectors.data = assSectors;
  assignedSectors.dim.length = 6;

  std::vector<std::pair<size_t,size_t>> secVector = sortVector(assignedSectors);

  // print out content:
  std::cout << "vector contains:";
  for (std::vector<std::pair<size_t,size_t>>::iterator it=secVector.begin(); it!=secVector.end(); ++it)
    std::cout << " " << it->second << " (" << it->first << ") ";
  std::cout << '\n';

  std::pair<size_t,size_t>* sortedArray = &secVector[0];

  //print indices for reselect
  for (size_t i=0; i<6;i++)
  {
	  std::cout << sortedArray[i].first ;
	  EXPECT_EQ(sortedArray[i].second, expectedSectors[i]);
  }
  std::cout << std::endl;
}

TEST(PrecomputationTest, AssignSectors3DSorted) {
	size_t imageWidth = 16; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

	const size_t coordCnt = 6;
	
	// Coords as StructureOfArrays
	// i.e. first x-vals, then y-vals and z-vals
	DType coords[coordCnt*3] = {-0.5,-0.3,-0.1, 0.1, 0.3, 0.5,//x
	                            -0.5,-0.5,   0,   0, 0.5, 0.45,//y
	                            -0.33,-0.16666,   0,   0, -0.23, 0.45};//z

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = coordCnt;

	GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)(imageWidth * osr);

	GriddingND::Dimensions sectorDims;
	sectorDims.width = std::ceil(static_cast<float>(gridDim.width) / sectorWidth);
	sectorDims.height = std::ceil(static_cast<float>(gridDim.height) / sectorWidth);
	sectorDims.depth = std::ceil(static_cast<float>(gridDim.depth) / sectorWidth);

	size_t expectedSec[6] = {0,9,13,13,8,26};

	GriddingND::Array<size_t> assignedSectors;
    assignedSectors.data = (size_t*)malloc(coordCnt * sizeof(size_t));
    assignedSectors.dim.length = coordCnt;

	for (int cCnt = 0; cCnt < coordCnt; cCnt++)
	{
		DType3 coord;
		coord.x = kSpaceData.data[cCnt];
		coord.y = kSpaceData.data[cCnt + kSpaceData.count()];
		coord.z = kSpaceData.data[cCnt + 2*kSpaceData.count()];

		size_t x_sector = std::floor(static_cast<float>(coord.x + 0.5) * sectorDims.width);
		if (x_sector == sectorDims.width) 
			x_sector--;

		size_t y_sector = std::floor(static_cast<float>(coord.y + 0.5) * sectorDims.height);
		if (y_sector == sectorDims.height) 
			y_sector--;

		size_t z_sector = std::floor(static_cast<float>(coord.z + 0.5) * sectorDims.depth);
		if (z_sector == sectorDims.depth) 
			z_sector--;
		size_t sector = x_sector + sectorDims.height * (y_sector + sectorDims.depth * z_sector);
		assignedSectors.data[cCnt] = sector;
		EXPECT_EQ(expectedSec[cCnt],sector);
	}

	size_t expectedSecSorted[6] = {0,8,9,13,13,26};
	size_t expectedSecIndexSorted[6] = {0,4,1,2,3,5};

    std::vector<std::pair<size_t,size_t>> secVector = sortVector(assignedSectors);
	
	for (int i=0; i<assignedSectors.count();i++)
	{
		//compare index
		EXPECT_EQ(expectedSecIndexSorted[i],secVector[i].first);
		EXPECT_EQ(expectedSecSorted[i],secVector[i].second);
	}

	free(assignedSectors.data);
}