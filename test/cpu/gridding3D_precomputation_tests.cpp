#include <limits.h>
#include "gridding_cpu.hpp"

#include "gtest/gtest.h"
#include "gridding_operator.hpp"

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