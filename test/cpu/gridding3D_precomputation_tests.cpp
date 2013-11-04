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

size_t computeSectorCountPerDimension(size_t dim, size_t sectorWidth)
{
	return std::ceil(static_cast<DType>(dim) / sectorWidth);
}

GriddingND::Dimensions computeSectorCountPerDimension(GriddingND::Dimensions dim, size_t sectorWidth)
{
	GriddingND::Dimensions sectorDims;
	sectorDims.width = computeSectorCountPerDimension(dim.width,sectorWidth);
	sectorDims.height = computeSectorCountPerDimension(dim.height,sectorWidth);
	sectorDims.depth = computeSectorCountPerDimension(dim.depth,sectorWidth);
	return sectorDims;
}

size_t computeTotalSectorCount(GriddingND::Dimensions dim, size_t sectorWidth)
{
	return computeSectorCountPerDimension(dim,sectorWidth).count();
}

TEST(PrecomputationTest, ComputeIsotropicSectorCount) {
	size_t imageWidth = 128; 
	size_t sectorWidth = 8;

	size_t sectors = computeSectorCountPerDimension(imageWidth,sectorWidth);
	EXPECT_EQ(16,sectors);

	imageWidth = 124; 
	sectorWidth = 8;
    sectors = computeSectorCountPerDimension(imageWidth,sectorWidth);
	EXPECT_EQ(16,sectors);

	imageWidth = 7; 
	sectorWidth = 8;
    sectors = computeSectorCountPerDimension(imageWidth,sectorWidth);
	EXPECT_EQ(1,sectors);

	imageWidth = 120; 
	sectorWidth = 8;
    sectors = computeSectorCountPerDimension(imageWidth,sectorWidth);
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

	size_t sectors = computeSectorCountPerDimension(gridDim.width, sectorWidth);
	size_t sectorDim = computeTotalSectorCount(gridDim,sectorWidth);
	EXPECT_EQ(16*osr,sectors);
	
	size_t expected = 16*16*16*osr*osr*osr;
	EXPECT_EQ(expected,sectorDim);	

	GriddingND::Dimensions sectorDims = computeSectorCountPerDimension(gridDim,sectorWidth);

	EXPECT_EQ(expected,sectorDims.count());
}

TEST(PrecomputationTest, ComputeAnisotropicSectorDim) {
	size_t imageWidth = 128; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

    GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth);
	gridDim.height = (size_t)(imageWidth);
	gridDim.depth = (size_t)((imageWidth-16));
	
	std::cout << " dimensions before: w: " << gridDim.width << " h: " << gridDim.height << " d: " << gridDim.depth << std::endl;
	gridDim = gridDim * osr;
	std::cout << " dimensions scaled: w: " << gridDim.width << " h: " << gridDim.height << " d: " << gridDim.depth << std::endl;

	size_t sectorDim = computeTotalSectorCount(gridDim,sectorWidth);
	 
	size_t expected = 16*16*14*osr*osr*osr;
	EXPECT_EQ(expected,sectorDim);	

	GriddingND::Dimensions sectorDims = computeSectorCountPerDimension(gridDim,sectorWidth);

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

	GriddingND::Dimensions sectorDims = computeSectorCountPerDimension(gridDim,sectorWidth);

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

	GriddingND::Dimensions sectorDims = computeSectorCountPerDimension(gridDim,sectorWidth);

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

	GriddingND::Dimensions sectorDims= computeSectorCountPerDimension(gridDim,sectorWidth);

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

	GriddingND::Dimensions sectorDims= computeSectorCountPerDimension(gridDim,sectorWidth);

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

size_t computeSectorMapping(DType coord, size_t sectorCount)
{
	size_t sector = std::floor(static_cast<DType>(coord + 0.5) * sectorCount);
	if (sector == sectorCount) 
		sector--;
	return sector;
}

IndType3 computeSectorMapping(DType3 coord, GriddingND::Dimensions sectorDims)
{
	IndType3 sector;
	sector.x = computeSectorMapping(coord.x,sectorDims.width);
	sector.y  = computeSectorMapping(coord.y,sectorDims.height);
	sector.z  = computeSectorMapping(coord.z,sectorDims.depth);
	return sector;
}

IndType2 computeSectorMapping(DType2 coord, GriddingND::Dimensions sectorDims)
{
	IndType2 sector;
	sector.x = computeSectorMapping(coord.x,sectorDims.width);
	sector.y  = computeSectorMapping(coord.y,sectorDims.height);
	return sector;
}

size_t computeXYZ2Lin(size_t x, size_t y, size_t z, GriddingND::Dimensions dim)
{
	return x + dim.height * (y + dim.depth * z);
}

size_t computeInd32Lin(IndType3 sector, GriddingND::Dimensions dim)
{
	return sector.x + dim.height * (sector.y + dim.depth * sector.z);
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

	GriddingND::Dimensions sectorDims= computeSectorCountPerDimension(gridDim,sectorWidth);

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

		size_t x_sector = computeSectorMapping(coord.x,sectorDims.width);
		size_t y_sector = computeSectorMapping(coord.y,sectorDims.height);
		size_t z_sector = computeSectorMapping(coord.z,sectorDims.depth);

		std::cout << "into sector x: " << x_sector << " y: " << y_sector << " z: " << z_sector << std::endl;
		EXPECT_EQ(expectedSec[cCnt],computeXYZ2Lin(x_sector,y_sector,z_sector,sectorDims));

		IndType3 mappedSectors = computeSectorMapping(coord,sectorDims);
		EXPECT_EQ(expectedSec[cCnt],computeInd32Lin(mappedSectors,sectorDims));
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

	GriddingND::Dimensions sectorDims= computeSectorCountPerDimension(gridDim,sectorWidth);

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

		IndType3 mappedSectors = computeSectorMapping(coord,sectorDims);

		size_t sector = computeInd32Lin(mappedSectors,sectorDims);
		assignedSectors.data[cCnt] = sector;
		EXPECT_EQ(expectedSec[cCnt],sector);
	}

	size_t expectedSecSorted[6] = {0,8,9,13,13,26};
	size_t expectedSecIndexSorted[6] = {0,4,1,2,3,5};

    std::vector<std::pair<size_t,size_t>> secVector = sortVector(assignedSectors);
	
	DType coords_sorted[coordCnt*3];

	for (int i=0; i<assignedSectors.count();i++)
	{
		//compare index
		EXPECT_EQ(expectedSecIndexSorted[i],secVector[i].first);
		EXPECT_EQ(expectedSecSorted[i],secVector[i].second);
		coords_sorted[i] = kSpaceData.data[secVector[i].first];
		coords_sorted[i + 1*coordCnt] = kSpaceData.data[secVector[i].first + 1*coordCnt];
		coords_sorted[i + 2*coordCnt] = kSpaceData.data[secVector[i].first + 2*coordCnt];
	}

	for (int i=0;i<kSpaceData.count();i++)
	{
		std::cout << " x: "  << coords_sorted[i] << " y: " << coords_sorted[i+ 1*coordCnt] << " z:" << coords_sorted[i+ 2*coordCnt] << std::endl;
	}

	free(assignedSectors.data);
}


TEST(PrecomputationTest, ComputeSectorCenters) {
	size_t imageWidth = 16; 
	DType osr = 1.5;
	size_t sectorWidth = 8;

	GriddingND::Dimensions gridDim;
	gridDim.width = (size_t)(imageWidth * osr);
	gridDim.height = (size_t)(imageWidth * osr);
	gridDim.depth = (size_t)(imageWidth * osr);
	
	GriddingND::Dimensions sectorDims= computeSectorCountPerDimension(gridDim,sectorWidth);

	GriddingND::Array<IndType3> sectorCenters; 
	sectorCenters.data = (IndType3*)malloc(sectorDims.count() * sizeof(IndType3));
	sectorCenters.dim.length = sectorDims.count();

	for (size_t z=0;z<sectorDims.depth; z++)
		for (size_t y=0;y<sectorDims.height;y++)
			for (size_t x=0;x<sectorDims.width;x++)
			{
				IndType3 center;
				center.x = x*sectorWidth +  std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
				center.y = y*sectorWidth +  std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
				center.z = z*sectorWidth +  std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
				sectorCenters.data[computeXYZ2Lin(x,y,z,sectorDims)] = center;
			}

	for (int i=0; i<sectorDims.count(); i++)
	{
		std::cout << " x: " << sectorCenters.data[i].x << " y: " << sectorCenters.data[i].y << " z: " << sectorCenters.data[i].z << std::endl;
	}

	free(sectorCenters.data);
}