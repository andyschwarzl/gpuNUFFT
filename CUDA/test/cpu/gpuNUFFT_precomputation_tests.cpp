#include <limits.h>
#include "gpuNUFFT_cpu.hpp"

#include "gtest/gtest.h"
#include "gpuNUFFT_operator.hpp"
#include "precomp_utils.hpp"

// sort algorithm example
#include <iostream>   // std::cout
#include <algorithm>  // std::sort
#include <vector>     // std::vector
#include <map>

#include <cmath>
#include <stdexcept>

#define EPS 0.0001

IndType computeSectorCountPerDimension(IndType dim, IndType sectorWidth)
{
  return (IndType)std::ceil(static_cast<DType>(dim) / sectorWidth);
}

gpuNUFFT::Dimensions computeSectorCountPerDimension(gpuNUFFT::Dimensions dim,
                                                    IndType sectorWidth)
{
  gpuNUFFT::Dimensions sectorDims;
  sectorDims.width = computeSectorCountPerDimension(dim.width, sectorWidth);
  sectorDims.height = computeSectorCountPerDimension(dim.height, sectorWidth);
  sectorDims.depth = computeSectorCountPerDimension(dim.depth, sectorWidth);
  return sectorDims;
}

IndType computeTotalSectorCount(gpuNUFFT::Dimensions dim, IndType sectorWidth)
{
  return computeSectorCountPerDimension(dim, sectorWidth).count();
}

TEST(PrecomputationTest, ComputeIsotropicSectorCount)
{
  IndType imageWidth = 128;
  IndType sectorWidth = 8;

  IndType sectors = computeSectorCountPerDimension(imageWidth, sectorWidth);
  EXPECT_EQ(16u, sectors);

  imageWidth = 124;
  sectorWidth = 8;
  sectors = computeSectorCountPerDimension(imageWidth, sectorWidth);
  EXPECT_EQ(16u, sectors);

  imageWidth = 7;
  sectorWidth = 8;
  sectors = computeSectorCountPerDimension(imageWidth, sectorWidth);
  EXPECT_EQ(1u, sectors);

  imageWidth = 120;
  sectorWidth = 8;
  sectors = computeSectorCountPerDimension(imageWidth, sectorWidth);
  EXPECT_EQ(15u, sectors);
}

TEST(PrecomputationTest, ComputeIsotropicSectorDim)
{
  IndType imageWidth = 128;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  IndType sectors = computeSectorCountPerDimension(gridDim.width, sectorWidth);
  IndType sectorDim = computeTotalSectorCount(gridDim, sectorWidth);
  EXPECT_EQ(16 * osr, sectors);

  IndType expected = (IndType)std::pow(16 * osr, 3.0);
  EXPECT_EQ(expected, sectorDim);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  EXPECT_EQ(expected, sectorDims.count());
}

TEST(PrecomputationTest, testAddScalarToDimension)
{
  gpuNUFFT::Dimensions d;
  d.width = 10;
  d.height = 10;

  d = d + 3;

  EXPECT_EQ(13u, d.width);
  EXPECT_EQ(13u, d.height);
  EXPECT_EQ(0u, d.depth);
}

TEST(PrecomputationTest, ComputeSectorRanges)
{
  IndType imageWidth = 128;
  DType osr = 1.0;
  IndType sectorWidth = 8;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  DType expected[17] = { -0.5000, -0.4375, -0.3750, -0.3125, -0.2500, -0.1875,
                         -0.1250, -0.0625, 0,       0.0625,  0.1250,  0.1875,
                         0.2500,  0.3125,  0.3750,  0.4375,  0.5000 };

  DType *sectorRange = (DType *)malloc((sectorDims.width + 1) * sizeof(DType));
  // linspace in range from -0.5 to 0.5
  for (unsigned i = 0; i <= sectorDims.width; i++)
  {
    sectorRange[i] =
        (DType)-0.5 + (DType)i * (static_cast<DType>(1.0) / (sectorDims.width));
    if (DEBUG)
      printf("%5.4f ", sectorRange[i]);
    EXPECT_NEAR(sectorRange[i], expected[i], EPS);
  }
  if (DEBUG)
    std::cout << std::endl;

  free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors1D)
{
  IndType imageWidth = 16;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  const IndType coordCnt = 6;

  // Coords as StructureOfArrays
  // i.e. first x-vals, then y-vals and z-vals
  DType coords[coordCnt] = {(DType)-0.5, (DType)-0.3, (DType)-0.1,
                            (DType)0.1,  (DType)0.3,  (DType)0.5 };  // x

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = coordCnt;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  DType expected[4] = {(DType)-0.5000, (DType)-0.16666, (DType)0.16666,
                       (DType)0.5000 };

  DType *sectorRange = (DType *)malloc((sectorDims.width + 1) * sizeof(DType));
  // linspace in range from -0.5 to 0.5
  for (unsigned i = 0; i <= sectorDims.width; i++)
  {
    sectorRange[i] =
        (DType)-0.5 + (DType)i * (static_cast<DType>(1.0) / (sectorDims.width));
    if (DEBUG)
      printf("%5.4f ", sectorRange[i]);
    EXPECT_NEAR(sectorRange[i], expected[i], EPS);
  }
  if (DEBUG)
    std::cout << std::endl;

  IndType expectedSec[6] = { 0, 0, 1, 1, 2, 2 };

  for (unsigned cCnt = 0; cCnt < coordCnt; cCnt++)
  {
    DType x = kSpaceData.data[cCnt];
    if (DEBUG)
      std::cout << "processing x var: " << x << std::endl;
    IndType sector =
        (IndType)std::floor(static_cast<DType>(x + 0.5) * sectorDims.width);
    if (sector == sectorDims.width)
      sector--;
    if (DEBUG)
      std::cout << "into sector : " << sector << std::endl;
    EXPECT_EQ(expectedSec[cCnt], sector);
  }

  free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors1DOnBorders)
{
  IndType imageWidth = 16;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  const IndType coordCnt = 4;

  // Coords as StructureOfArrays
  // i.e. first x-vals, then y-vals and z-vals
  DType coords[coordCnt] = {(DType)-0.5, (DType)-0.1666, (DType)0.1666,
                            (DType)0.5 };  // x

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = coordCnt;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  DType expected[4] = {(DType)-0.5000, (DType)-0.16666, (DType)0.16666,
                       (DType)0.5000 };

  DType *sectorRange = (DType *)malloc((sectorDims.width + 1) * sizeof(DType));
  // linspace in range from -0.5 to 0.5
  for (unsigned i = 0; i <= sectorDims.width; i++)
  {
    sectorRange[i] =
        (DType)-0.5 + (DType)i * (static_cast<DType>(1.0) / (sectorDims.width));
    if (DEBUG)
      printf("%5.4f ", sectorRange[i]);
    EXPECT_NEAR(sectorRange[i], expected[i], EPS);
  }
  if (DEBUG)
    std::cout << std::endl;

  IndType expectedSec[4] = { 0, 1, 2, 2 };

  for (unsigned cCnt = 0; cCnt < coordCnt; cCnt++)
  {
    DType x = kSpaceData.data[cCnt];
    if (DEBUG)
      std::cout << "processing x var: " << x << std::endl;
    IndType sector =
        (IndType)std::round(static_cast<DType>(x + 0.5) * sectorDims.width);
    if (sector == sectorDims.width)
      sector--;
    if (DEBUG)
      std::cout << "into sector : " << sector << std::endl;
    EXPECT_EQ(expectedSec[cCnt], sector);
  }

  free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors2D)
{
  IndType imageWidth = 16;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  const IndType coordCnt = 6;

  // Coords as StructureOfArrays
  // i.e. first x-vals, then y-vals and z-vals
  DType coords[coordCnt * 2] = {(DType)-0.5, (DType)-0.3, (DType)-0.1,
                                (DType)0.1,  (DType)0.3,  (DType)0.5,  // x
                                (DType)-0.5, (DType)-0.5, 0,
                                0,           (DType)0.5,  (DType)0.45 };  // y

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = coordCnt;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  DType expected[4] = {(DType)-0.5000, (DType)-0.16666, (DType)0.16666,
                       (DType)0.5000 };

  DType *sectorRange = (DType *)malloc((sectorDims.width + 1) * sizeof(DType));
  // linspace in range from -0.5 to 0.5
  for (unsigned i = 0; i <= sectorDims.width; i++)
  {
    sectorRange[i] =
        (DType)-0.5 + (DType)i * (static_cast<DType>(1.0) / (sectorDims.width));
    EXPECT_NEAR(sectorRange[i], expected[i], EPS);
  }
  if (DEBUG)
    std::cout << std::endl;

  IndType expectedSec[6] = { 0, 0, 4, 4, 8, 8 };

  for (unsigned cCnt = 0; cCnt < coordCnt; cCnt++)
  {
    DType2 coord;
    coord.x = kSpaceData.data[cCnt];
    coord.y = kSpaceData.data[cCnt + kSpaceData.count()];

    if (DEBUG)
      std::cout << "processing x var: " << coord.x << " y: " << coord.y
                << std::endl;

    IndType x_sector = (IndType)std::floor(static_cast<float>(coord.x + 0.5) *
                                           sectorDims.width);
    if (x_sector == sectorDims.width)
      x_sector--;

    IndType y_sector = (IndType)std::floor(static_cast<float>(coord.y + 0.5) *
                                           sectorDims.height);
    if (y_sector == sectorDims.height)
      y_sector--;
    if (DEBUG)
      std::cout << "into sector x: " << x_sector << " y: " << y_sector
                << std::endl;
    EXPECT_EQ(expectedSec[cCnt], x_sector + y_sector * sectorDims.height);
  }

  free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors3D_16x16x16_15)
{
  IndType imageWidth = 16;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  const IndType coordCnt = 6;

  // Coords as StructureOfArrays
  // i.e. first x-vals, then y-vals and z-vals
  DType coords[coordCnt * 3] = {
    (DType)-0.5,  (DType)-0.3,     (DType)-0.1,
    (DType)0.1,   (DType)0.3,      (DType)0.5,  // x
    (DType)-0.5,  (DType)-0.5,     0,
    0,            (DType)0.5,      (DType)0.45,  // y
    (DType)-0.33, (DType)-0.16666, 0,
    0,            (DType)-0.23,    (DType)0.45
  };  // z

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = coordCnt;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  DType expected[4] = {(DType)-0.5000, (DType)-0.16666, (DType)0.16666,
                       (DType)0.5000 };

  DType *sectorRange = (DType *)malloc((sectorDims.width + 1) * sizeof(DType));
  // linspace in range from -0.5 to 0.5
  for (unsigned i = 0; i <= sectorDims.width; i++)
  {
    sectorRange[i] =
        (DType)-0.5 + (DType)i * (static_cast<DType>(1.0) / (sectorDims.width));
    EXPECT_NEAR(sectorRange[i], expected[i], EPS);
  }
  if (DEBUG)
    std::cout << std::endl;

  IndType expectedSec[6] = { 0u, 9u, 13u, 13u, 8u, 26u };

  for (unsigned cCnt = 0; cCnt < coordCnt; cCnt++)
  {
    DType3 coord;
    coord.x = kSpaceData.data[cCnt];
    coord.y = kSpaceData.data[cCnt + kSpaceData.count()];
    coord.z = kSpaceData.data[cCnt + 2 * kSpaceData.count()];

    if (DEBUG)
      std::cout << "processing x var: " << coord.x << " y: " << coord.y
                << " z: " << coord.z << std::endl;

    IndType x_sector = computeSectorMapping(coord.x, sectorDims.width);
    IndType y_sector = computeSectorMapping(coord.y, sectorDims.height);
    IndType z_sector = computeSectorMapping(coord.z, sectorDims.depth);

    if (DEBUG)
      std::cout << "into sector x: " << x_sector << " y: " << y_sector
                << " z: " << z_sector << std::endl;
    EXPECT_EQ(expectedSec[cCnt], (unsigned)computeXYZ2Lin(
                                     x_sector, y_sector, z_sector, sectorDims));

    IndType3 mappedSectors = computeSectorMapping(coord, sectorDims);
    EXPECT_EQ(expectedSec[cCnt],
              (unsigned)computeInd32Lin(mappedSectors, sectorDims));
  }

  free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors3D_20x20x10)
{

  gpuNUFFT::Dimensions imgDim;
  imgDim.width = 20;
  imgDim.height = 20;
  imgDim.depth = 10;
  DType osr = 1.0f;
  IndType sectorWidth = 5;

  gpuNUFFT::Dimensions gridDim = imgDim * osr;

  const IndType coordCnt = 6;

  // Coords as StructureOfArrays
  // i.e. first x-vals, then y-vals and z-vals
  DType coords[coordCnt * 3] = {
    (DType)-0.5,  (DType)-0.3,     (DType)-0.1,
    (DType)0.1,   (DType)0.3,      (DType)0.5,  // x
    (DType)-0.5,  (DType)-0.5,     0,
    0,            (DType)0.5,      (DType)0.45,  // y
    (DType)-0.33, (DType)-0.16666, (DType)-0.08,
    0,            (DType)0.08,     (DType)0.45
  };  // z

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = coordCnt;

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  DType expected[5] = { -0.5f, -0.25f, 0.0f, 0.25f, 0.5f };

  DType *sectorRange = (DType *)malloc((sectorDims.width + 1) * sizeof(DType));
  // linspace in range from -0.5 to 0.5
  for (unsigned i = 0; i <= sectorDims.width; i++)
  {
    sectorRange[i] =
        (DType)-0.5 + (DType)i * (static_cast<DType>(1.0) / (sectorDims.width));
    EXPECT_NEAR(sectorRange[i], expected[i], EPS);
  }

  IndType expectedSec[6] = { 0, 0, 9, 26, 31, 31 };

  for (unsigned cCnt = 0; cCnt < coordCnt; cCnt++)
  {
    DType3 coord;
    coord.x = kSpaceData.data[cCnt];
    coord.y = kSpaceData.data[cCnt + kSpaceData.count()];
    coord.z = kSpaceData.data[cCnt + 2 * kSpaceData.count()];

    if (DEBUG)
      std::cout << "processing x var: " << coord.x << " y: " << coord.y
                << " z: " << coord.z << std::endl;

    IndType x_sector = computeSectorMapping(coord.x, sectorDims.width);
    IndType y_sector = computeSectorMapping(coord.y, sectorDims.height);
    IndType z_sector =
        computeSectorMapping(coord.z, sectorDims.depth, sectorDims.width);

    if (DEBUG)
      std::cout << "into sector x: " << x_sector << " y: " << y_sector
                << " z: " << z_sector << std::endl;
    EXPECT_EQ(expectedSec[cCnt], (unsigned)computeXYZ2Lin(
                                     x_sector, y_sector, z_sector, sectorDims));

    IndType3 mappedSectors = computeSectorMapping(coord, sectorDims);
    EXPECT_EQ(expectedSec[cCnt],
              (unsigned)computeInd32Lin(mappedSectors, sectorDims));
  }

  free(sectorRange);
}

TEST(PrecomputationTest, AssignSectors3D_20x20x10_15)
{

  gpuNUFFT::Dimensions imgDim;
  imgDim.width = 20;
  imgDim.height = 20;
  imgDim.depth = 10;
  DType osr = 1.5;
  IndType sectorWidth = 5;

  gpuNUFFT::Dimensions gridDim = imgDim * osr;

  const IndType coordCnt = 6;

  // Coords as StructureOfArrays
  // i.e. first x-vals, then y-vals and z-vals
  DType coords[coordCnt * 3] = {
    (DType)-0.5,  (DType)-0.3,     (DType)-0.1,
    (DType)0.1,   (DType)0.3,      (DType)0.5,  // x
    (DType)-0.5,  (DType)-0.5,     0,
    0,            (DType)0.5,      (DType)0.45,  // y
    (DType)-0.33, (DType)-0.16666, (DType)-0.08,
    0,            (DType)0.08,     (DType)0.45
  };  // z

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = coordCnt;

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  DType expected[7] = {(DType)-0.5000, (DType)-0.3333, (DType)-0.1666, 0,
                       (DType)0.1666,  (DType)0.3333,  (DType)0.5000 };

  DType *sectorRange = (DType *)malloc((sectorDims.width + 1) * sizeof(DType));
  // linspace in range from -0.5 to 0.5
  for (unsigned i = 0; i <= sectorDims.width; i++)
  {
    sectorRange[i] =
        (DType)-0.5 + (DType)i * (static_cast<DType>(1.0) / (sectorDims.width));
    EXPECT_NEAR(sectorRange[i], expected[i], EPS);
  }

  IndType expectedSec[6] = { 0, 37, 56, 57, 70, 107 };

  for (unsigned cCnt = 0; cCnt < coordCnt; cCnt++)
  {
    DType3 coord;
    coord.x = kSpaceData.data[cCnt];
    coord.y = kSpaceData.data[cCnt + kSpaceData.count()];
    coord.z = kSpaceData.data[cCnt + 2 * kSpaceData.count()];

    if (DEBUG)
      std::cout << "processing x var: " << coord.x << " y: " << coord.y
                << " z: " << coord.z << std::endl;

    IndType x_sector = computeSectorMapping(coord.x, sectorDims.width);
    IndType y_sector = computeSectorMapping(coord.y, sectorDims.height);
    IndType z_sector = computeSectorMapping(coord.z, sectorDims.depth);

    if (DEBUG)
      std::cout << "into sector x: " << x_sector << " y: " << y_sector
                << " z: " << z_sector << std::endl;
    EXPECT_EQ(expectedSec[cCnt], (unsigned)computeXYZ2Lin(
                                     x_sector, y_sector, z_sector, sectorDims));

    IndType3 mappedSectors = computeSectorMapping(coord, sectorDims);
    EXPECT_EQ(expectedSec[cCnt],
              (unsigned)computeInd32Lin(mappedSectors, sectorDims));
  }

  free(sectorRange);
}

bool pairComp(std::pair<IndType, IndType> i, std::pair<IndType, IndType> j)
{
  return (i.second < j.second);
}

std::vector<gpuNUFFT::IndPair>
sortVector(gpuNUFFT::Array<IndType> assignedSectors)
{
  std::vector<gpuNUFFT::IndPair> secVector;

  for (IndType i = 0; i < assignedSectors.count(); i++)
    secVector.push_back(gpuNUFFT::IndPair(i, assignedSectors.data[i]));

  // using function as comp
  std::sort(secVector.begin(), secVector.end());

  return secVector;
}

TEST(PrecomputationTest, TestIndexSorting)
{
  IndType assSectors[6] = { 0, 9, 13, 13, 8, 26 };
  IndType expectedSectors[6] = { 0, 8, 9, 13, 13, 26 };

  gpuNUFFT::Array<IndType> assignedSectors;
  assignedSectors.data = assSectors;
  assignedSectors.dim.length = 6;

  std::vector<gpuNUFFT::IndPair> secVector = sortVector(assignedSectors);

  // print out content:
  if (DEBUG)
  {
    std::cout << "vector contains:";
    for (std::vector<gpuNUFFT::IndPair>::iterator it = secVector.begin();
         it != secVector.end(); ++it)
      std::cout << " " << it->second << " (" << it->first << ") ";
    std::cout << '\n';
  }
  gpuNUFFT::IndPair *sortedArray = &secVector[0];

  // print indices for reselect
  for (IndType i = 0; i < 6; i++)
  {
    if (DEBUG)
      std::cout << sortedArray[i].first;
    EXPECT_EQ(sortedArray[i].second, expectedSectors[i]);
  }
  if (DEBUG)
    std::cout << std::endl;
}

TEST(PrecomputationTest, AssignSectors3DSorted)
{
  IndType imageWidth = 16;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  const IndType coordCnt = 6;

  // Coords as StructureOfArrays
  // i.e. first x-vals, then y-vals and z-vals
  DType coords[coordCnt * 3] = {
    (DType)-0.5,  (DType)-0.3,     (DType)-0.1,
    (DType)0.1,   (DType)0.3,      (DType)0.5,  // x
    (DType)-0.5,  (DType)-0.5,     0,
    0,            (DType)0.5,      (DType)0.45,  // y
    (DType)-0.33, (DType)-0.16666, 0,
    0,            (DType)-0.23,    (DType)0.45
  };  // z

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = coordCnt;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  IndType expectedSec[6] = { 0, 9, 13, 13, 8, 26 };

  gpuNUFFT::Array<IndType> assignedSectors;
  assignedSectors.data = (IndType *)malloc(coordCnt * sizeof(IndType));
  assignedSectors.dim.length = coordCnt;

  for (unsigned cCnt = 0; cCnt < coordCnt; cCnt++)
  {
    DType3 coord;
    coord.x = kSpaceData.data[cCnt];
    coord.y = kSpaceData.data[cCnt + kSpaceData.count()];
    coord.z = kSpaceData.data[cCnt + 2 * kSpaceData.count()];

    IndType3 mappedSectors = computeSectorMapping(coord, sectorDims);

    IndType sector = computeInd32Lin(mappedSectors, sectorDims);
    assignedSectors.data[cCnt] = sector;
    EXPECT_EQ(expectedSec[cCnt], sector);
  }

  IndType expectedSecSorted[6] = { 0, 8, 9, 13, 13, 26 };
  IndType expectedSecIndexSorted[6] = { 0, 4, 1, 2, 3, 5 };

  std::vector<gpuNUFFT::IndPair> secVector = sortVector(assignedSectors);

  DType coords_sorted[coordCnt * 3];

  for (unsigned i = 0; i < assignedSectors.count(); i++)
  {
    // compare index
    EXPECT_EQ(expectedSecIndexSorted[i], secVector[i].first);
    EXPECT_EQ(expectedSecSorted[i], secVector[i].second);
    coords_sorted[i] = kSpaceData.data[secVector[i].first];
    coords_sorted[i + 1 * coordCnt] =
        kSpaceData.data[secVector[i].first + 1 * coordCnt];
    coords_sorted[i + 2 * coordCnt] =
        kSpaceData.data[secVector[i].first + 2 * coordCnt];
  }

  if (DEBUG)
    for (unsigned i = 0; i < kSpaceData.count(); i++)
    {
      std::cout << " x: " << coords_sorted[i]
                << " y: " << coords_sorted[i + 1 * coordCnt]
                << " z:" << coords_sorted[i + 2 * coordCnt] << std::endl;
    }

  free(assignedSectors.data);
}

TEST(PrecomputationTest, ComputeDataIndices)
{
  IndType imageWidth = 16;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  const IndType coordCnt = 6;

  // Coords as StructureOfArrays
  // i.e. first x-vals, then y-vals and z-vals
  DType coords[coordCnt * 3] = {
    (DType)-0.5,  (DType)-0.3,     (DType)-0.1,
    (DType)0.1,   (DType)0.3,      (DType)0.5,  // x
    (DType)-0.5,  (DType)-0.5,     0,
    0,            (DType)0.5,      (DType)0.45,  // y
    (DType)-0.33, (DType)-0.16666, 0,
    0,            (DType)-0.23,    (DType)0.45
  };  // z

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = coordCnt;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  IndType expectedSec[6] = { 0, 9, 13, 13, 8, 26 };

  gpuNUFFT::Array<IndType> assignedSectors;
  assignedSectors.data = (IndType *)malloc(coordCnt * sizeof(IndType));
  assignedSectors.dim.length = coordCnt;

  for (unsigned cCnt = 0; cCnt < coordCnt; cCnt++)
  {
    DType3 coord;
    coord.x = kSpaceData.data[cCnt];
    coord.y = kSpaceData.data[cCnt + kSpaceData.count()];
    coord.z = kSpaceData.data[cCnt + 2 * kSpaceData.count()];

    IndType3 mappedSectors = computeSectorMapping(coord, sectorDims);

    IndType sector = computeInd32Lin(mappedSectors, sectorDims);
    assignedSectors.data[cCnt] = sector;
    EXPECT_EQ(expectedSec[cCnt], sector);
  }

  IndType expectedSecSorted[6] = { 0, 8, 9, 13, 13, 26 };
  IndType expectedSecIndexSorted[6] = { 0, 4, 1, 2, 3, 5 };

  std::vector<gpuNUFFT::IndPair> secVector = sortVector(assignedSectors);

  for (unsigned i = 0; i < coordCnt; i++)
  {
    // compare index
    EXPECT_EQ(expectedSecIndexSorted[i], secVector[i].first);
    EXPECT_EQ(expectedSecSorted[i], secVector[i].second);
  }

  IndType cnt = 0;
  std::vector<IndType> dataIndices;

  IndType sectorDataCount[29] = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 5,
                                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6 };

  dataIndices.push_back(0);
  for (unsigned i = 0; i <= sectorDims.count(); i++)
  {
    while (cnt < coordCnt && i == secVector[cnt].second)
      cnt++;

    dataIndices.push_back(cnt);
    EXPECT_EQ(sectorDataCount[i + 1], cnt);
  }

  if (DEBUG)
  {
    for (unsigned i = 0; i < dataIndices.size(); i++)
    {
      std::cout << dataIndices.at(i) << " ";
    }
    std::cout << std::endl;
  }
  free(assignedSectors.data);
}

TEST(PrecomputationTest, ComputeSectorCenters)
{
  IndType imageWidth = 16;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth * osr);
  gridDim.height = (IndType)(imageWidth * osr);
  gridDim.depth = (IndType)(imageWidth * osr);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  EXPECT_EQ(27u, sectorDims.count());

  gpuNUFFT::Array<IndType3> sectorCenters;
  sectorCenters.data =
      (IndType3 *)malloc(sectorDims.count() * sizeof(IndType3));
  sectorCenters.dim.length = sectorDims.count();

  for (IndType z = 0; z < sectorDims.depth; z++)
    for (IndType y = 0; y < sectorDims.height; y++)
      for (IndType x = 0; x < sectorDims.width; x++)
      {
        IndType3 center;
        center.x =
            x * sectorWidth +
            (IndType)std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
        center.y =
            y * sectorWidth +
            (IndType)std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
        center.z =
            z * sectorWidth +
            (IndType)std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
        sectorCenters.data[computeXYZ2Lin(x, y, z, sectorDims)] = center;
      }

  if (DEBUG)
    for (unsigned i = 0; i < sectorDims.count(); i++)
    {
      std::cout << " x: " << sectorCenters.data[i].x
                << " y: " << sectorCenters.data[i].y
                << " z: " << sectorCenters.data[i].z << std::endl;
    }

  EXPECT_EQ(4u, sectorCenters.data[0].x);
  EXPECT_EQ(4u, sectorCenters.data[0].y);
  EXPECT_EQ(4u, sectorCenters.data[0].z);

  free(sectorCenters.data);
}

TEST(PrecomputationTest, AnisotropicComputeSectorDim)
{
  IndType imageWidth = 128;
  DType osr = 1.5;
  IndType sectorWidth = 8;

  gpuNUFFT::Dimensions gridDim;
  gridDim.width = (IndType)(imageWidth);
  gridDim.height = (IndType)(imageWidth);
  gridDim.depth = (IndType)((imageWidth - 16));

  if (DEBUG)
    std::cout << " dimensions before: w: " << gridDim.width
              << " h: " << gridDim.height << " d: " << gridDim.depth
              << std::endl;
  gridDim = gridDim * osr;

  if (DEBUG)
    std::cout << " dimensions scaled: w: " << gridDim.width
              << " h: " << gridDim.height << " d: " << gridDim.depth
              << std::endl;

  IndType sectorDim = computeTotalSectorCount(gridDim, sectorWidth);

  IndType expected = (IndType)(16 * 16 * 14 * osr * osr * osr);
  EXPECT_EQ(expected, sectorDim);

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  EXPECT_EQ(expected, sectorDims.count());
}

TEST(PrecomputationTest, AnisotropicComputeSectorCenters)
{
  gpuNUFFT::Dimensions imgDim;
  imgDim.width = 16;
  imgDim.height = 16;
  imgDim.depth = 8;

  DType osr = 1.5;
  IndType sectorWidth = 8;

  gpuNUFFT::Dimensions gridDim = imgDim * osr;

  gpuNUFFT::Dimensions sectorDims =
      computeSectorCountPerDimension(gridDim, sectorWidth);

  EXPECT_EQ(18u, sectorDims.count());
  EXPECT_EQ(2u, sectorDims.depth);

  gpuNUFFT::Array<IndType3> sectorCenters;
  sectorCenters.data =
      (IndType3 *)malloc(sectorDims.count() * sizeof(IndType3));
  sectorCenters.dim.length = sectorDims.count();

  for (IndType z = 0; z < sectorDims.depth; z++)
    for (IndType y = 0; y < sectorDims.height; y++)
      for (IndType x = 0; x < sectorDims.width; x++)
      {
        IndType3 center;
        center.x =
            x * sectorWidth +
            (IndType)std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
        center.y =
            y * sectorWidth +
            (IndType)std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
        center.z =
            z * sectorWidth +
            (IndType)std::floor(static_cast<DType>(sectorWidth) / (DType)2.0);
        sectorCenters.data[computeXYZ2Lin(x, y, z, sectorDims)] = center;
      }

  if (DEBUG)
    for (unsigned i = 0; i < sectorDims.count(); i++)
    {
      std::cout << " x: " << sectorCenters.data[i].x
                << " y: " << sectorCenters.data[i].y
                << " z: " << sectorCenters.data[i].z << std::endl;
    }

  EXPECT_EQ(4u, sectorCenters.data[0].x);
  EXPECT_EQ(4u, sectorCenters.data[0].y);
  EXPECT_EQ(4u, sectorCenters.data[0].z);

  EXPECT_EQ(4u, sectorCenters.data[9].x);
  EXPECT_EQ(4u, sectorCenters.data[9].y);
  EXPECT_EQ(12u, sectorCenters.data[9].z);

  EXPECT_EQ(20u, sectorCenters.data[17].x);
  EXPECT_EQ(20u, sectorCenters.data[17].y);
  EXPECT_EQ(12u, sectorCenters.data[17].z);

  free(sectorCenters.data);
}

TEST(IndexComputationTest, TestGetCoordsFromIndex_2x2x2)
{
  const int N = 8;

  IndType3 gridDims;
  gridDims.x = 2;
  gridDims.y = 2;
  gridDims.z = 2;

  int x, y, z;
  int expected[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
                     0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1 };

  for (int i = 0; i < N; i++)
  {
    getCoordsFromIndex(i, &x, &y, &z, gridDims.x, gridDims.y, gridDims.z);
    EXPECT_EQ(expected[3 * i], x);
    EXPECT_EQ(expected[3 * i + 1], y);
    EXPECT_EQ(expected[3 * i + 2], z);
  }
}

TEST(IndexComputationTest, TestGetCoordsFromIndex_2x2x1)
{
  int N = 4;

  IndType3 gridDims;
  gridDims.x = 2;
  gridDims.y = 2;
  gridDims.z = 1;

  int x, y, z;
  int expected[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0 };
  for (int i = 0; i < N; i++)
  {
    getCoordsFromIndex(i, &x, &y, &z, gridDims.x, gridDims.y, gridDims.z);
    EXPECT_EQ(expected[3 * i], x);
    EXPECT_EQ(expected[3 * i + 1], y);
    EXPECT_EQ(expected[3 * i + 2], z);
  }
}

TEST(IndexComputationTest, TestGetCoordsFromIndex_4x4x2)
{
  const int N = 32;

  IndType3 gridDims;
  gridDims.x = 4;
  gridDims.y = 4;
  gridDims.z = 2;

  int x, y, z;
  int expected[N * 3];
  int cnt = 0;
  for (unsigned z = 0; z < gridDims.z; z++)
    for (unsigned y = 0; y < gridDims.y; y++)
      for (unsigned x = 0; x < gridDims.x; x++)
      {
        expected[cnt * 3] = x;
        expected[cnt * 3 + 1] = y;
        expected[cnt * 3 + 2] = z;
        cnt++;
      }

  for (int i = 0; i < N; i++)
  {
    getCoordsFromIndex(i, &x, &y, &z, gridDims.x, gridDims.y, gridDims.z);
    EXPECT_EQ(expected[3 * i], x);
    EXPECT_EQ(expected[3 * i + 1], y);
    EXPECT_EQ(expected[3 * i + 2], z);
  }
}

int computePossibleConcurrentCoilCount(int n_coils,
                                       gpuNUFFT::Dimensions imgDims,
                                       int free_mem)
{
  int possibleCoilCount = n_coils;

  while ((free_mem / (possibleCoilCount * imgDims.count() * 8.0 * 2.0)) < 1.0 &&
         possibleCoilCount-- > 1)
    ;

  return possibleCoilCount;
}

TEST(TestPossibleCoilCount, EnoughMemoryFor12Coils)
{
  unsigned n_coils = 12;
  unsigned free_mem = 256 * 256 * 13 * 8 * 2;
  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 256;
  imgDims.height = 256;

  EXPECT_EQ(computePossibleConcurrentCoilCount(n_coils, imgDims, free_mem),
            n_coils);
}

TEST(TestPossibleCoilCount, EnoughMemoryFor11Coils)
{
  unsigned n_coils = 12;
  unsigned free_mem = 256 * 256 * 11 * 8 * 2;
  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 256;
  imgDims.height = 256;

  EXPECT_EQ(computePossibleConcurrentCoilCount(n_coils, imgDims, free_mem),
            n_coils - 1);
}

TEST(TestPossibleCoilCount, EnoughMemoryFor1Coil)
{
  unsigned n_coils = 12;
  unsigned free_mem = 256 * 256 * 1 * 8 * 2;
  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 256;
  imgDims.height = 256;

  EXPECT_EQ(computePossibleConcurrentCoilCount(n_coils, imgDims, free_mem), 1);
}

TEST(TestPossibleCoilCount, NotEnoughMemory)
{
  unsigned n_coils = 12;
  unsigned free_mem = 256 * 255 * 8 * 2;
  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 256;
  imgDims.height = 256;

  EXPECT_EQ(computePossibleConcurrentCoilCount(n_coils, imgDims, free_mem), 0);
}

