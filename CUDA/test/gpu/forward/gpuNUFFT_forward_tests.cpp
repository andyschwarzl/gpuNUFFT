#include <limits.h>

#include "gtest/gtest.h"

#include "gpuNUFFT_operator_factory.hpp"

#define epsilon 0.02f

#define get3DC2lin(_x, _y, _z, _width)                                         \
  ((_x) + (_width) * ((_y) + (_z) * (_width)))

TEST(TestGPUGpuNUFFTForwardConv, KernelCall1Sector)
{
  int kernel_width = 5;

  // oversampling ratio
  float osr = 1.25f;

  // Image
  int im_width = 32;

  // Data
  int data_entries = 1;

  // Coords
  // Scaled between -0.5 and 0.5
  // in triplets (x,y,z)
  DType *coords = (DType *)calloc(3 * data_entries, sizeof(DType));  // 3* x,y,z
  coords[0] = -0.31719f;  // should result in 7,7,7 center
  coords[1] = -0.38650f;
  coords[2] = 0;

  DType2 *im_data;
  unsigned long dims_g[4];
  dims_g[0] = 2; /* complex */
  dims_g[1] = (unsigned long)(im_width);
  dims_g[2] = (unsigned long)(im_width);
  dims_g[3] = (unsigned long)(im_width);

  long im_size = dims_g[1] * dims_g[2] * dims_g[3];

  im_data = (DType2 *)calloc(im_size, sizeof(DType2));

  for (int x = 0; x < im_size; x++)
  {
    im_data[x].x = 1.0f;
    im_data[x].y = 1.0f;
  }

  // sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> dataArray;

  gpuNUFFT::Array<DType2> im_dataArray;
  im_dataArray.data = im_data;
  im_dataArray.dim.width = im_width;
  im_dataArray.dim.height = im_width;
  im_dataArray.dim.depth = im_width;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  // gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = new
  // gpuNUFFT::GpuNUFFTOperator(kernel_width,sector_width,osr);
  gpuNUFFT::GpuNUFFTOperatorFactory factory(false, true, true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(
      kSpaceData, kernel_width, sector_width, osr, imgDims);

  dataArray = gpuNUFFTOp->performForwardGpuNUFFT(im_dataArray);

  if (DEBUG)
    for (unsigned j = 0; j < dataArray.count(); j++)
    {
      printf("%.4f %.4f \n", dataArray.data[j].x, dataArray.data[j].y);
    }
  EXPECT_NEAR(0.0001973f, dataArray.data[0].x, epsilon);

  if (DEBUG)
    printf("\n");

  free(coords);
  free(im_data);
  free(dataArray.data);

  delete gpuNUFFTOp;
}

TEST(TestGPUGpuNUFFTForwardConv, KernelCall1Sector2Channels)
{
  int kernel_width = 5;
  float osr = 1.25f;
  int im_width = 32;

  // Data
  int data_entries = 1;

  // Coords
  // Scaled between -0.5 and 0.5
  // in triplets (x,y,z)
  DType *coords = (DType *)calloc(3 * data_entries, sizeof(DType));  // 3* x,y,z
  coords[0] = -0.31719f;  // should result in 7,7,7 center
  coords[1] = -0.38650f;
  coords[2] = 0;

  gpuNUFFT::Array<DType2> im_dataArray;
  im_dataArray.data = NULL;
  im_dataArray.dim.width = im_width;
  im_dataArray.dim.height = im_width;
  im_dataArray.dim.depth = im_width;
  im_dataArray.dim.channels = 2;

  im_dataArray.data = (DType2 *)calloc(im_dataArray.count(), sizeof(DType2));

  for (unsigned x = 0; x < im_dataArray.count(); x++)
  {
    im_dataArray.data[x].x = 1.0f;
    im_dataArray.data[x].y = 1.0f;
  }

  // sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> dataArray;

  // imgDims -> no Channel size here
  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_dataArray.dim.width;
  imgDims.height = im_dataArray.dim.height;
  imgDims.depth = im_dataArray.dim.depth;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false, true, true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(
      kSpaceData, kernel_width, sector_width, osr, imgDims);

  dataArray = gpuNUFFTOp->performForwardGpuNUFFT(im_dataArray);

  if (DEBUG)
  {
    std::cout << "result count : " << dataArray.count() << std::endl;

    for (unsigned j = 0; j < dataArray.count(); j++)
    {
      printf("%.4f %.4f \n", dataArray.data[j].x, dataArray.data[j].y);
    }
  }
  EXPECT_NEAR(0.0001973f, dataArray.data[0].x, epsilon);
  EXPECT_EQ(dataArray.data[1].x, dataArray.data[0].x);
  EXPECT_EQ(dataArray.data[1].y, dataArray.data[0].y);

  if (DEBUG)
    printf("\n");

  free(coords);
  free(im_dataArray.data);
  delete gpuNUFFTOp;
}

TEST(TestGPUGpuNUFFTForwardConv, 2D_32_32_4)
{
  int kernel_width = 5;
  float osr = 1.25f;
  int im_width = 32;

  // Data
  int data_count = 3;
  int n_coils = 4;

  // Coords
  // Scaled between -0.5 and 0.5
  // in triplets (x,y,z)
  DType *coords = (DType *)calloc(2 * data_count, sizeof(DType));  // 2* x,y,z
  coords[0] = -0.31719f;  // should result in 7,7,7 center
  coords[1] = -0.38650f;

  coords[2] = 0.0;  // should result in 7,7,7 center
  coords[3] = 0.0;

  coords[4] = 0.31719f;  // should result in 7,7,7 center
  coords[5] = 0.38650f;

  gpuNUFFT::Array<DType2> im_dataArray;
  im_dataArray.data = NULL;
  im_dataArray.dim.width = im_width;
  im_dataArray.dim.height = im_width;
  im_dataArray.dim.channels = n_coils;

  im_dataArray.data = (DType2 *)calloc(im_dataArray.count(), sizeof(DType2));

  for (unsigned x = 0; x < im_dataArray.count(); x++)
  {
    im_dataArray.data[x].x = 1.0f;
    im_dataArray.data[x].y = 1.0f;
  }

  // sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_count;

  gpuNUFFT::Array<CufftType> dataArray;

  // imgDims -> no Channel size here
  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_dataArray.dim.width;
  imgDims.height = im_dataArray.dim.height;

  for (int cnt = 0; cnt < 8; cnt++)
  {
    bool useTextures = cnt & 1;
    bool useGpu = cnt & 2;
    bool loadBalancing = cnt & 4;

    std::cout << "Use Textures: " << useTextures << std::endl
              << "Use GPU: " << useGpu << std::endl
              << "Use LoadBalancing:" << loadBalancing << std::endl;

    gpuNUFFT::GpuNUFFTOperatorFactory factory(useTextures, useGpu,
                                              loadBalancing);
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(
        kSpaceData, kernel_width, sector_width, osr, imgDims);

    dataArray = gpuNUFFTOp->performForwardGpuNUFFT(im_dataArray);

    if (DEBUG)
    {
      std::cout << "result count : " << dataArray.count() << std::endl;

      for (unsigned j = 0; j < dataArray.count(); j++)
      {
        printf("%.9f %.9f \n", dataArray.data[j].x, dataArray.data[j].y);
      }
    }

    for (int chn = 0; chn < n_coils; chn++)
    {
      if (useTextures)
        EXPECT_NEAR(0.816, dataArray.data[0 + chn * data_count].x, epsilon);
      else
        EXPECT_NEAR(0.73328, dataArray.data[0 + chn * data_count].x, epsilon);
      EXPECT_NEAR(dataArray.data[0].x, dataArray.data[0 + chn * data_count].x,
                  1E-5);
      EXPECT_NEAR(dataArray.data[0].y, dataArray.data[0 + chn * data_count].y,
                  1E-5);
    }

    if (DEBUG)
      printf("\n");
    delete gpuNUFFTOp;
  }

  free(coords);
  free(im_dataArray.data);
}

TEST(TestGPUGpuNUFFTForwardConv, 2D_32_32_8_sens)
{
  int kernel_width = 5;
  float osr = 1.25f;
  int im_width = 32;

  // Data
  int data_count = 3;
  int n_coils = 8;

  // Coords
  // Scaled between -0.5 and 0.5
  // in duplets (x,y)
  DType *coords = (DType *)calloc(2 * data_count, sizeof(DType));  // 2* x,y,z
  coords[0] = -0.31719f;
  coords[1] = -0.38650f;

  coords[2] = 0;
  coords[3] = 0;

  coords[4] = 0.25;
  coords[5] = 0.25;

  gpuNUFFT::Array<DType2> im_dataArray;
  im_dataArray.data = NULL;
  im_dataArray.dim.width = im_width;
  im_dataArray.dim.height = im_width;
  im_dataArray.dim.channels = 1;

  im_dataArray.data = (DType2 *)calloc(im_dataArray.count(), sizeof(DType2));

  for (unsigned x = 0; x < im_dataArray.count(); x++)
  {
    im_dataArray.data[x].x = 1.0f;
    im_dataArray.data[x].y = 1.0f;
  }

  gpuNUFFT::Array<DType2> sensArray;
  sensArray.data = NULL;
  sensArray.dim.width = im_width;
  sensArray.dim.height = im_width;
  sensArray.dim.channels = n_coils;

  sensArray.data = (DType2 *)calloc(sensArray.count(), sizeof(DType2));

  for (int coil = 0; coil < n_coils; coil++)
  {
    for (unsigned x = 0; x < im_dataArray.count(); x++)
    {
      sensArray.data[coil * im_dataArray.count() + x].x = (DType)coil + 1;
      sensArray.data[coil * im_dataArray.count() + x].y = (DType)coil + 1;
    }
  }

  gpuNUFFT::Array<DType> densArray;
  densArray.data = (DType *)calloc(data_count, sizeof(DType));
  densArray.dim.length = data_count;
  for (int cnt = 0; cnt < data_count; cnt++)
    densArray.data[cnt] = 1.0;

  // sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_count;

  gpuNUFFT::Array<CufftType> dataArray;

  // imgDims -> no Channel size here
  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_dataArray.dim.width;
  imgDims.height = im_dataArray.dim.height;

  for (int cnt = 0; cnt < 8; cnt++)
  {
    bool useTextures = cnt & 1;
    bool useGpu = cnt & 2;
    bool loadBalancing = cnt & 4;

    std::cout << "Use Textures: " << useTextures << std::endl
              << "Use GPU: " << useGpu << std::endl
              << "Use LoadBalancing:" << loadBalancing << std::endl;

    gpuNUFFT::GpuNUFFTOperatorFactory factory(useTextures, useGpu,
                                              loadBalancing);
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(
        kSpaceData, densArray, sensArray, kernel_width, sector_width, osr,
        imgDims);

    dataArray = gpuNUFFTOp->performForwardGpuNUFFT(im_dataArray);

    if (DEBUG)
    {
      std::cout << "result count : " << dataArray.count() << std::endl;

      for (unsigned j = 0; j < dataArray.count(); j++)
      {
        printf("x: %.6f %.6f \n", dataArray.data[j].x, dataArray.data[j].y);
      }
    }

    EXPECT_EQ(dataArray.dim.channels, n_coils);

    for (int chn = 1; chn < n_coils; chn++)
    {
      EXPECT_NEAR((chn + 1) * dataArray.data[0].x,
                  dataArray.data[0 + chn * data_count].x, 1E-3);
      EXPECT_NEAR((chn + 1) * dataArray.data[0].y,
                  dataArray.data[0 + chn * data_count].y, 1E-3);
    }

    if (DEBUG)
      printf("\n");
    delete gpuNUFFTOp;
  }

  free(coords);
  free(densArray.data);
  free(sensArray.data);
  free(im_dataArray.data);
}

