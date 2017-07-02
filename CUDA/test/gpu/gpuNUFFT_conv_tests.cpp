#include <limits.h>

#include "gtest/gtest.h"

#include "gpuNUFFT_operator_factory.hpp"


#define epsilon 0.02f

#define get3DC2lin(_x,_y,_z,_width) ((_x) + (_width) * ( (_y) + (_z) * (_width)))


TEST(TestKernel, LoadKernel) {
  if (DEBUG) printf("start creating kernel...\n");
  long kernel_entries = calculateGrid3KernelSize();

  assert(kernel_entries > 0);

  DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
  if (kern != NULL)
  {
    loadGrid3Kernel(kern,kernel_entries);
    EXPECT_EQ(1.0f,kern[0]);
    EXPECT_LT(0.9940f-kern[1],epsilon);
    EXPECT_LT(0.0621f-kern[401],epsilon);
    EXPECT_LT(0.0041f-kern[665],epsilon);
    EXPECT_EQ(0.0f,kern[kernel_entries-1]);
    free(kern);
  }
  EXPECT_EQ(1, 1);
}


TEST(TestKernel, LoadKernelFromGpuNUFFTFactory) {
  if (DEBUG) printf("start creating kernel...\n");
  IndType kernelWidth = 3;
  IndType sectorWidth = 8;
  DType osf = 1;
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = new gpuNUFFT::GpuNUFFTOperator(kernelWidth,sectorWidth,osf,gpuNUFFT::Dimensions(1,1,1));

  assert(gpuNUFFTOp->getKernel().count() > 0);

  if (gpuNUFFTOp->getKernel().data != NULL)
  {
    DType* kern = gpuNUFFTOp->getKernel().data;
    EXPECT_EQ(1.0f,kern[0]);
    EXPECT_LT(0.9940f-kern[1],epsilon);
    EXPECT_LT(0.0621f-kern[401],epsilon);
    EXPECT_LT(0.0041f-kern[665],epsilon);
    EXPECT_EQ(0.0f,kern[gpuNUFFTOp->getKernel().count()-1]);
  }

  delete gpuNUFFTOp;
}

TEST(TestKernel, Load2DKernel) {
  if (DEBUG) printf("start creating kernel...\n");
  long kernel_entries = 8;

  assert(kernel_entries > 0);

  DType *kern = (DType*) calloc(kernel_entries*kernel_entries,sizeof(DType));
  EXPECT_TRUE(kern != NULL);
  load2DKernel(kern,kernel_entries,3,1.0);
  if (DEBUG)
  {
    for (int i=0; i<kernel_entries;i++)
    {
      for (int j=0;j<kernel_entries;j++)
        printf("%.3f ",kern[j+i*kernel_entries]);
      printf("\n");
    }
  }
  EXPECT_EQ(1.0f,kern[0]);
  EXPECT_EQ(0.0f,kern[kernel_entries-1]);
  free(kern);
  EXPECT_EQ(1, 1);
}

TEST(TestKernel, Load3DKernel) {
  if (DEBUG) printf("start creating kernel...\n");
  long kernel_entries = 8;

  assert(kernel_entries > 0);

  DType *kern = (DType*) calloc(kernel_entries*kernel_entries*kernel_entries,sizeof(DType));
  EXPECT_TRUE(kern != NULL);
  load3DKernel(kern,kernel_entries,3,1.0);
  if (DEBUG)
  {
    for (int k=0; k<kernel_entries;k++)
    {
      for (int i=0; i<kernel_entries;i++)
      {
        for (int j=0;j<kernel_entries;j++)
          printf("%.3f ",kern[j+kernel_entries*(i+k*kernel_entries)]);
        printf("\n");
      }
      printf("---------------------------\n");
    }
  }
  EXPECT_EQ(1.0f,kern[0]);
  EXPECT_EQ(0.0f,kern[kernel_entries-1]);
  free(kern);
  EXPECT_EQ(1, 1);
}

TEST(TestGPUGpuNUFFTConv,KernelCall1Sector)
{
  int kernel_width = 3;

  //Image
  int im_width = 10;

  //Data
  int data_entries = 1;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  data[0].x = 1;
  data[0].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  coords[0] = 0; //should result in 7,7,7 center
  coords[1] = 0;
  coords[2] = 0;

  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;

  //Output Grid

  //sectors of data, count and start indices
  int sector_width = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,false,true); 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  if (DEBUG)
    for (int j=0; j<im_width; j++)
    {
      for (int i=0; i<im_width; i++)
        printf("%.4f ",gdata[get3DC2lin(i,j,5,im_width)].x);
      printf("\n");
    }

    if (DEBUG) printf("test %f \n",gdata[4].x);
    int index = get3DC2lin(5,5,5,im_width);
    if (DEBUG) printf("index to test %d\n",index);
    EXPECT_EQ(index,555);
    EXPECT_NEAR(1.0f,gdata[index].x,epsilon);
    EXPECT_NEAR(0.4502,gdata[get3DC2lin(5,4,5,im_width)].x,epsilon*10.0f);
    EXPECT_NEAR(0.4502,gdata[get3DC2lin(4,5,5,im_width)].x,epsilon*10.0f);
    EXPECT_NEAR(0.4502,gdata[get3DC2lin(5,6,5,im_width)].x,epsilon*10.0f);

    EXPECT_NEAR(0.2027,gdata[get3DC2lin(6,6,5,im_width)].x,epsilon*10.0f);
    EXPECT_NEAR(0.2027,gdata[get3DC2lin(4,4,5,im_width)].x,epsilon*10.0f);
    EXPECT_NEAR(0.2027,gdata[get3DC2lin(4,6,5,im_width)].x,epsilon*10.0f);


    free(data);
    free(coords);
    free(gdata);

    delete gpuNUFFTOp;

    EXPECT_EQ(1, 1);
}

TEST(TestGPUGpuNUFFTConv,KernelCall1SectorArbSW)
{
  int kernel_width = 3;

  //Image
  int im_width = 10;

  //Data
  int data_entries = 2;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  data[0].x = 1;
  data[0].y = 1;
  data[1].x = 1;
  data[1].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  coords[0] = 0; // x
  coords[1] = (DType)0.4;
  coords[2] = 0; // y
  coords[3] = 0; 
  coords[4] = 0; // z 
  coords[5] = 0;

  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;

  //Output Grid

  // sectors of data, count and start indices
  // different sector widths, some no integer multiple of im_width 
  for (int sector_width = 9; sector_width<=9; sector_width++)
  {
    gpuNUFFT::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

    gpuNUFFT::Dimensions imgDims;
    imgDims.width = im_width;
    imgDims.height = im_width;
    imgDims.depth = im_width;

    gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true); 
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

    gpuNUFFT::Array<DType2> dataArray;
    dataArray.data = data;
    dataArray.dim.length = data_entries;

    gpuNUFFT::Array<CufftType> gdataArray;

    gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
    //Output Grid
    CufftType* gdata = gdataArray.data;

    int center_slice = im_width / 2;

    if (DEBUG)
      for (int j=0; j<im_width; j++)
      {
        for (int i=0; i<im_width; i++)
          printf("%.4f ",gdata[get3DC2lin(i,j,center_slice,im_width)].x);
        printf("\n");
      }

      if (DEBUG) printf("test %f \n",gdata[4].x);
      int index = get3DC2lin(5,5,5,im_width);
      if (DEBUG) printf("index to test %d\n",index);
      EXPECT_EQ(index,555);
      EXPECT_NEAR(1.0f,gdata[index].x,epsilon);
      EXPECT_NEAR(0.4502,gdata[get3DC2lin(5,4,center_slice,im_width)].x,epsilon*10.0f);
      EXPECT_NEAR(0.4502,gdata[get3DC2lin(4,5,center_slice,im_width)].x,epsilon*10.0f);
      EXPECT_NEAR(0.4502,gdata[get3DC2lin(5,6,center_slice,im_width)].x,epsilon*10.0f);

      EXPECT_NEAR(0.2027,gdata[get3DC2lin(6,6,center_slice,im_width)].x,epsilon*10.0f);
      EXPECT_NEAR(0.2027,gdata[get3DC2lin(4,4,center_slice,im_width)].x,epsilon*10.0f);
      EXPECT_NEAR(0.2027,gdata[get3DC2lin(4,6,center_slice,im_width)].x,epsilon*10.0f);

      EXPECT_NEAR(0.4502,gdata[get3DC2lin(9,4,center_slice,im_width)].x,epsilon*10.0f);
      EXPECT_NEAR(0.4502,gdata[get3DC2lin(8,5,center_slice,im_width)].x,epsilon*10.0f);
      EXPECT_NEAR(0.4502,gdata[get3DC2lin(9,6,center_slice,im_width)].x,epsilon*10.0f);

      EXPECT_NEAR(0.2027,gdata[get3DC2lin(8,6,center_slice,im_width)].x,epsilon*10.0f);
      EXPECT_NEAR(0.2027,gdata[get3DC2lin(8,4,center_slice,im_width)].x,epsilon*10.0f);
      EXPECT_NEAR(0.2027,gdata[get3DC2lin(0,6,center_slice,im_width)].x,epsilon*10.0f);
      
      free(gdata);

      delete gpuNUFFTOp;
    }
    free(data);
    free(coords);
}

TEST(TestGPUGpuNUFFTConv,GPUTest_1SectorKernel5)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  //kernel width
  int kernel_width = 5;

  //Image
  int im_width = 10;

  //Data
  int data_entries = 1;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  data[0].x = 1;
  data[0].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  coords[0] = 0; //should result in 7,7,7 center
  coords[1] = 0;
  coords[2] = 0;

  //Output Grid

  //sectors of data, count and start indices
  int sector_width = 5;

  int sector_count = 1;
  IndType* sectors = (IndType*) calloc(2*sector_count,sizeof(IndType));
  sectors[0]=0;
  sectors[1]=1;

  IndType* sector_centers = (IndType*) calloc(3*sector_count,sizeof(IndType));
  sector_centers[0] = 5;
  sector_centers[1] = 5;
  sector_centers[2] = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  int index = get3DC2lin(5,5,5,im_width);
  if (DEBUG) printf("index to test %d\n",index);
  //EXPECT_EQ(index,2*555);
  EXPECT_NEAR(1.0f,gdata[index].x,epsilon);
  EXPECT_NEAR(0.0049,gdata[get3DC2lin(3,3,5,im_width)].x,epsilon*10.0f);
  EXPECT_NEAR(0.3218,gdata[get3DC2lin(4,4,5,im_width)].x,epsilon*10.0f);
  EXPECT_NEAR(0.5673,gdata[get3DC2lin(5,4,5,im_width)].x,epsilon*10.0f);

  EXPECT_NEAR(0.0697,gdata[get3DC2lin(5,7,5,im_width)].x,epsilon*10.0f);
  EXPECT_NEAR(0.0697,gdata[get3DC2lin(5,3,5,im_width)].x,epsilon*10.0f);

  //for (int j=0; j<im_width; j++)
  //{
  //	for (int i=0; i<im_width; i++)
  //		printf("%.4f ",gdata[get3DC2lin(i,im_width-j,5,im_width)]);
  //	printf("\n");
  //}

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}



TEST(TestGPUGpuNUFFTConv,GPUTest_2SectorsKernel3nData)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  //kernel width
  int kernel_width = 3;

  long kernel_entries = calculateGrid3KernelSize(osr, kernel_width);

  DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
  load1DKernel(kern,kernel_entries,kernel_width,osr);

  //Image
  int im_width = 10;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 0.7f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //1.Sektor
  coords[coord_cnt++] = -0.3f; //x
  coords[coord_cnt++] = -0.1f; 
  coords[coord_cnt++] = 0;    
  coords[coord_cnt++] = 0.5f; 
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0.2f; //y
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0; //z
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //Output Grid

  //sectors of data, count and start indices
  int sector_width = 5;

  int sector_count = 2;
  IndType* sectors = (IndType*) calloc(2*sector_count,sizeof(IndType));
  sectors[0]=0;
  sectors[1]=2;
  sectors[2]=5;

  IndType* sector_centers = (IndType*) calloc(3*sector_count,sizeof(IndType));
  sector_centers[0] = 2;
  sector_centers[1] = 7;
  sector_centers[2] = 5;

  sector_centers[3] = 7;
  sector_centers[4] = 7;
  sector_centers[5] = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  //gpuNUFFT_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

  int index = get3DC2lin(5,5,5,im_width);
  if (DEBUG) printf("index to test %d\n",index);
  //EXPECT_EQ(index,2*555);
  EXPECT_NEAR(1.3152f,gdata[index].x,epsilon);
  EXPECT_NEAR(0.2432,gdata[get3DC2lin(3,6,5,im_width)].x,epsilon*10.0f);

  EXPECT_NEAR(0.2251,gdata[get3DC2lin(1,7,5,im_width)].x,epsilon*10.0f);
  EXPECT_NEAR(0.4502,gdata[get3DC2lin(6,5,5,im_width)].x,epsilon*10.0f);

  EXPECT_NEAR(1.0f,gdata[get3DC2lin(8,8,5,im_width)].x,epsilon*10.0f);
  EXPECT_NEAR(0.2027,gdata[get3DC2lin(9,9,5,im_width)].x,epsilon*10.0f);

  //for (int j=0; j<im_width; j++)
  //{
  //	for (int i=0; i<im_width; i++)
  //		printf("%.4f ",gdata[get3DC2lin(i,im_width-1-j,5,im_width)]);
  //	printf("\n");
  //}

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}


TEST(TestGPUGpuNUFFTConv,GPUTest_8SectorsKernel3nData)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  //kernel width
  int kernel_width = 3;
  //Image
  int im_width = 10;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;

  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 0.7f;
  data[data_cnt++].y= 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;


  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor

  coords[coord_cnt++] = -0.3f; //x
  coords[coord_cnt++] = -0.1f; 
  coords[coord_cnt++] = 0;    
  coords[coord_cnt++] = 0.5f; 
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0.2f; //y
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0; //z
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //sectors of data, count and start indices
  int sector_width = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true); 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  if (DEBUG)
    for (int j=0; j<im_width; j++)
    {
      for (int i=0; i<im_width; i++)
        printf("%.4f ",gdata[get3DC2lin(i,im_width-1-j,5,im_width)].x);
      printf("\n");
    }

    int index = get3DC2lin(5,5,5,im_width);
    if (DEBUG) printf("index to test %d\n",index);
    //EXPECT_EQ(index,2*555);
    EXPECT_NEAR(1.3152f,gdata[index].x,epsilon);
    EXPECT_NEAR(0.2432,gdata[get3DC2lin(3,6,5,im_width)].x,epsilon*10.0f);

    EXPECT_NEAR(0.2251,gdata[get3DC2lin(1,7,5,im_width)].x,epsilon*10.0f);
    EXPECT_NEAR(0.4502,gdata[get3DC2lin(6,5,5,im_width)].x,epsilon*10.0f);

    EXPECT_NEAR(1.0f,gdata[get3DC2lin(8,8,5,im_width)].x,epsilon*10.0f);
    EXPECT_NEAR(0.2027,gdata[get3DC2lin(9,9,5,im_width)].x,epsilon*10.0f);

    free(data);
    free(coords);
    free(gdata);

    delete gpuNUFFTOp;
}


TEST(TestGPUGpuNUFFTConv,GPUTest_8SectorsKernel4nData)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;

  //kernel width
  int kernel_width = 4;

  //Image
  int im_width = 10;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 0.7f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor
  coords[coord_cnt++] = -0.3f; //X
  coords[coord_cnt++] = -0.1f;
  coords[coord_cnt++] = 0; 
  coords[coord_cnt++] = 0.5f; 
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0.2f;//Y
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //sectors of data, count and start indices
  int sector_width = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  //gpuNUFFT_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);


  int index = get3DC2lin(5,5,5,im_width);
  if (DEBUG) printf("index to test %d\n",index);
  //EXPECT_EQ(index,2*555);
  EXPECT_NEAR(1.3558f,gdata[index].x,epsilon);
  EXPECT_NEAR(0.3101f,gdata[get3DC2lin(3,6,5,im_width)].x,epsilon*10.0f);

  EXPECT_NEAR(0.2542f,gdata[get3DC2lin(1,7,5,im_width)].x,epsilon*10.0f);
  EXPECT_NEAR(0.5084f,gdata[get3DC2lin(6,5,5,im_width)].x,epsilon*10.0f);

  EXPECT_NEAR(1.0f,gdata[get3DC2lin(8,8,5,im_width)].x,epsilon*10.0f);
  EXPECT_NEAR(0.2585f,gdata[get3DC2lin(9,9,5,im_width)].x,epsilon*10.0f);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}



TEST(TestGPUGpuNUFFTConv,GPUTest_8SectorsKernel5nData)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;

  //kernel width
  int kernel_width = 5;

  //Image
  int im_width = 10;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 0.7f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor
  coords[coord_cnt++] = -0.3f; 
  coords[coord_cnt++] = -0.1f;
  coords[coord_cnt++] = 0; 
  coords[coord_cnt++] = 0.5f; 
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0.2f;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //sectors of data, count and start indices
  int sector_width = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  //gpuNUFFT_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

  int index = get3DC2lin(5,5,5,im_width);
  if (DEBUG) printf("index to test %d\n",index);

  EXPECT_NEAR(1.3976f,gdata[index].x,epsilon);
  EXPECT_NEAR(0.4268f,gdata[get3DC2lin(3,6,5,im_width)].x,epsilon*10.0f);

  EXPECT_NEAR(0.0430,gdata[get3DC2lin(6,3,5,im_width)].x,epsilon*10.0f);

  EXPECT_NEAR(0.1093f,gdata[get3DC2lin(8,6,5,im_width)].x,epsilon*10.0f);

  if (DEBUG)
    for (int j=0; j<im_width; j++)
    {
      for (int i=0; i<im_width; i++)
        printf("%.4f ",gdata[get3DC2lin(i,im_width-1-j,5,im_width)].x);
      printf("\n");
    }

    free(data);
    free(coords);
    free(gdata);

    delete gpuNUFFTOp;
}


TEST(TestGPUGpuNUFFTConv,GPUTest_8SectorsKernel3nDataw120)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  //kernel width
  int kernel_width = 3;

  //Image
  int im_width = 120;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 0.7f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor
  coords[coord_cnt++] = -0.3f; 
  coords[coord_cnt++] = 0.2f;
  coords[coord_cnt++] = 0;

  coords[coord_cnt++] = -0.1f;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //8.Sektor
  coords[coord_cnt++] = 0; 
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  coords[coord_cnt++] = 0.5f; 
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  coords[coord_cnt++] = 0.3f;
  coords[coord_cnt++] = 0.3f;
  coords[coord_cnt++] = 0;

  //sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  //gpuNUFFT_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

  int index = get3DC2lin(5,5,5,im_width);
  if (DEBUG) printf("index to test %d\n",index);
  //EXPECT_EQ(index,2*555);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}

TEST(TestGPUGpuNUFFTConv,GPUTest_FactorTwoTest)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  //kernel width
  int kernel_width = 3;

  //Image
  int im_width = 16;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor
  coords[coord_cnt++] = 0; 
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0.5f; 
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0; 
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  //gpuNUFFT_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

  int index = get3DC2lin(5,5,5,im_width);
  if (DEBUG) printf("index to test %d\n",index);

  EXPECT_NEAR(gdata[get3DC2lin(8,8,8,16)].x,2.0f,epsilon);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}


TEST(TestGPUGpuNUFFTConv,GPUTest_8SectorsKernel3nDataw32)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  //kernel width
  int kernel_width = 3;

  //Image
  int im_width = 32;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;

  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0;

  data[data_cnt].x = 0.7f;
  data[data_cnt++].y = 0;

  data[data_cnt].x = -0.2f;
  data[data_cnt++].y = 0.8f;

  data[data_cnt].x = -0.2f;
  data[data_cnt++].y = 0.8f;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 0;


  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  coords[coord_cnt++] = -0.3f; 
  coords[coord_cnt++] = -0.1f;
  coords[coord_cnt++] = 0; 
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0.5f; 

  coords[coord_cnt++] = 0.2f;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  //gpuNUFFT_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

  /*for (int j=0; j<im_width; j++)
  {
  for (int i=0; i<im_width; i++)
  {
  float dpr = gdata[get3DC2lin(i,im_width-1-j,16,im_width)].x;
  float dpi = gdata[get3DC2lin(i,im_width-1-j,16,im_width)].y;

  if (abs(dpr) > 0.0f)
  printf("(%d,%d)= %.4f + %.4f i ",i,im_width-1-j,dpr,dpi);
  }
  printf("\n");
  }*/

  EXPECT_NEAR(gdata[get3DC2lin(12,16,16,im_width)].x,0.4289f,epsilon);
  EXPECT_NEAR(gdata[get3DC2lin(13,16,16,im_width)].x,0.6803f,epsilon);
  EXPECT_NEAR(gdata[get3DC2lin(14,16,16,im_width)].x,0.2065f,epsilon);
  EXPECT_NEAR(gdata[get3DC2lin(15,16,16,im_width)].x,-0.1801f,epsilon);//Re
  EXPECT_NEAR(gdata[get3DC2lin(15,16,16,im_width)].y,0.7206f,epsilon);//Im
  EXPECT_NEAR(gdata[get3DC2lin(16,16,16,im_width)].x,-0.4f,epsilon);
  EXPECT_NEAR(gdata[get3DC2lin(16,16,16,im_width)].y,1.6f,epsilon);
  EXPECT_NEAR(gdata[get3DC2lin(17,16,16,im_width)].x,-0.1801f,epsilon);//Re
  EXPECT_NEAR(gdata[get3DC2lin(17,16,16,im_width)].y,0.7206f,epsilon);//Im

  EXPECT_NEAR(gdata[get3DC2lin(12,15,16,im_width)].x,0.1932f,epsilon);
  EXPECT_NEAR(gdata[get3DC2lin(14,17,16,im_width)].x,0.0930f,epsilon);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}


TEST(TestGPUGpuNUFFTConv,MatlabTest_8SK3w32)
{
  //oversampling ratio
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  //kernel width
  int kernel_width = 3;

  long kernel_entries = calculateGrid3KernelSize(osr, kernel_width);

  DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
  load1DKernel(kern,kernel_entries,kernel_width,osr);

  //Image
  int im_width = 32;

  //Data
  int data_entries = 1;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;

  data[data_cnt].x = 0.0046f;
  data[data_cnt++].y = -0.0021f;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;

  coords[coord_cnt++] = 0.2500f;
  coords[coord_cnt++] = -0.4330f;
  coords[coord_cnt++] = 0;

  //Output Grid
  //sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = im_width;
  imgDims.height = im_width;
  imgDims.depth = im_width;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);
  //Output Grid
  CufftType* gdata = gdataArray.data;

  /*for (int j=0; j<im_width; j++)
  {
  for (int i=0; i<im_width; i++)
  {
  float dpr = gdata[get3DC2lin(i,im_width-1-j,16,im_width)];
  float dpi = gdata[get3DC2lin(i,im_width-1-j,16,im_width)+1];

  if (abs(dpr) > 0.0f)
  printf("(%d,%d)= %.4f + %.4f i ",i,im_width-1-j,dpr,dpi);
  }
  printf("\n");
  }*/

  EXPECT_NEAR(gdata[computeXYZ2Lin(23,3,16,imgDims)].x,0.0012f,epsilon);
  EXPECT_NEAR(gdata[computeXYZ2Lin(23,2,16,imgDims)].x,0.0020f,epsilon);
  EXPECT_NEAR(gdata[computeXYZ2Lin(23,1,16,imgDims)].x,0.0007f,epsilon);
  EXPECT_NEAR(gdata[computeXYZ2Lin(24,3,16,imgDims)].x,0.0026f,epsilon);//Re
  EXPECT_NEAR(gdata[computeXYZ2Lin(24,3,16,imgDims)].y,-0.0012f,epsilon);//Im
  EXPECT_NEAR(gdata[computeXYZ2Lin(24,2,16,imgDims)].x,0.0045f,epsilon);
  EXPECT_NEAR(gdata[computeXYZ2Lin(24,1,16,imgDims)].x,0.0016f,epsilon);
  EXPECT_NEAR(gdata[computeXYZ2Lin(25,3,16,imgDims)].x,0.0012f,epsilon);
  EXPECT_NEAR(gdata[computeXYZ2Lin(25,2,16,imgDims)].x,0.0020f,epsilon);//Re
  EXPECT_NEAR(gdata[computeXYZ2Lin(25,2,16,imgDims)].y,-0.0009f,epsilon);//Im
  EXPECT_NEAR(gdata[computeXYZ2Lin(25,1,16,imgDims)].x,0.0007f,epsilon);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}


TEST(TestGPUGpuNUFFTConvAnisotropic,GPUTest_4SectorsKernel4nData)
{
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  int kernel_width = 4;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 0.7f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor
  coords[coord_cnt++] = -0.3f; //X
  coords[coord_cnt++] = -0.1f;
  coords[coord_cnt++] = 0; 
  coords[coord_cnt++] = 0.5f;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0.2f;//Y
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0; //Z
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //sectors of data, count and start indices
  int sector_width = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 10;
  imgDims.height = 10;
  imgDims.depth = 5;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true); 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);

  //Output Grid
  CufftType* gdata = gdataArray.data;

  if (DEBUG) 
  {
    for (unsigned k=0; k<imgDims.depth; k++)
    {
      for (unsigned j=0; j<imgDims.height; j++)
      {
        for (unsigned i=0; i<imgDims.width; i++)
          printf("%.4f ",gdata[computeXYZ2Lin(i,imgDims.width-1-j,k,imgDims)].x);
        printf("\n");
      }
      printf("-------------------------------------------------------------\n");
    }
  }

  int index = computeXYZ2Lin(5,5,2,imgDims);
  if (DEBUG) printf("index to test %d\n",index);
  //EXPECT_EQ(index,2*555);
  EXPECT_NEAR(1.1548f,gdata[index].x,epsilon);
  EXPECT_NEAR(0.2636f,gdata[computeXYZ2Lin(3,6,2,imgDims)].x,epsilon*10.0f);

  EXPECT_NEAR(0.2161f,gdata[computeXYZ2Lin(1,7,2,imgDims)].x,epsilon*10.0f);
  EXPECT_NEAR(0.4335,gdata[computeXYZ2Lin(6,5,2,imgDims)].x,epsilon*10.0f);

  EXPECT_NEAR(0.8518,gdata[computeXYZ2Lin(8,8,2,imgDims)].x,epsilon*10.0f);
  EXPECT_NEAR(0.2197f,gdata[computeXYZ2Lin(9,9,2,imgDims)].x,epsilon*10.0f);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}

TEST(TestGPUGpuNUFFTConvAnisotropic,GPUTest_32SectorsKernel4nData)
{
  float osr = DEFAULT_OVERSAMPLING_RATIO;
  int kernel_width = 4;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 0.5f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 0.7f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 1;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor
  coords[coord_cnt++] = -0.3f; //X
  coords[coord_cnt++] = -0.1f;
  coords[coord_cnt++] = 0; 
  coords[coord_cnt++] = 0.5f;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0.2f;//Y
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0; //Z
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;
  coords[coord_cnt++] = 0;

  //sectors of data, count and start indices
  int sector_width = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 20;
  imgDims.height = 20;
  imgDims.depth = 10;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true); 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);

  //Output Grid
  CufftType* gdata = gdataArray.data;


  //	if (DEBUG) 
  //  {
  //    for (int k=0; k<imgDims.depth; k++)
  //    {
  //      for (int j=0; j<imgDims.height; j++)
  //	    {
  //		    for (int i=0; i<imgDims.width; i++)
  //			    printf("%.4f ",gdata[computeXYZ2Lin(i,imgDims.width-1-j,k,imgDims)].x);
  //		    printf("\n");
  //	    }
  //     printf("-------------------------------------------------------------\n");
  //  }
  //  }


  int index = computeXYZ2Lin(10,10,5,imgDims);
  if (DEBUG) printf("index to test %d\n",index);
  //EXPECT_EQ(index,2*555);
  EXPECT_NEAR(1.0f,gdata[index].x,epsilon);
  EXPECT_NEAR(0.8639f,gdata[computeXYZ2Lin(9,10,5,imgDims)].x,epsilon*10.0f);

  EXPECT_NEAR(0.2582f,gdata[computeXYZ2Lin(1,11,5,imgDims)].x,epsilon*10.0f);
  EXPECT_NEAR(0.1808f,gdata[computeXYZ2Lin(7,11,5,imgDims)].x,epsilon*10.0f);

  EXPECT_NEAR(0.2231f,gdata[computeXYZ2Lin(9,9,4,imgDims)].x,epsilon*10.0f);
  EXPECT_NEAR(0.2231f,gdata[computeXYZ2Lin(9,9,6,imgDims)].x,epsilon*10.0f);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}

TEST(TestGPUGpuNUFFTConvAnisotropic,GPUTest_20x20x10_osf_15)
{
  float osr = 1.5f;
  int kernel_width = 4;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 1.0f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 3.0f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 3.0f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 4.0f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 5.0f;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor
  coords[coord_cnt++] = -0.3f; //X
  coords[coord_cnt++] = 0.3f;
  coords[coord_cnt++] = 0.0f; 
  coords[coord_cnt++] = 0.5f;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0.2f;//Y
  coords[coord_cnt++] = 0.0f;
  coords[coord_cnt++] = 0.0f;
  coords[coord_cnt++] = 0.0f;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = -0.25f; //Z
  coords[coord_cnt++] = -0.005f;
  coords[coord_cnt++] = 0.0f;
  coords[coord_cnt++] = 0.1f;
  coords[coord_cnt++] = 0.22f;

  //sectors of data, count and start indices
  int sector_width = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 20;
  imgDims.height = 20;
  imgDims.depth = 10;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true); 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);

  //Output Grid
  CufftType* gdata = gdataArray.data;

  EXPECT_NEAR(0.5853f,gdata[computeXYZ2Lin(6,21,3,gdataArray.dim)].x,epsilon);
  EXPECT_NEAR(0.9445f,gdata[computeXYZ2Lin(6,21,4,gdataArray.dim)].x,epsilon);
  EXPECT_NEAR(4.8931f,gdata[computeXYZ2Lin(24,24,11,gdataArray.dim)].x,10*epsilon);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}


TEST(TestGPUGpuNUFFTConvAnisotropic,GPUTest_20x20x10_osf_15_Balanced)
{
  float osr = 1.5f;
  int kernel_width = 4;

  //Data
  int data_entries = 5;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  data[data_cnt].x = 1.0f;
  data[data_cnt++].y = 0.5f;

  data[data_cnt].x = 3.0f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 3.0f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 4.0f;
  data[data_cnt++].y = 1;

  data[data_cnt].x = 5.0f;
  data[data_cnt++].y = 1;

  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  int coord_cnt = 0;
  //7.Sektor
  coords[coord_cnt++] = -0.3f; //X
  coords[coord_cnt++] = 0.3f;
  coords[coord_cnt++] = 0.0f; 
  coords[coord_cnt++] = 0.5f;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = 0.2f;//Y
  coords[coord_cnt++] = 0.0f;
  coords[coord_cnt++] = 0.0f;
  coords[coord_cnt++] = 0.0f;
  coords[coord_cnt++] = 0.3f;

  coords[coord_cnt++] = -0.25f; //Z
  coords[coord_cnt++] = -0.005f;
  coords[coord_cnt++] = 0.0f;
  coords[coord_cnt++] = 0.1f;
  coords[coord_cnt++] = 0.22f;

  //sectors of data, count and start indices
  int sector_width = 5;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 20;
  imgDims.height = 20;
  imgDims.depth = 10;

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true); 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;

  gpuNUFFT::Array<CufftType> gdataArray;

  gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::CONVOLUTION);

  //Output Grid
  CufftType* gdata = gdataArray.data;

  EXPECT_NEAR(0.5853f,gdata[computeXYZ2Lin(6,21,3,gdataArray.dim)].x,epsilon);
  EXPECT_NEAR(0.9445f,gdata[computeXYZ2Lin(6,21,4,gdataArray.dim)].x,epsilon);
  EXPECT_NEAR(4.8931f,gdata[computeXYZ2Lin(24,24,11,gdataArray.dim)].x,10*epsilon);

  free(data);
  free(coords);
  free(gdata);

  delete gpuNUFFTOp;
}

float randomNumber(float start, float stop) {
  assert(stop >= start);
  return start + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (stop - start)));
}

TEST(TestOperatorFactory, DISABLED_GPUTest_256x256x128_osf_15_Balanced)
{
  float osr = 1.5f;
  int kernel_width = 3;

  //Data
  int nFE = 256;
  int nSpokes = 128;

  int data_entries = nFE * nSpokes;

  DType2* data = (DType2*)calloc(data_entries, sizeof(DType2)); //2* re + im
  int data_cnt = 0;
  for (int data_cnt = 0; data_cnt < data_entries; data_cnt++) {
    data[data_cnt].x = 1.0f;
    data[data_cnt].y = 0.5f;
  }
  
  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z)
  DType* coords = (DType*)calloc(3 * data_entries, sizeof(DType));//3* x,y,z
  for (int coord_cnt = 0; coord_cnt < 3 * data_entries; coord_cnt++) {
    coords[coord_cnt++] = randomNumber(-0.5,0.5);
    coords[coord_cnt++] = randomNumber(-0.5, 0.5);
    coords[coord_cnt] = randomNumber(-0.5, 0.5);
  }
  
  //sectors of data, count and start indices
  int sector_width = 8;

  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 256;
  imgDims.height = 256;
  imgDims.depth = 128;

  for (int cnt = 0; cnt < 10000; cnt++) {
    gpuNUFFT::GpuNUFFTOperatorFactory factory(true, true, true);
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData, kernel_width, sector_width, osr, imgDims);

    gpuNUFFT::Array<DType2> dataArray;
    dataArray.data = data;
    dataArray.dim.length = data_entries;

    gpuNUFFT::Array<CufftType> gdataArray;

    gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray);
    //Output Grid
    CufftType* gdata = gdataArray.data;

    gpuNUFFT::Array<DType2> data = gpuNUFFTOp->performForwardGpuNUFFT(gdataArray);

    free(gdata);
    free(data.data);
    delete gpuNUFFTOp;
  }
  
  free(data);
  free(coords);
}
