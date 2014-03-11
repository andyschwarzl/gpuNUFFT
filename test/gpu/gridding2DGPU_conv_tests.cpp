#include <limits.h>

#include "gtest/gtest.h"

#include "gridding_operator_factory.hpp"

#define epsilon 0.0001f

#define get2DC2lin(_x,_y,_width) ((_x) + (_width) *(_y))

TEST(Test2DGPUGriddingConv,KernelCall1Sector)
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
    DType* coords = (DType*) calloc(2*data_entries,sizeof(DType));//2* x,y
	coords[0] = 0; //should result in 7,7 center
	coords[1] = 0;

	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;

	//Output Grid
  
	//sectors of data, count and start indices
	int sector_width = 5;

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

  if (DEBUG)
	  for (int j=0; j<im_width; j++)
	  {
		  for (int i=0; i<im_width; i++)
			  printf("%.4f ",gdata[get2DC2lin(i,j,im_width)].x);
		  printf("\n");
	  }
  if (DEBUG)
	  printf("test %f \n",gdata[4].x);
	int index = get2DC2lin(5,5,im_width);
	if (DEBUG) 
    printf("index to test %d\n",index);
	EXPECT_EQ(index,55);
	EXPECT_NEAR(1.0f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.4502,gdata[get2DC2lin(5,4,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get2DC2lin(4,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get2DC2lin(5,6,im_width)].x,epsilon*10.0f);

	EXPECT_NEAR(0.2027,gdata[get2DC2lin(6,6,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get2DC2lin(4,4,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get2DC2lin(4,6,im_width)].x,epsilon*10.0f);

	free(data);
	free(coords);
	free(gdata);

	delete griddingOp;

	EXPECT_EQ(1, 1);
}


TEST(Test2DGPUGriddingConv,GPUTest_1SectorKernel5)
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
    DType* coords = (DType*) calloc(2*data_entries,sizeof(DType));//2* x,y,z
	coords[0] = 0; //should result in 7,7 center
	coords[1] = 0;

	//sectors of data, count and start indices
	int sector_width = 5;

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);
	
    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

	int index = get2DC2lin(5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.0f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.0049,gdata[get2DC2lin(3,3,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.3218,gdata[get2DC2lin(4,4,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.5673,gdata[get2DC2lin(5,4,im_width)].x,epsilon*10.0f);

	EXPECT_NEAR(0.0697,gdata[get2DC2lin(5,7,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.0697,gdata[get2DC2lin(5,3,im_width)].x,epsilon*10.0f);
	
	//for (int j=0; j<im_width; j++)
	//{
	//	for (int i=0; i<im_width; i++)
	//		printf("%.4f ",gdata[get2DC2lin(i,im_width-j,5,im_width)]);
	//	printf("\n");
	//}

	free(data);
	free(coords);
	free(gdata);
	
	delete griddingOp;
}



TEST(Test2DGPUGriddingConv,GPUTest_2SectorsKernel3nData)
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
    DType* coords = (DType*) calloc(2*data_entries,sizeof(DType));//2* x,y
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
	
	//sectors of data, count and start indices
	int sector_width = 5;
	
	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);
	
    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

	//gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

	int index = get2DC2lin(5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.3152f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.2432,gdata[get2DC2lin(3,6,im_width)].x,epsilon*10.0f);
	
	EXPECT_NEAR(0.2251,gdata[get2DC2lin(1,7,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get2DC2lin(6,5,im_width)].x,epsilon*10.0f);

	EXPECT_NEAR(1.0f,gdata[get2DC2lin(8,8,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get2DC2lin(9,9,im_width)].x,epsilon*10.0f);
	
	//for (int j=0; j<im_width; j++)
	//{
	//	for (int i=0; i<im_width; i++)
	//		printf("%.4f ",gdata[get2DC2lin(i,im_width-1-j,5,im_width)]);
	//	printf("\n");
	//}

	free(data);
	free(coords);
	free(gdata);
	
	delete griddingOp;
}


TEST(Test2DGPUGriddingConv,GPUTest_8SectorsKernel3nData)
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
    DType* coords = (DType*) calloc(2*data_entries,sizeof(DType));//2* x,y
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
	
	//sectors of data, count and start indices
	int sector_width = 5;
	
    GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);
	
    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

  if (DEBUG)
	  for (int j=0; j<im_width; j++)
	  {
		  for (int i=0; i<im_width; i++)
			  printf("%.4f ",gdata[get2DC2lin(i,im_width-1-j,im_width)].x);
		  printf("\n");
	  }

	int index = get2DC2lin(5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.3152f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.2432,gdata[get2DC2lin(3,6,im_width)].x,epsilon*10.0f);
	
	EXPECT_NEAR(0.2251,gdata[get2DC2lin(1,7,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get2DC2lin(6,5,im_width)].x,epsilon*10.0f);

	EXPECT_NEAR(1.0f,gdata[get2DC2lin(8,8,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get2DC2lin(9,9,im_width)].x,epsilon*10.0f);

	free(data);
	free(coords);
	free(gdata);
	
	delete griddingOp;
}


TEST(Test2DGPUGriddingConv,GPUTest_8SectorsKernel4nData)
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
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//2* x,y
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
	
	//sectors of data, count and start indices
	int sector_width = 5;

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);
	
    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);

	//Output Grid
	CufftType* gdata = gdataArray.data;

	int index = get2DC2lin(5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.3558f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.3101f,gdata[get2DC2lin(3,6,im_width)].x,epsilon*10.0f);
	
	EXPECT_NEAR(0.2542f,gdata[get2DC2lin(1,7,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.5084f,gdata[get2DC2lin(6,5,im_width)].x,epsilon*10.0f);

	EXPECT_NEAR(1.0f,gdata[get2DC2lin(8,8,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.2585f,gdata[get2DC2lin(9,9,im_width)].x,epsilon*10.0f);
	
	//for (int j=0; j<im_width; j++)
	//{
	//	for (int i=0; i<im_width; i++)
	//		printf("%.4f ",gdata[get2DC2lin(i,im_width-1-j,5,im_width)]);
	//	printf("\n");
	//}

	free(data);
	free(coords);
	free(gdata);
	
	delete griddingOp;
}

TEST(Test2DGPUGriddingConv,GPUTest_8SectorsKernel5nData)
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
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//2* x,y
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

	//sectors of data, count and start indices
	int sector_width = 5;

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

	int index = get2DC2lin(5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	
	EXPECT_NEAR(1.3970f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.4256f,gdata[get2DC2lin(3,6,im_width)].x,epsilon*10.0f);
	
	EXPECT_NEAR(0.0430,gdata[get2DC2lin(6,3,im_width)].x,epsilon*10.0f);
	
	EXPECT_NEAR(0.1093f,gdata[get2DC2lin(8,6,im_width)].x,epsilon*10.0f);
	
  if (DEBUG)
	  for (int j=0; j<im_width; j++)
	  {
		  for (int i=0; i<im_width; i++)
			  printf("%.4f ",gdata[get2DC2lin(i,im_width-1-j,im_width)].x);
		  printf("\n");
	  }

	free(data);
	free(coords);
	free(gdata);

	delete griddingOp;
}


TEST(Test2DGPUGriddingConv,GPUTest_8SectorsKernel3nDataw120)
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
	//in triplets (x,y)
    DType* coords = (DType*) calloc(2*data_entries,sizeof(DType));//2* x,y
	int coord_cnt = 0;
	//x
	coords[coord_cnt++] = -0.3f; 
	coords[coord_cnt++] = 0.2f;
	coords[coord_cnt++] = -0.1f;
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0; 
	
	//y
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0.5f; 
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0.3f;
	coords[coord_cnt++] = 0.3f;

	//sectors of data, count and start indices
	int sector_width = 8;

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

	int index = get2DC2lin(5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	
	free(data);
	free(coords);
	free(gdata);

	delete griddingOp;
}

TEST(Test2DGPUGriddingConv,GPUTest_FactorTwoTest)
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
    DType* coords = (DType*) calloc(2*data_entries,sizeof(DType));//2* x,y
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
	
	//sectors of data, count and start indices
	int sector_width = 8;
	
    GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

	int index = get2DC2lin(5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	
	EXPECT_NEAR(gdata[get2DC2lin(8,8,16)].x,2.0f,epsilon);

	free(data);
	free(coords);
	free(gdata);

	delete griddingOp;
}


TEST(Test2DGPUGriddingConv,GPUTest_8SectorsKernel3nDataw32)
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
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//2* x,y
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

	//sectors of data, count and start indices
	int sector_width = 8;
	
    GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

    //gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

	/*for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
		{
			float dpr = gdata[get2DC2lin(i,im_width-1-j,16,im_width)].x;
			float dpi = gdata[get2DC2lin(i,im_width-1-j,16,im_width)].y;

			if (abs(dpr) > 0.0f)
				printf("(%d,%d)= %.4f + %.4f i ",i,im_width-1-j,dpr,dpi);
		}
		printf("\n");
	}*/

	EXPECT_NEAR(gdata[get2DC2lin(12,16,im_width)].x,0.4289f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(13,16,im_width)].x,0.6803f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(14,16,im_width)].x,0.2065f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(15,16,im_width)].x,-0.1801f,epsilon);//Re
	EXPECT_NEAR(gdata[get2DC2lin(15,16,im_width)].y,0.7206f,epsilon);//Im
	EXPECT_NEAR(gdata[get2DC2lin(16,16,im_width)].x,-0.4f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(16,16,im_width)].y,1.6f,epsilon);
  EXPECT_NEAR(gdata[get2DC2lin(17,16,im_width)].x,-0.1801f,epsilon);//Re
	EXPECT_NEAR(gdata[get2DC2lin(17,16,im_width)].y,0.7206f,epsilon);//Im

	EXPECT_NEAR(gdata[get2DC2lin(12,15,im_width)].x,0.1932f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(14,17,im_width)].x,0.0930f,epsilon);
	
	free(data);
	free(coords);
	free(gdata);

	delete griddingOp;
}


TEST(Test2DGPUGriddingConv,MatlabTest_8SK3w32)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;
	//kernel width
	int kernel_width = 3;

	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);

	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

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
    DType* coords = (DType*) calloc(2*data_entries,sizeof(DType));//2* x,y
	int coord_cnt = 0;

	coords[coord_cnt++] = 0.2500f;
	coords[coord_cnt++] = -0.4330f;
	
	//sectors of data, count and start indices
	int sector_width = 8;
	
	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims(im_width,im_width);

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::CONVOLUTION);
	//Output Grid
	CufftType* gdata = gdataArray.data;

	/*for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
		{
			float dpr = gdata[get2DC2lin(i,im_width-1-j,16,im_width)];
			float dpi = gdata[get2DC2lin(i,im_width-1-j,16,im_width)+1];

			if (abs(dpr) > 0.0f)
				printf("(%d,%d)= %.4f + %.4f i ",i,im_width-1-j,dpr,dpi);
		}
		printf("\n");
	}*/

	EXPECT_NEAR(gdata[get2DC2lin(23,3,im_width)].x,0.0012f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(23,2,im_width)].x,0.0020f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(23,1,im_width)].x,0.0007f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(24,3,im_width)].x,0.0026f,epsilon);//Re
	EXPECT_NEAR(gdata[get2DC2lin(24,3,im_width)].y,-0.0012f,epsilon);//Im
	EXPECT_NEAR(gdata[get2DC2lin(24,2,im_width)].x,0.0045f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(24,1,im_width)].x,0.0016f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(25,3,im_width)].x,0.0012f,epsilon);
	EXPECT_NEAR(gdata[get2DC2lin(25,2,im_width)].x,0.0020f,epsilon);//Re
	EXPECT_NEAR(gdata[get2DC2lin(25,2,im_width)].y,-0.0009f,epsilon);//Im
	EXPECT_NEAR(gdata[get2DC2lin(25,1,im_width)].x,0.0007f,epsilon);
	
	free(data);
	free(coords);
	free(gdata);

	delete griddingOp;
}
