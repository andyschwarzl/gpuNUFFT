#include <limits.h>

#include "gtest/gtest.h"

#include "gridding_operator_factory.hpp"

#define epsilon 0.0001f

#define get3DC2lin(_x,_y,_z,_width) ((_x) + (_width) * ( (_y) + (_z) * (_width)))

void fftShift(int* data,int N,IndType3 gridDims,IndType3 offset)
{
  int t = 0;
  int x, y, z, x_opp, y_opp, z_opp, ind_opp;
  while (t < N/2) 
  { 
    getCoordsFromIndex(t, &x, &y, &z, gridDims.x,gridDims.y,gridDims.z);
    //calculate "opposite" coord pair
    x_opp = (x + offset.x) % gridDims.x;
    y_opp = (y + offset.y) % gridDims.y;
    z_opp = (z + offset.z) % gridDims.z;
    ind_opp = computeXYZ2Lin(x_opp,y_opp,z_opp,gridDims);
    //swap points
    int temp = data[t];
    data[t] = data[ind_opp];
    data[ind_opp] = temp;

    t++;
  }
}

void debugGrid(int* data,IndType3 gridDims)
{
  if (DEBUG) 
  {
    for (int k=0; k<gridDims.z; k++)
    {
      for (int j=0; j<gridDims.y; j++)
	    {
		    for (int i=0; i<gridDims.x; i++)
			    printf("%d ",data[computeXYZ2Lin(i,gridDims.x-1-j,k,gridDims)]);
		    printf("\n");
	    }
     printf("-------------------------------------------------------------\n");
    }
  }
}

TEST(TestFFTShift,Shift_2x2x2)
{
  int N = 8;
  int data[8];
  for (int i = 0; i<N; i++)
    data[i] = i;
  
  IndType3 gridDims;
  gridDims.x = 2;
  gridDims.y = 2;
  gridDims.z = 2;

  IndType3 offset;
  offset.x = 1;
  offset.y = 1;
  offset.z = 1;
  
  int expected[] = {7,6,5,4,3,2,1,0};

  debugGrid(data,gridDims);

  fftShift(&data[0],N,gridDims,offset);
    
  for (int i=0; i<N;i++)
    EXPECT_EQ(expected[i],data[i]);


  printf("shifted: \n");
  debugGrid(data,gridDims);
}


TEST(TestFFTShift,Shift_2x2x1)
{
  const int N = 4;
  int data[N];
  for (int i = 0; i<N; i++)
    data[i] = i;
  
  IndType3 gridDims;
  gridDims.x = 2;
  gridDims.y = 2;
  gridDims.z = 1;

  IndType3 offset;
  offset.x = 1;
  offset.y = 1;
  offset.z = 1;

  int expected[] = {3,2,1,0};

  debugGrid(data,gridDims);

  fftShift(&data[0],N,gridDims,offset);
  
  for (int i=0; i<N;i++)
    EXPECT_EQ(expected[i],data[i]);

  printf("shifted: \n");
  debugGrid(data,gridDims);
}

TEST(TestFFTShift,Shift_4x4x2)
{
  const int N = 32;
  int data[N];
  for (int i = 0; i<N; i++)
    data[i] = i;
  
  IndType3 gridDims;
  gridDims.x = 4;
  gridDims.y = 4;
  gridDims.z = 2;

  IndType3 offset;
  offset.x = 2;
  offset.y = 2;
  offset.z = 1;

  int expected[] = {26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21,10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5};

  debugGrid(data,gridDims);

  fftShift(&data[0],N,gridDims,offset);
  
  for (int i=0; i<N;i++)
    EXPECT_EQ(expected[i],data[i]);

  printf("shifted: \n");
  debugGrid(data,gridDims);
}


TEST(TestGPUGriddingFFT,KernelCall1Sector)
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

	//sectors of data, count and start indices
	int sector_width = 5;
	
	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims;
	imgDims.width = im_width;
	imgDims.height = im_width;
	imgDims.depth = im_width;

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::FFT);
	//Output Grid
	CufftType* gdata = gdataArray.data;

//    gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,FFT);
	
	if (DEBUG) printf("test %f \n",gdata[4].x);
	int index = get3DC2lin(5,5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	EXPECT_EQ(index,555);
	EXPECT_NEAR(0.00097f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.0027f,gdata[get3DC2lin(5,4,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.0027f,gdata[get3DC2lin(4,5,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.3262f,gdata[get3DC2lin(0,1,5,im_width)].x,epsilon*10.0f);
	
	/*for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			printf("%.4f ",gdata[get3DC2lin(i,j,5,im_width)].x);
		printf("\n");
	}*/

	free(data);
	free(coords);
	free(gdata);

	EXPECT_EQ(1, 1);
}


TEST(TestGPUGriddingFFT,GPUTest_Kernel5w64)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;
	//kernel width
	int kernel_width = 5;

	//Image
	int im_width = 64;

	//Data
	int data_entries = 1;
    DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
	data[0].x = 1;
	data[0].y = 0;

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	coords[0] = 0; //should result in 7,7,7 center
	coords[1] = 0;
	coords[2] = 0;

	//sectors of data, count and start indices
	int sector_width = 8;

    //gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,FFT);

	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims;
	imgDims.width = im_width;
	imgDims.height = im_width;
	imgDims.depth = im_width;

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::FFT);

	//Output Grid
	CufftType* gdata = gdataArray.data;

	if (DEBUG) printf("test: 57,33,32 = %.4f\n",gdata[get3DC2lin(57,33,32,im_width)].x);

	EXPECT_NEAR(0.0001,gdata[get3DC2lin(57,33,32,im_width)].x,epsilon*10.0f);

	/*for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			if (abs(gdata[get3DC2lin(i,im_width-j,32,im_width)].x)>0.0f)
				printf("(%d,%d,%d):%.4f ",i,j,32,gdata[get3DC2lin(i,im_width-j,32,im_width)].x);
		printf("\n");
	}*/

	free(data);
	free(coords);
	free(gdata);

	delete griddingOp;
	//free(sectors);
	//free(sector_centers);
}

TEST(TestGPUGriddingFFT,GPUTest_FactorTwoTest)
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
	
	GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Dimensions imgDims;
	imgDims.width = im_width;
	imgDims.height = im_width;
	imgDims.depth = im_width;

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	GriddingND::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;

	GriddingND::Array<CufftType> gdataArray;
	
	gdataArray = griddingOp->performGriddingAdj(dataArray,GriddingND::FFT);
	//Output Grid
	CufftType* gdata = gdataArray.data;

	int index = get3DC2lin(5,5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	
  if (DEBUG)
	  for (int j=0; j<im_width; j++)
	  {
		  for (int i=0; i<im_width; i++)
			  printf("%.4f ",gdata[get3DC2lin(i,j,5,im_width)].x);
		  printf("\n");
	  }

	EXPECT_NEAR(gdata[get3DC2lin(8,8,5,16)].x,0.0514f,epsilon);
	EXPECT_NEAR(gdata[get3DC2lin(2,2,5,16)].x,4.0694f,epsilon);
	EXPECT_NEAR(gdata[get3DC2lin(4,3,5,16)].x,3.22198f,epsilon);
	
	free(data);
	free(coords);
	free(gdata);
	
	delete griddingOp;
}
