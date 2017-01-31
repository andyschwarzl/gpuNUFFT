#include <limits.h>

#include "gtest/gtest.h"

#include "gpuNUFFT_operator_factory.hpp"

#define epsilon 0.01f

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
    for (unsigned k=0; k<gridDims.z; k++)
    {
      for (unsigned j=0; j<gridDims.y; j++)
	    {
		    for (unsigned i=0; i<gridDims.x; i++)
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


TEST(TestGPUGpuNUFFTFFT,KernelCall1Sector)
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
	
	gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::FFT);
	//Output Grid
	CufftType* gdata = gdataArray.data;

//    gpuNUFFT_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,FFT);
	
	if (DEBUG) printf("test %f \n",gdata[4].x);
	int index = get3DC2lin(5,5,5,im_width);
	if (DEBUG) printf("index to test %d\n",index);
	EXPECT_EQ(index,555);
  EXPECT_NEAR(0.2172, gdata[index].x, epsilon);
  EXPECT_NEAR(0.1975, gdata[get3DC2lin(5, 4, 5, im_width)].x, epsilon*10.0f);
  EXPECT_NEAR(0.1975, gdata[get3DC2lin(4, 5, 5, im_width)].x, epsilon*10.0f);
  EXPECT_NEAR(0.001619, gdata[get3DC2lin(0, 1, 5, im_width)].x, epsilon*10.0f);
	
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


TEST(TestGPUGpuNUFFTFFT,GPUTest_Kernel5w64)
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

    //gpuNUFFT_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,FFT);

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
	
	gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::FFT);

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

	delete gpuNUFFTOp;
	//free(sectors);
	//free(sector_centers);
}

TEST(TestGPUGpuNUFFTFFT,GPUTest_FactorTwoTest)
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
	
	gdataArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray,gpuNUFFT::FFT);
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

  EXPECT_NEAR(gdata[get3DC2lin(8, 8, 5, 16)].x, 0.28397, epsilon);
  EXPECT_NEAR(gdata[get3DC2lin(2, 2, 5, 16)].x, 0.0039668, epsilon);
  EXPECT_NEAR(gdata[get3DC2lin(4, 3, 5, 16)].x, 0.0159846, epsilon);
	
	free(data);
	free(coords);
	free(gdata);
	
	delete gpuNUFFTOp;
}

TEST(TestForwardBackward,Test64)
{
  int kernel_width = 3;
	float osf = 1.25;//oversampling ratio
	int sector_width = 8;
	
	//Data
	int data_entries = 2;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
	data[0].x = 5;//Re
	data[0].y = 0;//Im
	data[1].x = 1;//Re
	data[1].y = 0;//Im

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z) as structure of array
	//p0 = (0,0,0)
	//p1 0 (0.25,0.25,0.25)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	coords[0] = 0.00; //x0
	coords[1] = 0.25; //x1
	
	coords[2] = 0.00; //y0
	coords[3] = 0.25; //y0
	
	coords[4] = 0.00; //z0
	coords[5] = 0.25; //z1

  //Input data array, complex values
	gpuNUFFT::Array<DType2> dataArray;
	dataArray.data = data;
	dataArray.dim.length = data_entries;
	
	//Input array containing trajectory in k-space
	gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

	gpuNUFFT::Dimensions imgDims;
	imgDims.width = 64;
	imgDims.height = 64;
	imgDims.depth = 64;

  //precomputation performed by factory
  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true); 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osf,imgDims);

	//Output Array
	gpuNUFFT::Array<CufftType> imgArray;
	
	//Perform FT^H Operation
	imgArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray);
	
	//Output Image
	CufftType* gdata = imgArray.data;
	
	//Perform FT Operation
	gpuNUFFT::Array<CufftType> kSpace = gpuNUFFTOp->performForwardGpuNUFFT(imgArray);
	
	printf("contrast %f \n",kSpace.data[0].x/kSpace.data[1].x);
  EXPECT_NEAR(data[0].x/data[1].x,5.0,epsilon);
	free(data);
	free(coords);
	free(gdata);

  delete gpuNUFFTOp;
}

class TestFFT : public ::testing::Test
{
 public:
  TestFFT()
  {
    osf = 1.25;//oversampling ratio
  }
  static void SetUpTestCase()
  {
    gpuNUFFT::Dimensions dims(16,16,6);

    data.clear();
    // row major
    for (unsigned slice = 0; slice < dims.depth; slice++)
      for (unsigned row = 0; row < dims.height; ++row)
        for (unsigned column = 0; column < dims.width; ++column)
        {
          CufftType val;
          val.x = (DType)computeXYZ2Lin(column, row, slice, dims);
          val.y = (DType)0.0;
          data.push_back(val);
        }
  }
  static const int kernel_width = 3;
	float osf;//oversampling ratio
	static const int sector_width = 8;
  static std::vector<CufftType> data;
};

std::vector<CufftType> TestFFT::data;
  
void debug(const char* title, std::vector<CufftType> data, gpuNUFFT::Dimensions imgDims)
{
  if (DEBUG) 
  {
    printf("%s:\n",title);
    for (unsigned k=0; k<std::max(imgDims.depth,1u); k++)
    {
      for (unsigned j=0; j<imgDims.height; j++)
      {
        for (unsigned i=0; i<imgDims.width; i++)
          printf("%3.0f ",data[computeXYZ2Lin(i,j,k,imgDims)].x);
        printf("\n");
      }
      printf("---------------------------------------------------\n");
    }
  }
}

TEST_F(TestFFT,Test8x8)
{
	gpuNUFFT::Dimensions imgDims;
	imgDims.width = 8;
	imgDims.height = 8;
   
  //Input data array, complex values
	gpuNUFFT::Array<CufftType> dataArray;
	dataArray.data = &data[0];
	dataArray.dim = imgDims;

  CufftType* data_d;
  allocateAndCopyToDeviceMem<CufftType>(&data_d,dataArray.data,dataArray.count());

  gpuNUFFT::GpuNUFFTInfo gi_host;

  gi_host.is2Dprocessing = true;
  gi_host.gridDims.x = imgDims.width;
  gi_host.gridDims.y = imgDims.height;
  gi_host.n_coils_cc = 1;

  initConstSymbol("GI",&gi_host,sizeof(gpuNUFFT::GpuNUFFTInfo));

  debug("Input:",data,imgDims);
  
  performFFTShift(data_d,gpuNUFFT::FORWARD,imgDims,&gi_host);

  std::vector<CufftType> result(imgDims.count());
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output FFTSHIFT(data):",result,imgDims);
  
  performFFTShift(data_d,gpuNUFFT::INVERSE,imgDims,&gi_host);
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output IFFTSHIFT(FFTSHIFT(data)):",result,imgDims);
  
  for (unsigned i=0; i < imgDims.count(); i++)
  {
    EXPECT_NEAR(data[i].x,result[i].x,epsilon);
    EXPECT_NEAR(data[i].y,result[i].y,epsilon);
  }

  cudaFree(data_d);
}

TEST_F(TestFFT,Test9x9)
{
	gpuNUFFT::Dimensions imgDims;
	imgDims.width = 9;
	imgDims.height = 9;
   
  //Input data array, complex values
	gpuNUFFT::Array<CufftType> dataArray;
	dataArray.data = &data[0];
	dataArray.dim = imgDims;

  CufftType* data_d;
  allocateAndCopyToDeviceMem<CufftType>(&data_d,dataArray.data,dataArray.count());

  gpuNUFFT::GpuNUFFTInfo gi_host;

  gi_host.is2Dprocessing = true;
  gi_host.gridDims.x = imgDims.width;
  gi_host.gridDims.y = imgDims.height;
  gi_host.n_coils_cc = 1;

  initConstSymbol("GI",&gi_host,sizeof(gpuNUFFT::GpuNUFFTInfo));

  debug("Input:",data,imgDims);
  
  performFFTShift(data_d,gpuNUFFT::FORWARD,imgDims,&gi_host);

  std::vector<CufftType> result(imgDims.count());
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output FFTSHIFT(data):",result,imgDims);

  performFFTShift(data_d,gpuNUFFT::INVERSE,imgDims,&gi_host);
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output IFFTSHIFT(FFTSHIFT(data)):",result,imgDims);
  
  for (unsigned i=0; i < imgDims.count(); i++)
  {
    EXPECT_NEAR(data[i].x,result[i].x,epsilon);
    EXPECT_NEAR(data[i].y,result[i].y,epsilon);
  }

  cudaFree(data_d);
}

TEST_F(TestFFT,Test9x11)
{
	gpuNUFFT::Dimensions imgDims;
	imgDims.width = 9;
	imgDims.height = 11;
   
  //Input data array, complex values
	gpuNUFFT::Array<CufftType> dataArray;
	dataArray.data = &data[0];
	dataArray.dim = imgDims;

  CufftType* data_d;
  allocateAndCopyToDeviceMem<CufftType>(&data_d,dataArray.data,dataArray.count());

  gpuNUFFT::GpuNUFFTInfo gi_host;

  gi_host.is2Dprocessing = true;
  gi_host.gridDims.x = imgDims.width;
  gi_host.gridDims.y = imgDims.height;
  gi_host.n_coils_cc = 1;

  initConstSymbol("GI",&gi_host,sizeof(gpuNUFFT::GpuNUFFTInfo));

  debug("Input:",data,imgDims);
  
  performFFTShift(data_d,gpuNUFFT::FORWARD,imgDims,&gi_host);

  std::vector<CufftType> result(imgDims.count());
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output FFTSHIFT(data):",result,imgDims);

  performFFTShift(data_d,gpuNUFFT::INVERSE,imgDims,&gi_host);
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output IFFTSHIFT(FFTSHIFT(data)):",result,imgDims);

  for (unsigned i=0; i < imgDims.count(); i++)
  {
    EXPECT_NEAR(data[i].x,result[i].x,epsilon);
    EXPECT_NEAR(data[i].y,result[i].y,epsilon);
  }

  cudaFree(data_d);
}

TEST_F(TestFFT,Test4x4x4)
{
	gpuNUFFT::Dimensions imgDims;
	imgDims.width = 4;
	imgDims.height = 4;
	imgDims.depth = 4;
   
  //Input data array, complex values
	gpuNUFFT::Array<CufftType> dataArray;
	dataArray.data = &data[0];
	dataArray.dim = imgDims;

  CufftType* data_d;
  allocateAndCopyToDeviceMem<CufftType>(&data_d,dataArray.data,dataArray.count());

  gpuNUFFT::GpuNUFFTInfo gi_host;

  gi_host.is2Dprocessing = false;

  gi_host.gridDims.x = imgDims.width;
  gi_host.gridDims.y = imgDims.height;
  gi_host.gridDims.z = imgDims.depth;
  gi_host.n_coils_cc = 1;

  initConstSymbol("GI",&gi_host,sizeof(gpuNUFFT::GpuNUFFTInfo));

  debug("Input:", data, imgDims);
  
  performFFTShift(data_d, gpuNUFFT::FORWARD, imgDims, &gi_host);

  std::vector<CufftType> result(imgDims.count());
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output FFTSHIFT(data):",result,imgDims);

  performFFTShift(data_d,gpuNUFFT::INVERSE,imgDims,&gi_host);
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output IFFTSHIFT(FFTSHIFT(data)):",result,imgDims);


  for (unsigned i=0; i < imgDims.count(); i++)
  {
    EXPECT_NEAR(data[i].x,result[i].x,epsilon);
    EXPECT_NEAR(data[i].y,result[i].y,epsilon);
  }

  cudaFree(data_d);
}

TEST_F(TestFFT,Test8x11x4)
{
	gpuNUFFT::Dimensions imgDims;
	imgDims.width = 8;
	imgDims.height = 11;
	imgDims.depth = 4;
   
  //Input data array, complex values
	gpuNUFFT::Array<CufftType> dataArray;
	dataArray.data = &data[0];
	dataArray.dim = imgDims;

  CufftType* data_d;
  allocateAndCopyToDeviceMem<CufftType>(&data_d,dataArray.data,dataArray.count());

  gpuNUFFT::GpuNUFFTInfo gi_host;

  gi_host.is2Dprocessing = false;

  gi_host.gridDims.x = imgDims.width;
  gi_host.gridDims.y = imgDims.height;
  gi_host.gridDims.z = imgDims.depth;
  gi_host.n_coils_cc = 1;

  initConstSymbol("GI",&gi_host,sizeof(gpuNUFFT::GpuNUFFTInfo));

  debug("Input:", data, imgDims);
  
  performFFTShift(data_d, gpuNUFFT::FORWARD, imgDims, &gi_host);

  std::vector<CufftType> result(imgDims.count());
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output FFTSHIFT(data):",result,imgDims);

  performFFTShift(data_d,gpuNUFFT::INVERSE,imgDims,&gi_host);
  copyFromDevice(data_d, &result[0], dataArray.count());

  debug("Output IFFTSHIFT(FFTSHIFT(data)):",result,imgDims);


  for (unsigned i=0; i < imgDims.count(); i++)
  {
    EXPECT_NEAR(data[i].x,result[i].x,epsilon);
    EXPECT_NEAR(data[i].y,result[i].y,epsilon);
  }

  cudaFree(data_d);
}

TEST(TestForwardBackward,Test_GpuArray)
{
  //Test the same as above but use GpuArray data structure  

  int kernel_width = 3;
	float osf = 1.25;//oversampling ratio
	int sector_width = 8;
	
	//Data
	int data_entries = 2;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
	data[0].x = 5;//Re
	data[0].y = 0;//Im
	data[1].x = 1;//Re
	data[1].y = 0;//Im

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z) as structure of array
	//p0 = (0,0,0)
	//p1 0 (0.25,0.25,0.25)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	coords[0] = 0.00; //x0
	coords[1] = 0.25; //x1
	
	coords[2] = 0.00; //y0
	coords[3] = 0.25; //y0
	
	coords[4] = 0.00; //z0
	coords[5] = 0.25; //z1

  //Input data array, complex values
  //and copy to GPU 
	gpuNUFFT::GpuArray<DType2> dataArray_gpu;
	dataArray_gpu.dim.length = data_entries;

  allocateAndCopyToDeviceMem<DType2>(&dataArray_gpu.data,data,data_entries);
	
	//Input array containing trajectory in k-space
	gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;

	gpuNUFFT::Dimensions imgDims;
	imgDims.width = 64;
	imgDims.height = 64;
	imgDims.depth = 64;

  //precomputation performed by factory
  gpuNUFFT::GpuNUFFTOperatorFactory factory; 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osf,imgDims);

	//Output Array
	gpuNUFFT::GpuArray<CufftType> imgArray_gpu;
	imgArray_gpu.dim = imgDims;
  allocateDeviceMem<CufftType>(&imgArray_gpu.data,imgArray_gpu.count());
	
	//Perform FT^H Operation
	gpuNUFFTOp->performGpuNUFFTAdj(dataArray_gpu, imgArray_gpu);
	
	//Perform FT Operation
  gpuNUFFTOp->performForwardGpuNUFFT(imgArray_gpu,dataArray_gpu);
	copyFromDevice(dataArray_gpu.data,data,data_entries);
	
	printf("contrast %f \n",data[0].x/data[1].x);
  EXPECT_NEAR(data[0].x/data[1].x,5.0,epsilon);

	free(data);
	free(coords);
  freeTotalDeviceMemory(dataArray_gpu.data,imgArray_gpu.data,NULL);

  delete gpuNUFFTOp;
}
