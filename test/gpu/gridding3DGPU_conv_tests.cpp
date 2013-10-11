#include <limits.h>

#include "gridding_gpu.hpp"

#include "gtest/gtest.h"

#include "gridding_operator_factory.hpp"


#define epsilon 0.0001f

#define get3DC2lin(_x,_y,_z,_width) ((_x) + (_width) * ( (_y) + (_z) * (_width)))

TEST(TestKernel, LoadKernel) {
	printf("start creating kernel...\n");
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

TEST(TestGPUGriddingConv,KernelCall1Sector)
{
	int kernel_width = 3;
	long kernel_entries = calculateGrid3KernelSize();
	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries);

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
  CufftType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 1; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

	gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 9;
	
	int sector_count = 1;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=1;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	sector_centers[0] = 5;
	sector_centers[1] = 5;
	sector_centers[2] = 5;

	gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);
	
	printf("test %f \n",gdata[4].x);
	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	EXPECT_EQ(index,555);
	EXPECT_NEAR(1.0f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.4502,gdata[get3DC2lin(5,4,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get3DC2lin(4,5,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get3DC2lin(5,6,5,im_width)].x,epsilon*10.0f);

	EXPECT_NEAR(0.2027,gdata[get3DC2lin(6,6,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get3DC2lin(4,4,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get3DC2lin(4,6,5,im_width)].x,epsilon*10.0f);
	
	for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			printf("%.4f ",gdata[get3DC2lin(i,j,5,im_width)].x);
		printf("\n");
	}

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);

	EXPECT_EQ(1, 1);
}


TEST(TestGPUGriddingConv,GPUTest_1SectorKernel5)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;
	//kernel width
	int kernel_width = 5;
	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);
	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

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
    CufftType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 1; // complex /
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 5;
	
	int sector_count = 1;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=1;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	sector_centers[0] = 5;
	sector_centers[1] = 5;
	sector_centers[2] = 5;

	gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);
	
	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
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
	free(kern);
	free(sectors);
	free(sector_centers);
}



TEST(TestGPUGriddingConv,GPUTest_2SectorsKernel3nData)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;
	//kernel width
	int kernel_width = 3;

	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);

	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

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
    CufftType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 1; // complex 
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 5;
	
	int sector_count = 2;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=2;
	sectors[2]=5;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	sector_centers[0] = 2;
	sector_centers[1] = 7;
	sector_centers[2] = 5;

	sector_centers[3] = 7;
	sector_centers[4] = 7;
	sector_centers[5] = 5;

	gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
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
	free(kern);
	free(sectors);
	free(sector_centers);
}


TEST(TestGPUGriddingConv,GPUTest_8SectorsKernel3nData)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;
	//kernel width
	int kernel_width = 3;

	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);

	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

	//Image
	int im_width = 10;

	//Data
	int data_entries = 5;
    DType2* data = (DType2*) calloc(2*data_entries,sizeof(DType2)); //2* re + im
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

	coords[coord_cnt++] = 0.2f;//Y
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0.3f;
	
	coords[coord_cnt++] = 0;//Z
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0;

	//Output Grid
    CufftType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 1; // complex
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 5;
	
	int sector_count = 8;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=0;
	sectors[2]=0;
	sectors[3]=0;
	sectors[4]=0;
	sectors[5]=0;
	sectors[6]=0;
	sectors[7]=2;
	sectors[8]=5;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	int sector_cnt = 0;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

    GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.width  = im_width;
    kSpaceData.dim.height = im_width;
    kSpaceData.dim.depth  = im_width;

    GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernel_width,sector_width,osr);

    //GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance()->createGriddingOperator<DType>(kSpaceData,kernel_width,sector_width,osr);

    griddingOp->setDataCount(data_entries);
	griddingOp->setChnCount(1);	
	griddingOp->setSectorCount(sector_count);
	griddingOp->setOsf(osr);

	//griddingOp->setData(data);
    //griddingOp->setKspaceCoords(coords);
	griddingOp->setDens(NULL);
	griddingOp->setSectors((size_t*)sectors);
	griddingOp->setSectorCenters((size_t*)sector_centers);

    //griddingOp->setKSpaceWidth(im_width);
    //griddingOp->setKSpaceHeight(im_width);
    //griddingOp->setKSpaceDepth(im_width);

	griddingOp->performGriddingAdj(data,&gdata,CONVOLUTION);

    //gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
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
	free(kern);
	free(sectors);
	free(sector_centers);
}


TEST(TestGPUGriddingConv,GPUTest_8SectorsKernel4nData)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;

	//kernel width
	int kernel_width = 4;

	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);

	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

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

	//Output Grid
    CufftType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 1; // complex
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 5;
	
	int sector_count = 8;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=0;
	sectors[2]=0;
	sectors[3]=0;
	sectors[4]=0;
	sectors[5]=0;
	sectors[6]=0;
	sectors[7]=2;
	sectors[8]=5;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	int sector_cnt = 0;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

    gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);


	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.3558f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.3101f,gdata[get3DC2lin(3,6,5,im_width)].x,epsilon*10.0f);
	
	EXPECT_NEAR(0.2542f,gdata[get3DC2lin(1,7,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.5084f,gdata[get3DC2lin(6,5,5,im_width)].x,epsilon*10.0f);

	EXPECT_NEAR(1.0f,gdata[get3DC2lin(8,8,5,im_width)].x,epsilon*10.0f);
	EXPECT_NEAR(0.2585f,gdata[get3DC2lin(9,9,5,im_width)].x,epsilon*10.0f);
	
	//for (int j=0; j<im_width; j++)
	//{
	//	for (int i=0; i<im_width; i++)
	//		printf("%.4f ",gdata[get3DC2lin(i,im_width-1-j,5,im_width)]);
	//	printf("\n");
	//}

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
}



TEST(TestGPUGriddingConv,GPUTest_8SectorsKernel5nData)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;

	//kernel width
	int kernel_width = 5;

	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);

	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

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

	//Output Grid
    CufftType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 1; // complex
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 5;
	
	int sector_count = 8;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=0;
	sectors[2]=0;
	sectors[3]=0;
	sectors[4]=0;
	sectors[5]=0;
	sectors[6]=0;
	sectors[7]=2;
	sectors[8]=5;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	int sector_cnt = 0;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

    gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	
	EXPECT_NEAR(1.3970f,gdata[index].x,epsilon);
	EXPECT_NEAR(0.4256f,gdata[get3DC2lin(3,6,5,im_width)].x,epsilon*10.0f);
	
	EXPECT_NEAR(0.0430,gdata[get3DC2lin(6,3,5,im_width)].x,epsilon*10.0f);
	
	EXPECT_NEAR(0.1093f,gdata[get3DC2lin(8,6,5,im_width)].x,epsilon*10.0f);
	
	for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			printf("%.4f ",gdata[get3DC2lin(i,im_width-1-j,5,im_width)].x);
		printf("\n");
	}

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
}


TEST(TestGPUGriddingConv,GPUTest_8SectorsKernel3nDataw128)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;
	//kernel width
	int kernel_width = 3;

	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);

	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

	//Image
	int im_width = 128;

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

	//Output Grid
    CufftType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 1; // complex
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 5;
	
	int sector_count = 8;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=0;
	sectors[2]=0;
	sectors[3]=0;
	sectors[4]=0;
	sectors[5]=0;
	sectors[6]=0;
	sectors[7]=2;
	sectors[8]=5;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	int sector_cnt = 0;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

    gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	
	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
}

TEST(TestGPUGriddingConv,GPUTest_FactorTwoTest)
{
	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;
	//kernel width
	int kernel_width = 3;

	long kernel_entries = calculateGrid3KernelSize(osr, kernel_width/2.0f);
	printf("Kernel Entries %d\n",kernel_entries);
	
	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	
	loadGrid3Kernel(kern,kernel_entries,kernel_width,osr);

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

	//Output Grid
    CufftType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 1; // complex
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 8;
	
	int sector_count = 8;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=0;
	sectors[2]=0;
	sectors[3]=0;
	sectors[4]=0;
	sectors[5]=0;
	sectors[6]=0;
	sectors[7]=0;
	sectors[8]=5;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	int sector_cnt = 0;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 2;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;
	sector_centers[sector_cnt++] = 7;

    gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	
	EXPECT_NEAR(gdata[get3DC2lin(8,8,8,16)].x,2.0f,epsilon);

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
}


TEST(TestGPUGriddingConv,GPUTest_8SectorsKernel3nDataw32)
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

	//Output Grid
  CufftType* gdata;
	unsigned long dims_g[4];
  dims_g[0] = 1; // complex
	dims_g[1] = (unsigned long)(im_width * osr); 
  dims_g[2] = (unsigned long)(im_width * osr);
  dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];
  gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 8;
	
	const int sector_count = 64;
	//int* sectors = (int*) calloc(sector_count+1,sizeof(int));
	//extracted from matlab
	int sectors[sector_count+1] = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5};

	//int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	int sector_cnt = 0;
	
	int sector_centers[3*sector_count] = {4,4,4,4,4,12,4,4,20,4,4,28,4,12,4,4,12,12,4,12,20,4,12,28,4,20,4,4,20,12,4,20,20,4,20,28,4,28,4,4,28,12,4,28,20,4,28,28,12,4,4,12,4,12,12,4,20,12,4,28,12,12,4,12,12,12,12,12,20,12,12,28,12,20,4,12,20,12,12,20,20,12,20,28,12,28,4,12,28,12,12,28,20,12,28,28,20,4,4,20,4,12,20,4,20,20,4,28,20,12,4,20,12,12,20,12,20,20,12,28,20,20,4,20,20,12,20,20,20,20,20,28,20,28,4,20,28,12,20,28,20,20,28,28,28,4,4,28,4,12,28,4,20,28,4,28,28,12,4,28,12,12,28,12,20,28,12,28,28,20,4,28,20,12,28,20,20,28,20,28,28,28,4,28,28,12,28,28,20,28,28,28};

    gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

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
	free(kern);
	//free(sectors);
	//free(sector_centers);
}


TEST(TestGPUGriddingConv,MatlabTest_8SK3w32)
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
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	int coord_cnt = 0;

	coords[coord_cnt++] = 0.2500f;
	coords[coord_cnt++] = -0.4330f;
	coords[coord_cnt++] = 0;
	
	//Output Grid
	CufftType* gdata;
	unsigned long dims_g[4];
	dims_g[0] = 1; // complex
	dims_g[1] = (unsigned long)(im_width * osr); 
	dims_g[2] = (unsigned long)(im_width * osr);
	dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];
  gdata = (CufftType*) calloc(grid_size,sizeof(CufftType));
	
	//sectors of data, count and start indices
	int sector_width = 8;
	
	const int sector_count = 64;
	//int* sectors = (int*) calloc(sector_count+1,sizeof(int));
	//extracted from matlab
	int sectors[sector_count+1] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

	//int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	int sector_cnt = 0;
	
	int sector_centers[3*sector_count] = {4,4,4,4,4,12,4,4,20,4,4,28,4,12,4,4,12,12,4,12,20,4,12,28,4,20,4,4,20,12,4,20,20,4,20,28,4,28,4,4,28,12,4,28,20,4,28,28,12,4,4,12,4,12,12,4,20,12,4,28,12,12,4,12,12,12,12,12,20,12,12,28,12,20,4,12,20,12,12,20,20,12,20,28,12,28,4,12,28,12,12,28,20,12,28,28,20,4,4,20,4,12,20,4,20,20,4,28,20,12,4,20,12,12,20,12,20,20,12,28,20,20,4,20,20,12,20,20,20,20,20,28,20,28,4,20,28,12,20,28,20,20,28,28,28,4,4,28,4,12,28,4,20,28,4,28,28,12,4,28,12,12,28,12,20,28,12,28,28,20,4,28,20,12,28,20,20,28,20,28,28,28,4,28,28,12,28,28,20,28,28,28};

    gridding3D_gpu_adj(data,data_entries,1,coords,&gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,false,NULL,CONVOLUTION);

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

	EXPECT_NEAR(gdata[get3DC2lin(23,3,16,im_width)].x,0.0012f,epsilon);
	EXPECT_NEAR(gdata[get3DC2lin(23,2,16,im_width)].x,0.0020f,epsilon);
	EXPECT_NEAR(gdata[get3DC2lin(23,1,16,im_width)].x,0.0007f,epsilon);
	EXPECT_NEAR(gdata[get3DC2lin(24,3,16,im_width)].x,0.0026f,epsilon);//Re
	EXPECT_NEAR(gdata[get3DC2lin(24,3,16,im_width)].y,-0.0012f,epsilon);//Im
	EXPECT_NEAR(gdata[get3DC2lin(24,2,16,im_width)].x,0.0045f,epsilon);
	EXPECT_NEAR(gdata[get3DC2lin(24,1,16,im_width)].x,0.0016f,epsilon);
	EXPECT_NEAR(gdata[get3DC2lin(25,3,16,im_width)].x,0.0012f,epsilon);
	EXPECT_NEAR(gdata[get3DC2lin(25,2,16,im_width)].x,0.0020f,epsilon);//Re
	EXPECT_NEAR(gdata[get3DC2lin(25,2,16,im_width)].y,-0.0009f,epsilon);//Im
	EXPECT_NEAR(gdata[get3DC2lin(25,1,16,im_width)].x,0.0007f,epsilon);
	
	free(data);
	free(coords);
	free(gdata);
	free(kern);
	//free(sectors);
	//free(sector_centers);
}
