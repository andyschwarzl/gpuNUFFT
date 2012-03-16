#include <limits.h>

#include "gridding_gpu.hpp"

#include "gtest/gtest.h"

#define epsilon 0.0001f

#define get3DC2lin(_x,_y,_z,_width) 2*((_x) + (_width) * ( (_y) + (_z) * (_width)))

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

TEST(TestGPULib,KernelCall1Sector)
{
	int kernel_width = 3;
	long kernel_entries = calculateGrid3KernelSize();
	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries);

	//Image
	int im_width = 10;

	//Data
	int data_entries = 1;
    DType* data = (DType*) calloc(2*data_entries,sizeof(DType)); //2* re + im
	data[0] = 1;
	data[1] = 1;

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
    DType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (DType*) calloc(grid_size,sizeof(DType));
	
	//sectors of data, count and start indices
	int sector_width = 10;
	
	int sector_count = 1;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=1;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	sector_centers[0] = 5;
	sector_centers[1] = 5;
	sector_centers[2] = 5;

	gridding3D_gpu(data,data_entries,coords,gdata,grid_size,kern,kernel_entries,sectors,sector_count,sector_centers,sector_width, kernel_width, kernel_entries,dims_g[1]);

	printf("test %f \n",gdata[4]);
	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.0f,gdata[index],epsilon);
	EXPECT_NEAR(0.4502,gdata[get3DC2lin(5,4,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get3DC2lin(4,5,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get3DC2lin(5,6,5,im_width)],epsilon*10.0f);

	EXPECT_NEAR(0.2027,gdata[get3DC2lin(6,6,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get3DC2lin(4,4,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get3DC2lin(4,6,5,im_width)],epsilon*10.0f);
	
	/*for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			printf("%.4f ",gdata[get3DC2lin(i,j,5,im_width)]);
		printf("\n");
	}*/

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);

	EXPECT_EQ(1, 1);
}

TEST(TestGPULib,GPUTest_1SectorKernel5)
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
    DType* data = (DType*) calloc(2*data_entries,sizeof(DType)); //2* re + im
	data[0] = 1;
	data[1] = 1;

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	coords[0] = 0; //should result in 7,7,7 center
	coords[1] = 0;
	coords[2] = 0;

	//Output Grid
    DType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (DType*) calloc(grid_size,sizeof(DType));
	
	//sectors of data, count and start indices
	int sector_width = 8;
	
	int sector_count = 1;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=1;

	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	sector_centers[0] = 5;
	sector_centers[1] = 5;
	sector_centers[2] = 5;

	gridding3D_gpu(data,data_entries,coords,gdata,grid_size,kern,kernel_entries,sectors,sector_count,sector_centers,sector_width, kernel_width, kernel_entries,dims_g[1]);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.0f,gdata[index],epsilon);
	EXPECT_NEAR(0.0049,gdata[get3DC2lin(3,3,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.3218,gdata[get3DC2lin(4,4,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.5673,gdata[get3DC2lin(5,4,5,im_width)],epsilon*10.0f);

	EXPECT_NEAR(0.0697,gdata[get3DC2lin(5,7,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.0697,gdata[get3DC2lin(5,3,5,im_width)],epsilon*10.0f);
	
	/*for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			printf("%.4f ",gdata[get3DC2lin(i,im_width-j,5,im_width)]);
		printf("\n");
	}*/

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
}



TEST(TestGPULib,GPUTest_2SectorsKernel3nData)
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
    DType* data = (DType*) calloc(2*data_entries,sizeof(DType)); //2* re + im
	int data_cnt = 0;
	data[data_cnt++] = 0.5f;
	data[data_cnt++] = 0.5f;
	
	data[data_cnt++] = 0.7f;
	data[data_cnt++] = 1;
	
	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	int coord_cnt = 0;
	//1.Sektor
	coords[coord_cnt++] = -0.3f; 
	coords[coord_cnt++] = 0.2f;
	coords[coord_cnt++] = 0;

	coords[coord_cnt++] = -0.1f;
	coords[coord_cnt++] = 0;
	coords[coord_cnt++] = 0;

	//2.Sektor
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
    DType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (DType*) calloc(grid_size,sizeof(DType));
	
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

	gridding3D_gpu(data,data_entries,coords,gdata,grid_size,kern,kernel_entries,sectors,sector_count,sector_centers,sector_width, kernel_width, kernel_entries,dims_g[1]);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.3152f,gdata[index],epsilon);
	EXPECT_NEAR(0.2432,gdata[get3DC2lin(3,6,5,im_width)],epsilon*10.0f);
	
	EXPECT_NEAR(0.2251,gdata[get3DC2lin(1,7,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get3DC2lin(6,5,5,im_width)],epsilon*10.0f);

	EXPECT_NEAR(1.0f,gdata[get3DC2lin(8,8,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get3DC2lin(9,9,5,im_width)],epsilon*10.0f);
	
	/*for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			printf("%.4f ",gdata[get3DC2lin(i,im_width-1-j,5,im_width)]);
		printf("\n");
	}*/

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
}


TEST(TestGridding,CPUTest_8SectorsKernel3nData)
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
    DType* data = (DType*) calloc(2*data_entries,sizeof(DType)); //2* re + im
	int data_cnt = 0;
	data[data_cnt++] = 0.5f;
	data[data_cnt++] = 0.5f;
	
	data[data_cnt++] = 0.7f;
	data[data_cnt++] = 1;
	
	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

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
    DType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (DType*) calloc(grid_size,sizeof(DType));
	
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

	gridding3D_gpu(data,data_entries,coords,gdata,grid_size,kern,kernel_entries,sectors,sector_count,sector_centers,sector_width, kernel_width, kernel_entries,dims_g[1]);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.3152f,gdata[index],epsilon);
	EXPECT_NEAR(0.2432,gdata[get3DC2lin(3,6,5,im_width)],epsilon*10.0f);
	
	EXPECT_NEAR(0.2251,gdata[get3DC2lin(1,7,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.4502,gdata[get3DC2lin(6,5,5,im_width)],epsilon*10.0f);

	EXPECT_NEAR(1.0f,gdata[get3DC2lin(8,8,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.2027,gdata[get3DC2lin(9,9,5,im_width)],epsilon*10.0f);

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
}


TEST(TestGridding,CPUTest_8SectorsKernel4nData)
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
    DType* data = (DType*) calloc(2*data_entries,sizeof(DType)); //2* re + im
	int data_cnt = 0;
	data[data_cnt++] = 0.5f;
	data[data_cnt++] = 0.5f;
	
	data[data_cnt++] = 0.7f;
	data[data_cnt++] = 1;
	
	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

	data[data_cnt++] = 1;
	data[data_cnt++] = 1;

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
    DType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (DType*) calloc(grid_size,sizeof(DType));
	
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

	gridding3D_gpu(data,data_entries,coords,gdata,grid_size,kern,kernel_entries,sectors,sector_count,sector_centers,sector_width, kernel_width, kernel_entries,dims_g[1]);

	int index = get3DC2lin(5,5,5,im_width);
	printf("index to test %d\n",index);
	//EXPECT_EQ(index,2*555);
	EXPECT_NEAR(1.3558f,gdata[index],epsilon);
	EXPECT_NEAR(0.3101f,gdata[get3DC2lin(3,6,5,im_width)],epsilon*10.0f);
	
	EXPECT_NEAR(0.2542f,gdata[get3DC2lin(1,7,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.5084f,gdata[get3DC2lin(6,5,5,im_width)],epsilon*10.0f);

	EXPECT_NEAR(1.0f,gdata[get3DC2lin(8,8,5,im_width)],epsilon*10.0f);
	EXPECT_NEAR(0.2585f,gdata[get3DC2lin(9,9,5,im_width)],epsilon*10.0f);
	
	/*for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			printf("%.4f ",gdata[get3DC2lin(i,im_width-1-j,5,im_width)]);
		printf("\n");
	}*/

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
}