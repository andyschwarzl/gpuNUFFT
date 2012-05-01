#include <limits.h>

#include "gridding_gpu.hpp"

#include "gtest/gtest.h"

#define epsilon 0.0001f

#define get3DC2lin(_x,_y,_z,_width) ((_x) + (_width) * ( (_y) + (_z) * (_width)))

TEST(TestGPUGriddingDeapo,KernelCall1Sector)
{
	int kernel_width = 5;
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

	gridding3D_gpu(data,data_entries,coords,gdata,grid_size,kern,kernel_entries,sectors,sector_count,sector_centers,sector_width, kernel_width, kernel_entries,dims_g[1],osr,DEAPODIZATION);
	
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

