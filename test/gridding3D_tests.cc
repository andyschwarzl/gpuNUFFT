
#include <limits.h>
#include "griddingFunctions.h"
#include "gtest/gtest.h"

TEST(LoadGrid3KernelTest, LoadKernel) {
	printf("start creating kernel...\n");
	
	int kernel_entries = DEFAULT_KERNEL_TABLE_SIZE;
	EXPECT_EQ(kernel_entries,800);
	float epsilon = 0.0001;

	float *kern = (float*) calloc(kernel_entries,sizeof(float));
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

TEST(TestGridding,CPUTest)
{
	int kernel_width = 5;
	int kernel_entries = DEFAULT_KERNEL_TABLE_SIZE;
	
	float *kern = (float*) calloc(kernel_entries,sizeof(float));
	loadGrid3Kernel(kern,kernel_entries);

	//Image
	int im_width = 11;

	//Data
	int data_entries = 3;
    float* data = (float*) calloc(2*data_entries,sizeof(float)); //2* re + im
	data[0] = 1;
	data[1] = 1;
	
	data[2] = 1;
	data[3] = 1;

	data[4] = 1;
	data[5] = 1;

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
	//int coord_entries = 1;
    float* coords = (float*) calloc(3*data_entries,sizeof(float));//3* x,y,z
	coords[0] = 0.2272; //should return in 8,8,8 center
	coords[1] = 0.2272;
	coords[2] = 0.2272;

	coords[3] = -0.318181; //should return in 2,2,2 center
	coords[4] = -0.318181;
	coords[5] = -0.318181;
	
	coords[6] = -0.318181; //should return in 2,8,8 center
	coords[7] = 0.2272;
	coords[8] =0.2272;

	//Output Grid
    float* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = im_width * OVERSAMPLING_RATIO; 
    dims_g[2] = im_width * OVERSAMPLING_RATIO;
    dims_g[3] = im_width * OVERSAMPLING_RATIO;

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (float*) calloc(grid_size,sizeof(float));
	
	//sectors of data, count and start indices
	//e.g. 1000 sectors (50x50x50 image with
	//5x5x5 sector size -> 10x10x10 padded
	//
	int sector_count = 3;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=1;

	sectors[2]=2;
	sectors[3]=3;

	sectors[4]=4;
	sectors[5]=5;


	int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	sector_centers[0] = 8;
	sector_centers[1] = 8;
	sector_centers[2] = 8;

	sector_centers[3] = 2;
	sector_centers[4] = 2;
	sector_centers[5] = 2;

	sector_centers[6] = 2;
	sector_centers[7] = 8;
	sector_centers[8] = 8;

	gridding3D(data,coords,gdata,kern,sectors,sector_count,sector_centers, KERNEL_WIDTH, kernel_entries,dims_g[1]);

	for (int i=0; i<dims_g[1]; i++)
		for (int j=0; j<dims_g[2];j++)
			for (int k=0; k<dims_g[3];k++)
				printf("data[%d,%d,%d]=%f + i %f\n",i,j,k,gdata[2*(i + im_width * (j + k*im_width))],gdata[2*(i + im_width * (j + k*im_width))+1]);

	free(data);
	free(coords);
	free(gdata);
	free(kern);
	free(sectors);
	free(sector_centers);
	
	EXPECT_EQ(1, 1);
}


