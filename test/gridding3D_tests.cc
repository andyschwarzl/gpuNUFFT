
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
	int im_width = 3;

	//Data
	int data_entries = 1;
    float* data = (float*) calloc(2*data_entries,sizeof(float)); //2* re + im
	data[0] = 1;
	data[1] = 1;

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
	int coord_entries = 1;
    float* coords = (float*) calloc(3*coord_entries,sizeof(float));//3* x,y,z
	coords[0] = 0;
	coords[1] = 0;
	coords[2] = 0;

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
	int sector_count = 1;
	int* sectors = (int*) calloc(2,sizeof(int));
	sectors[0]=0;
	sectors[1]=1;

	int* sector_centers = (int*) calloc(3,sizeof(int));
	sector_centers[0] = 1;
	sector_centers[1] = 1;
	sector_centers[2] = 1;

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


