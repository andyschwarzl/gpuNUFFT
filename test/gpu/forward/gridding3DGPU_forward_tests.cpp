#include <limits.h>

#include "gridding_gpu.hpp"
#include "gridding_kernels.hpp"

#include "gtest/gtest.h"

#define epsilon 0.0001f

#define get3DC2lin(_x,_y,_z,_width) ((_x) + (_width) * ( (_y) + (_z) * (_width)))

TEST(TestGPUGriddingForwardConv,KernelCall1Sector)
{
	int kernel_width = 3;
	long kernel_entries = calculateGrid3KernelSize();
	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries);

	//Image
	int im_width = 10;

	//Data
	int data_entries = 1;
    CufftType* data = (CufftType*) calloc(data_entries,sizeof(CufftType)); //2* re + im

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	coords[0] = 0; //should result in 7,7,7 center
	coords[1] = 0;
	coords[2] = 0;

	//oversampling ratio
	float osr = DEFAULT_OVERSAMPLING_RATIO;

	DType* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width * osr); 
    dims_g[2] = (unsigned long)(im_width * osr);
    dims_g[3] = (unsigned long)(im_width * osr);

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

	gdata = (DType*) calloc(grid_size,sizeof(DType));
	
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

	gridding3D_gpu(data,data_entries,1,coords,gdata,grid_size,dims_g[1],kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,CONVOLUTION);
	
	printf("test %f \n",data[4].x);
	int index = get3DC2lin(5,5,5,im_width);
	
	for (int j=0; j<im_width; j++)
	{
		for (int i=0; i<im_width; i++)
			printf("%.4f ",data[get3DC2lin(i,j,5,im_width)].x);
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
