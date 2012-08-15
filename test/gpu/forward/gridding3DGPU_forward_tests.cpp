#include <limits.h>

#include "gridding_gpu.hpp"
#include "gridding_kernels.hpp"

#include "gtest/gtest.h"

#define epsilon 0.0001f

#define get3DC2lin(_x,_y,_z,_width) ((_x) + (_width) * ( (_y) + (_z) * (_width)))

TEST(TestGPUGriddingForwardConv,KernelCall1Sector)
{
	int kernel_width = 5;

	//oversampling ratio
	float osr = 1.25f;

	long kernel_entries = calculateGrid3KernelSize(osr,kernel_width/2.0f);
	DType *kern = (DType*) calloc(kernel_entries,sizeof(DType));
	loadGrid3Kernel(kern,kernel_entries);

	//Image
	int im_width = 32;

	//Data
	int data_entries = 1;
    CufftType* data = (CufftType*) calloc(data_entries,sizeof(CufftType)); //2* re + im

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	coords[0] = -0.31719f; //should result in 7,7,7 center
	coords[1] = -0.38650f;
	coords[2] = 0;

	DType* im_data;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width); 
    dims_g[2] = (unsigned long)(im_width);
    dims_g[3] = (unsigned long)(im_width);

	long im_size = dims_g[1]*dims_g[2]*dims_g[3];

	im_data = (DType*) calloc(2*im_size,sizeof(DType));
	
	for (int x=0;x<im_size;x++)
		im_data[x] = 1.0f;
	
	long grid_width = (unsigned long)(im_width * osr);

	//sectors of data, count and start indices
	int sector_width = 8;
	
	const int sector_count = 125;
	int* sectors = (int*) calloc(2*sector_count,sizeof(int));
	sectors[0]=0;
	sectors[1]=0;
	sectors[2]=1;

	/*int* sector_centers = (int*) calloc(3*sector_count,sizeof(int));
	sector_centers[0] = 5;
	sector_centers[1] = 5;
	sector_centers[2] = 5;*/
	int sector_centers[3 * sector_count] = {4,4,4,4,4,12,4,4,20,4,4,28,4,4,36,4,12,4,4,12,12,4,12,20,4,12,28,4,12,36,4,20,4,4,20,12,4,20,20,4,20,28,4,20,36,4,28,4,4,28,12,4,28,20,4,28,28,4,28,36,4,36,4,4,36,12,4,36,20,4,36,28,4,36,36,12,4,4,12,4,12,12,4,20,12,4,28,12,4,36,12,12,4,12,12,12,12,12,20,12,12,28,12,12,36,12,20,4,12,20,12,12,20,20,12,20,28,12,20,36,12,28,4,12,28,12,12,28,20,12,28,28,12,28,36,12,36,4,12,36,12,12,36,20,12,36,28,12,36,36,20,4,4,20,4,12,20,4,20,20,4,28,20,4,36,20,12,4,20,12,12,20,12,20,20,12,28,20,12,36,20,20,4,20,20,12,20,20,20,20,20,28,20,20,36,20,28,4,20,28,12,20,28,20,20,28,28,20,28,36,20,36,4,20,36,12,20,36,20,20,36,28,20,36,36,28,4,4,28,4,12,28,4,20,28,4,28,28,4,36,28,12,4,28,12,12,28,12,20,28,12,28,28,12,36,28,20,4,28,20,12,28,20,20,28,20,28,28,20,36,28,28,4,28,28,12,28,28,20,28,28,28,28,28,36,28,36,4,28,36,12,28,36,20,28,36,28,28,36,36,36,4,4,36,4,12,36,4,20,36,4,28,36,4,36,36,12,4,36,12,12,36,12,20,36,12,28,36,12,36,36,20,4,36,20,12,36,20,20,36,20,28,36,20,36,36,28,4,36,28,12,36,28,20,36,28,28,36,28,36,36,36,4,36,36,12,36,36,20,36,36,28,36,36,36};

	gridding3D_gpu(&data,data_entries,1,coords,im_data,im_size,grid_width,kern,kernel_entries, kernel_width,sectors,sector_count,sector_centers,sector_width, im_width,osr,CONVOLUTION);
	
	for (int j=0; j<data_entries; j++)
	{
		printf("%.4f ",data[j].x);
	}

	printf("\n");

	free(data);
	free(coords);
	free(im_data);
	free(kern);
	free(sectors);
	//free(sector_centers);

	EXPECT_EQ(1, 1);
}
