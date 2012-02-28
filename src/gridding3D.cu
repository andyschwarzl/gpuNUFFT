
#include <cuda.h>

#include <cuda_utils.h>

#include <cufft.h>

#include "griddingFunctions.h"

int main()
{
	printf("starting...\n");
	
	/* allocate kernel table */
    printf("allocate kernel\n");

	int im_width = 50;

	int kernel_width = 5;
	int kernel_entries = calculateGrid3KernelSize(DEFAULT_OVERSAMPLING_RATIO,kernel_width/2.0f);
	
	float *kern = (float*) calloc(kernel_entries,sizeof(float));
	loadGrid3Kernel(kern,kernel_entries);
	
	//cufftComplex *data, *output;
	
	/* DATA */
    float* data;

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
    float* coords;

	//Output Grid
    float* gdata;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = im_width * DEFAULT_OVERSAMPLING_RATIO; 
    dims_g[2] = im_width * DEFAULT_OVERSAMPLING_RATIO;
    dims_g[3] = im_width * DEFAULT_OVERSAMPLING_RATIO;

	long grid_size = dims_g[0]*dims_g[1]*dims_g[2]*dims_g[3];

    gdata = (float*) calloc(grid_size,sizeof(float));
	
	//sectors of data, count and start indices
	//e.g. 1000 sectors (50x50x50 image with
	//5x5x5 sector size -> 10x10x10 padded
	//
	int sector_count;
	int* sectors;
	int* sector_centers;
	int sector_width = 5;
	gridding3D(data,coords,gdata,kern,sectors,sector_count,sector_centers,sector_width, DEFAULT_KERNEL_WIDTH, kernel_entries,3);

	//HANDLE_ERROR(cudaMalloc(&a_d,N*sizeof(int)));

	//HANDLE_ERROR(cudaMemcpy( a_d, a_h, N* sizeof(int),
///                              cudaMemcpyHostToDevice ) );
	//free(c_h);
	//cudaFree(&c_d);
	
	//Wait for user input
	getchar();

	return 0;
}
