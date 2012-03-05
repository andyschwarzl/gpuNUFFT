#include "cuda.h"
#include "cuda_runtime.h"
#include "gridding_gpu.hpp"

__global__ void kernel_call(int *a)
{
    int tx = threadIdx.x;
    
    switch( tx % 2 )
    {
        case 0:
     a[tx] = a[tx] + 2;
     break;
        case 1:
     a[tx] = a[tx] + 3;
     break;
    }
}

void gridding3D_gpu(float* data, 
				float* crds, 
				float* gdata,
				float* kernel, 
				int* sectors, 
				int sector_count, 
				int* sector_centers,
				int sector_width,
				int kernel_width, 
				int kernel_count, 
				int width)
{
	printf("starting gpu implementation\n");
	int test_h[3];
	test_h[0] = 1;
	test_h[1] = 2;
	test_h[2] = 3;
	
	int* test_d;

	printf("input: %d , %d , %d\n",test_h[0],test_h[1],test_h[2]);
	cudaMalloc(&test_d,3*sizeof(int));
	
	cudaMemcpy(test_d, test_h,3*sizeof(int),cudaMemcpyHostToDevice );
	dim3 grid_size = 1;
	dim3 thread_size = 3;

	kernel_call<<<grid_size,thread_size>>>(test_d);

	cudaMemcpy(test_h, test_d, 3*sizeof(int),cudaMemcpyDeviceToHost);
	
	printf("output: %d , %d , %d\n",test_h[0],test_h[1],test_h[2]);

	cudaFree(&test_h);
}