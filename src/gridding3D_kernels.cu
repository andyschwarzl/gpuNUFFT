#include <stdio.h>

#include <iostream>

#include <cuda.h>

#define N 5000000

#define HANDLE_ERROR(err) { \
	if (err != cudaSuccess) \
	{ \
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), \
                __FILE__, __LINE__ ); \
        exit( EXIT_FAILURE ); \
	}}\

#include <time.h>

double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
	return diffms;
} 

__global__ void add(int* a, int* b, int* result)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	int id = x + y * gridDim.x + threadIdx.x * (gridDim.x * gridDim.y);
	
	while (id < N)
	{
		if (a[id] > 100)
			result[id] =a[id] + b[id] ;
		else
			result[id] = a[id] + b[id];

		id += blockDim.x * gridDim.x * gridDim.y;
	}

}

void add_cpu(int* a, int* b, int* result)
{
	for (int i=0; i<N;i++)
	{
		if (a[i] > 100)
			result[i] = a[i] + b[i] ;
		else
			result[i] = a[i] + b[i];
	}
}

void generateVectors(int** a_h, int** b_h)
{
	(*a_h)[0] = 3;
	(*b_h)[0] = 3;

	printf("test %d",(*a_h[0]));

	for (int i=1; i < N; i++)
	{
		(*a_h)[i] = (i % 27);//(*a_h)[i-1] + (i % 3);
		(*b_h)[i] = (i % 13) * (i % 39);//(*b_h)[i-1] - (i % 4);
	}

}

int main()
{
	printf("hello test\n");

	int* a_h, *b_h, *c_h;
	a_h = (int*) malloc(N*sizeof(int));
	b_h = (int*) malloc(N*sizeof(int));
	c_h = (int*) malloc(N*sizeof(int));
	
	generateVectors(&a_h,&b_h);

		
	clock_t  begin=clock();

		add_cpu(a_h,b_h,c_h);
	
	clock_t  end = clock();
	printf("Time elapsed at cpu processing: %f ms\n",double(diffclock(end,begin)));
	


	begin=clock();
	
	int* a_d;
	int* b_d;
	int* c_d; 
	HANDLE_ERROR(cudaMalloc(&a_d,N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&b_d,N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&c_d,N*sizeof(int)));

	HANDLE_ERROR( cudaMemcpy( a_d, a_h, N* sizeof(int),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( b_d, b_h, N* sizeof(int),
                              cudaMemcpyHostToDevice ) );
	
	int grid_size = 128;

	dim3 grid(grid_size,grid_size);

	int thread_size = 128;//((N + (grid_size*grid_size - 1)) / (grid_size*grid_size));

	printf("calculated thread size: %d\n" , thread_size);
	
	add<<<grid,thread_size>>>(a_d,b_d,c_d);

	HANDLE_ERROR( cudaMemcpy( c_h, c_d, N* sizeof(int),
                              cudaMemcpyDeviceToHost) );
	end=clock();
	printf("Time elapsed with gpu processing: %f ms\n",double(diffclock(end,begin)));

	for (int i = (N-100); i < N; i++)
	{
		printf("Test %d + %d = %d \n",a_h[i],b_h[i],c_h[i]);
	}

	free(a_h);
	free(b_h);
	free(c_h);
	cudaFree(&a_d);
	cudaFree(&b_d);
	cudaFree(&c_d);

	return 0;
}

