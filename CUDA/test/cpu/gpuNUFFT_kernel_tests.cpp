#include <limits.h>
#include "gpuNUFFT_cpu.hpp"

#include "gtest/gtest.h"

#define epsilon 0.0001f

TEST(TestKernel, LoadKernel) {
	if (DEBUG)
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