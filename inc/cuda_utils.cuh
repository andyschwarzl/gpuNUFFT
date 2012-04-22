#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

__constant__ GriddingInfo GI;

GriddingInfo* initAndCopyGriddingInfo(int sector_count, 
							 int sector_width,
							 int kernel_width,
							 int kernel_count, 
							 int width)
{
	GriddingInfo* gi_host = (GriddingInfo*)malloc(sizeof(GriddingInfo));

	gi_host->sector_count = sector_count;
	gi_host->sector_width = sector_width;
	
	gi_host->kernel_width = kernel_width; 
	gi_host->kernel_count = kernel_count;
	gi_host->width = width;

	DType kernel_radius = static_cast<DType>(kernel_width) / 2.0f;
	DType radius = kernel_radius / static_cast<DType>(width);
	//DType width_inv = 1.0f / width;
	DType radiusSquared = radius * radius;
	DType kernelRadius_invSqr = 1 / radiusSquared;
	DType dist_multiplier = (kernel_count - 1) * kernelRadius_invSqr;
	printf("radius rel. to grid width %f\n",radius);
	int sector_pad_width = sector_width + 2*(int)(floor(kernel_width / 2.0f));
	int sector_dim = sector_pad_width  * sector_pad_width  * sector_pad_width ;
	int sector_offset = (int)(floor(sector_pad_width / 2.0f));

	gi_host->kernel_radius = kernel_radius;
	gi_host->sector_pad_width = sector_pad_width;
	gi_host->sector_dim = sector_dim;
	gi_host->sector_offset = sector_offset;
	gi_host->radiusSquared = radiusSquared;
	gi_host->dist_multiplier = dist_multiplier;

	printf("sector offset = %d\n",sector_offset);
	
	gi_host->sector_pad_width = sector_pad_width;
	
	printf("copy Gridding Info to symbol memory...\n");
	cudaMemcpyToSymbol(GI, gi_host,sizeof(GriddingInfo));
	//free(gi_host);
	printf("...done!\n");
	return gi_host;
}

#endif