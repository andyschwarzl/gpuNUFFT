#ifndef STD_GRIDDING_KERNELS_CU
#define STD_GRIDDING_KERNELS_CU

#include "gridding_kernels.hpp"
#include "cuda_utils.hpp"

#include <string>

#include "cuda_utils.cuh"

// Method to initialize CONSTANT memory symbols. Needs to reside in *.cu file 
// to work properly
//
// TODO find better solution
//
void initConstSymbol(const char* symbol, const void* src, IndType size)
{
	if (std::string("GI").compare(symbol)==0)
		HANDLE_ERROR(cudaMemcpyToSymbol(GI, src,size));

	if (std::string("KERNEL").compare(symbol)==0)
		HANDLE_ERROR(cudaMemcpyToSymbol(KERNEL, src,size));
}

__global__ void fftScaleKernel(CufftType* data, DType scaling, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;

	while (t < N) 
	{
		CufftType data_p = data[t]; 
		data_p.x = data_p.x * scaling;
		data_p.y = data_p.y * scaling;
		data[t] = data_p;
		t = t+ blockDim.x*gridDim.x;
	}
}

void performFFTScaling(CufftType* data,int N, GriddingND::GriddingInfo* gi_host)
{
	dim3 block_dim(THREAD_BLOCK_SIZE);
	dim3 grid_dim(getOptimalGridDim(N,THREAD_BLOCK_SIZE));
	DType scaling_factor = (DType)1.0 / (DType) sqrt((DType)gi_host->im_width_dim);

	fftScaleKernel<<<grid_dim,block_dim>>>(data,scaling_factor,N);
}

__global__ void densityCompensationKernel(DType2* data, DType* density_comp, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;

	while (t < N) 
	{
		DType2 data_p = data[t]; 
		data_p.x = data_p.x * density_comp[t];
		data_p.y = data_p.y * density_comp[t];
		data[t] = data_p;
		t = t+ blockDim.x*gridDim.x;
	}
}

void performDensityCompensation(DType2* data, DType* density_comp, GriddingND::GriddingInfo* gi_host)
{
	dim3 block_dim(THREAD_BLOCK_SIZE);
	dim3 grid_dim(getOptimalGridDim(gi_host->data_count,THREAD_BLOCK_SIZE));
	densityCompensationKernel<<<grid_dim,block_dim>>>(data,density_comp,gi_host->data_count);
}

__global__ void deapodizationKernel(CufftType* gdata, DType beta, DType norm_val, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;

	int x, y, z;
	DType deapo;
	while (t < N) 
	{ 
	   getCoordsFromIndex(t, &x, &y, &z, GI.im_width);
	   
	   deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
	   //check if deapodization value is valid number
	   if (!isnan(deapo))
	   {
			 CufftType gdata_p = gdata[t]; 
		   gdata_p.x = gdata_p.x / deapo;//Re
		   gdata_p.y = gdata_p.y / deapo;//Im
			 gdata[t] = gdata_p;
	   }
	   t = t + blockDim.x*gridDim.x;
	}
}

__global__ void precomputeDeapodizationKernel(DType* deapo_d, DType beta, DType norm_val, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;
	int x, y, z;
	DType deapo;
	while (t < N) 
	{ 
	   getCoordsFromIndex(t, &x, &y, &z, GI.im_width);
	   
	   deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
	   //check if deapodization value is valid number
	   if (!isnan(deapo))// == deapo)
	   {
			 deapo_d[t] = (DType)1.0 / deapo;
	   }
		 else
			 deapo_d[t] = (DType)1.0;

	   t = t + blockDim.x*gridDim.x;
	}
}

__global__ void cropKernel(CufftType* gdata,CufftType* imdata, int offset, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;
	int x, y, z, grid_ind;
	while (t < N) 
	{
		getCoordsFromIndex(t, &x, &y, &z, GI.im_width);
		grid_ind = getIndex(offset+x,offset+y,offset+z,GI.grid_width);
		imdata[t] = gdata[grid_ind];
		t = t + blockDim.x*gridDim.x;
	}
}

__global__ void fftShiftKernel(CufftType* gdata, int offset, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;
	int x, y, z, x_opp, y_opp, z_opp, ind_opp;
	while (t < N) 
	{ 
		getCoordsFromIndex(t, &x, &y, &z, GI.grid_width);
		//calculate "opposite" coord pair
		x_opp = (x + offset) % GI.grid_width;
		y_opp = (y + offset) % GI.grid_width;
		z_opp = (z + offset) % GI.grid_width;
		ind_opp = getIndex(x_opp,y_opp,z_opp,GI.grid_width);
		//swap points
		CufftType temp = gdata[t];
		gdata[t] = gdata[ind_opp];
		gdata[ind_opp] = temp;
		
		t = t + blockDim.x*gridDim.x;
	}
}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performDeapodization(CufftType* imdata_d,
													GriddingND::GriddingInfo* gi_host)
{
	DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

	//Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
	DType norm_val = calculateDeapodizationValue(0,gi_host->grid_width_inv,gi_host->kernel_width,beta);
	norm_val = norm_val * norm_val * norm_val;
	if (DEBUG)
		printf("running deapodization with norm_val %.2f\n",norm_val);

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	deapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
}

__global__ void precomputedDeapodizationKernel(CufftType* imdata, DType* deapo, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;

	while (t < N) 
	{
		CufftType data_p = imdata[t]; 
		data_p.x = data_p.x * deapo[t];
		data_p.y = data_p.y * deapo[t];
		imdata[t] = data_p;
		t = t+ blockDim.x*gridDim.x;
	}
}

void performDeapodization(CufftType* imdata_d,
													DType* deapo_d,
													GriddingND::GriddingInfo* gi_host)
{
	if (DEBUG)
		printf("running deapodization with precomputed values\n");

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	precomputedDeapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,deapo_d,gi_host->im_width_dim);
}

void precomputeDeapodization(DType* deapo_d,
														 GriddingND::GriddingInfo* gi_host)
{
	DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

	//Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
	DType norm_val = calculateDeapodizationValue(0,gi_host->grid_width_inv,gi_host->kernel_width,beta);
	norm_val = norm_val * norm_val * norm_val;
	if (DEBUG)
		printf("running deapodization precomputation with norm_val %.2f\n",norm_val);

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	precomputeDeapodizationKernel<<<grid_dim,block_dim>>>(deapo_d,beta,norm_val,gi_host->im_width_dim);
}

void performCrop(CufftType* gdata_d,
				 CufftType* imdata_d,
				 GriddingND::GriddingInfo* gi_host)
{
	int ind_off = (int)(gi_host->im_width * ((DType)gi_host->osr - 1.0f)/(DType)2);
	if (DEBUG)
		printf("start cropping image with offset %d\n",ind_off);

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);

	cropKernel<<<grid_dim,block_dim>>>(gdata_d,imdata_d,ind_off,gi_host->im_width_dim);
}

void performFFTShift(CufftType* gdata_d,
					 GriddingND::FFTShiftDir shift_dir,
					 int width)
{
	int offset= 0;
	int t_width = 0;
	if (shift_dir == GriddingND::FORWARD)
	{
		offset = (int)ceil((DType)(width / (DType)2.0));
		if (width % 2)
			t_width = offset - 1;
		else 
			t_width = offset;
	}
	else
	{
		offset = (int)floor((DType)(width / (DType)2.0));
		t_width = offset;
	}

	dim3 grid_dim(getOptimalGridDim(width*width,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);

	fftShiftKernel<<<grid_dim,block_dim>>>(gdata_d,offset,width*width*t_width);
}

__global__ void forwardDeapodizationKernel(DType2* imdata, DType beta, DType norm_val, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;

	int x, y, z;
	DType deapo;
	while (t < N) 
	{ 
	   getCoordsFromIndex(t, &x, &y, &z, GI.im_width);
	   
	   deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
	   //check if deapodization value is valid number
	   if (!isnan(deapo))// == deapo)
	   {
		   imdata[t].x = imdata[t].x / deapo;//Re
		   imdata[t].y = imdata[t].y / deapo ; //Im
	   }
	   t = t + blockDim.x*gridDim.x;
	}
}

__global__ void paddingKernel(DType2* imdata,CufftType* gdata, int offset,int N)
{	
	int t = threadIdx.x +  blockIdx.x *blockDim.x;

	int x, y, z,grid_ind;
	while (t < N) 
	{ 
		getCoordsFromIndex(t, &x, &y, &z, GI.im_width);
		grid_ind =  getIndex(offset + x,offset + y,offset +z,GI.grid_width);
		
		gdata[grid_ind].x =  imdata[t].x;
		gdata[grid_ind].y = imdata[t].y;
		t = t+ blockDim.x*gridDim.x;
	}
}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performForwardDeapodization(DType2* imdata_d,
						  GriddingND::GriddingInfo* gi_host)
{
	DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	
	//Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
	DType norm_val = calculateDeapodizationValue(0,gi_host->grid_width_inv,gi_host->kernel_width,beta);
	norm_val = norm_val * norm_val * norm_val;

	forwardDeapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
}

void performForwardDeapodization(DType2* imdata_d,
																DType* deapo_d,
																GriddingND::GriddingInfo* gi_host)
{
	if (DEBUG)
		printf("running forward deapodization with precomputed values\n");

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	precomputedDeapodizationKernel<<<grid_dim,block_dim>>>((CufftType*)imdata_d,deapo_d,gi_host->im_width_dim);
}

void performPadding(DType2* imdata_d,
					CufftType* gdata_d,					
					GriddingND::GriddingInfo* gi_host)
{
	int ind_off = (int)(gi_host->im_width * ((DType)gi_host->osr -1.0f)/(DType)2);
	if (DEBUG)
		printf("start padding image with offset %d\n",ind_off);

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	paddingKernel<<<grid_dim,block_dim>>>(imdata_d,gdata_d,ind_off,gi_host->im_width_dim);
}

#endif //STD_GRIDDING_KERNELS_CU