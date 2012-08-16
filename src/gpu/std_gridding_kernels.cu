#include "gridding_kernels.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"

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
	   if (!isnan(deapo))// == deapo)
	   {
		   gdata[t].x = gdata[t].x / deapo;//Re
		   gdata[t].y = gdata[t].y / deapo;//Im
	   }
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
		//TODO shared?
		CufftType temp = gdata[t];
		gdata[t] = gdata[ind_opp];
		gdata[ind_opp] = temp;
		
		t = t + blockDim.x*gridDim.x;
	}
}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performDeapodization(CufftType* imdata_d,
						  GriddingInfo* gi_host)
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


void performCrop(CufftType* gdata_d,
				 CufftType* imdata_d,
				 GriddingInfo* gi_host)
{
	int ind_off = (int)(gi_host->im_width * ((DType)gi_host->osr - 1.0f)/(DType)2);
	if (DEBUG)
		printf("start cropping image with offset %d\n",ind_off);

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);

	cropKernel<<<grid_dim,block_dim>>>(gdata_d,imdata_d,ind_off,gi_host->im_width_dim);
}

void performFFTShift(CufftType* gdata_d,
					 FFTShiftDir shift_dir,
					 int width)
{
	dim3 grid_dim(getOptimalGridDim(width*width,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	int offset= 0;

	if (shift_dir == FORWARD)
	{
		offset = (int)ceil((DType)(width / (DType)2.0));
	}
	else
	{
		offset = (int)floor((DType)(width / (DType)2.0));
	}
	fftShiftKernel<<<grid_dim,block_dim>>>(gdata_d,offset,width*width*width/2);
}

__global__ void forwardDeapodizationKernel(DType* imdata, DType beta, DType norm_val, int N)
{
	int t = threadIdx.x +  blockIdx.x *blockDim.x;

	int x, y, z,ind;
	DType deapo;
	while (t < N) 
	{ 
	   getCoordsFromIndex(t, &x, &y, &z, GI.im_width);
	   
	   deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
	   //check if deapodization value is valid number
	   if (!isnan(deapo))// == deapo)
	   {
		   ind = 2*t;
		   imdata[ind] = imdata[ind] / deapo;//Re
		   imdata[ind+1] = imdata[ind+1] / deapo ; //Im
	   }
	   t = t + blockDim.x*gridDim.x;
	}
}

__global__ void paddingKernel(DType* imdata,CufftType* gdata, int offset,int N)
{	
	int t = threadIdx.x +  blockIdx.x *blockDim.x;

	int x, y, z, im_ind,grid_ind;
	while (t < N) 
	{ 
		getCoordsFromIndex(t, &x, &y, &z, GI.im_width);
		grid_ind =  getIndex(offset + x,offset + y,offset +z,GI.grid_width);
		im_ind = 2*t;
		gdata[grid_ind].x = imdata[im_ind];
		gdata[grid_ind].y = imdata[im_ind+1];
		t = t+ blockDim.x*gridDim.x;
	}
}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performForwardDeapodization(DType* imdata_d,
						  GriddingInfo* gi_host)
{
	DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	
	//Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
	DType norm_val = calculateDeapodizationValue(0,gi_host->grid_width_inv,gi_host->kernel_width,beta);
	norm_val = norm_val * norm_val * norm_val;

	forwardDeapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val,gi_host->im_width_dim);
}

void performPadding(DType* imdata_d,
					CufftType* gdata_d,					
					GriddingInfo* gi_host)
{
	int ind_off = (int)(gi_host->im_width * ((DType)gi_host->osr -1.0f)/(DType)2);
	if (DEBUG)
		printf("start padding image with offset %d\n",ind_off);

	dim3 grid_dim(getOptimalGridDim(gi_host->im_width_dim,THREAD_BLOCK_SIZE));
	dim3 block_dim(THREAD_BLOCK_SIZE);
	paddingKernel<<<grid_dim,block_dim>>>(imdata_d,gdata_d,ind_off,gi_host->im_width_dim);
}
