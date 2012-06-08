#include "gridding_kernels.hpp"
#include "cuda_utils.cuh"

__global__ void deapodizationKernel(CufftType* gdata, DType beta, DType norm_val)
{
	int x=blockIdx.x;
	int y=blockIdx.y;
	int z=threadIdx.x;

	int ind = getIndex(x,y,z,GI.im_width);
	
	DType deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
	
	//check if deapodization value is valid number
	if (!isnan(deapo))// == deapo)
	{
		gdata[ind].x = gdata[ind].x / deapo;//Re
		gdata[ind].y = gdata[ind].y / deapo;//Im
	}
}

__global__ void cropKernel(CufftType* gdata,CufftType* imdata, int offset)
{
	int x=blockIdx.x; //[0,N-1] N...im_width
	int y=blockIdx.y; //[0,N-1] N...im_width
	int z=threadIdx.x;//[0,N-1] N...im_width
	int grid_ind = getIndex(offset+x,offset+y,offset+z,GI.grid_width);
	int im_ind = getIndex(x,y,z,GI.im_width);

	imdata[im_ind] = gdata[grid_ind];
}

__global__ void fftShiftKernel(CufftType* gdata, int offset)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;

	//calculate "opposite" coord pair
	int x_opp = (x + offset) % GI.grid_width;
	int y_opp = (y + offset) % GI.grid_width;
	int z_opp = (z + offset) % GI.grid_width;

	//swap points
	CufftType temp = gdata[getIndex(x,y,z,GI.grid_width)];
	gdata[getIndex(x,y,z,GI.grid_width)] = gdata[getIndex(x_opp,y_opp,z_opp,GI.grid_width)];
	gdata[getIndex(x_opp,y_opp,z_opp,GI.grid_width)] = temp;

}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performDeapodization(CufftType* imdata_d,
						  GriddingInfo* gi_host)
{
	DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

	dim3 grid_dim(gi_host->im_width,gi_host->im_width,1);	
	dim3 block_dim(gi_host->im_width);
	//Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
	DType norm_val = calculateDeapodizationValue(0,gi_host->grid_width_inv,gi_host->kernel_width,beta);
	norm_val = norm_val * norm_val * norm_val;

	deapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val);
}


void performCrop(CufftType* gdata_d,
				 CufftType* imdata_d,
				 GriddingInfo* gi_host)
{
	/*crop data 
    ind_off = (a.params.im_width * (double(a.params.osr)-1) / 2) + 1;
    ind_start = ind_off;
    ind_end = ind_start + a.params.im_width -1;
    ress = m(ind_start:ind_end,ind_start:ind_end,ind_start:ind_end,:);*/
	int ind_off = (int)(gi_host->im_width * ((DType)gi_host->osr - 1.0f)/(DType)2);
	printf("start cropping image with offset %d\n",ind_off);

	dim3 grid_dim(gi_host->im_width,gi_host->im_width,1);
	dim3 block_dim(gi_host->im_width);
	cropKernel<<<grid_dim,block_dim>>>(gdata_d,imdata_d,ind_off);
}

void performFFTShift(CufftType* gdata_d,
					 FFTShiftDir shift_dir,
					 int width)
{
	dim3 grid_dim((int)ceil(width/(DType)2.0));
	dim3 block_dim(width,width);
	int offset= 0;

	if (shift_dir == FORWARD)
	{
		offset = (int)ceil((DType)(width / (DType)2.0));
	}
	else
	{
		offset = (int)floor((DType)(width / (DType)2.0));
	}
	fftShiftKernel<<<block_dim,grid_dim>>>(gdata_d,offset);
}

__global__ void forwardDeapodizationKernel(DType* imdata, DType beta, DType norm_val)
{
	int x=blockIdx.x;
	int y=blockIdx.y;
	int z=threadIdx.x;

	int ind = 2*getIndex(x,y,z,GI.im_width);
	
	DType deapo = calculateDeapodizationAt(x,y,z,GI.im_width_offset,GI.grid_width_inv,GI.kernel_width,beta,norm_val);
	
	//TODO reciporcal or not????
	//check if deapodization value is valid number
	if (!isnan(deapo))// == deapo)
	{
		imdata[ind] = imdata[ind] / deapo; // / deapo;//Re
		imdata[ind+1] = imdata[ind+1] / deapo ; /// deapo;//Im
	}
}

__global__ void paddingKernel(DType* imdata,CufftType* gdata, int offset)
{
	int x=blockIdx.x; //[0,N-1] N...im_width
	int y=blockIdx.y; //[0,N-1] N...im_width
	int z=threadIdx.x;//[0,N-1] N...im_width

	int grid_ind =  getIndex(offset + x,offset + y,offset +z,GI.grid_width);

	int im_ind = 2*getIndex(x,y,z,GI.im_width);

	gdata[grid_ind].x = imdata[im_ind];
	gdata[grid_ind].y = imdata[im_ind+1];
}

//see BEATTY et al.: RAPID GRIDDING RECONSTRUCTION
//eq. (4) and (5)
void performForwardDeapodization(DType* imdata_d,
						  GriddingInfo* gi_host)
{
	DType beta = (DType)BETA(gi_host->kernel_width,gi_host->osr);

	dim3 grid_dim(gi_host->im_width,gi_host->im_width,1);	
	dim3 block_dim(gi_host->im_width);
	//Calculate normalization value (should be at position 0 in interval [-N/2,N/2]) 
	DType norm_val = calculateDeapodizationValue(0,gi_host->grid_width_inv,gi_host->kernel_width,beta);
	norm_val = norm_val * norm_val * norm_val;

	forwardDeapodizationKernel<<<grid_dim,block_dim>>>(imdata_d,beta,norm_val);
}

void performPadding(DType* imdata_d,
					CufftType* gdata_d,					
					GriddingInfo* gi_host)
{
	int ind_off = (int)(gi_host->im_width * ((DType)gi_host->osr -1.0f)/(DType)2);

	printf("start padding image with offset %d\n",ind_off);

	dim3 grid_dim(gi_host->im_width,gi_host->im_width,1);
	dim3 block_dim(gi_host->im_width);
	paddingKernel<<<grid_dim,block_dim>>>(imdata_d,gdata_d,ind_off);
}
