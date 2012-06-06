

static __global__ void  backproj(cufftComplex *val,
                                 cufftComplex *tmp1,
                                 cufftComplex *_ipk_we,
                                 int *_the_index)
{
      int p   = threadIdx.x;
      int idx = _the_index[p];
      tmp1[idx].x +=   val[0].x * _ipk_we[p].x +  val[0].y*_ipk_we[p].y;
      tmp1[idx].y += - val[0].x*_ipk_we[p].y +  val[0].y*_ipk_we[p].x ;  
}



static __global__ void  backprojWS(cufftComplex *val,
                                   cufftComplex *tmp1,
                                   cufftComplex *_ipk_we,
                                   int *_the_index, 
								   int numP, 
								   int *ws_indices, 
								   int ws_size)
{
      int p = threadIdx.x;
      int k = blockIdx.x; 
      if (k < ws_size)
      {
          int j = ws_indices[k];     
          int q = p + numP*j;

          int idx = _the_index[q];
          tmp1[idx].x +=   val[j].x * _ipk_we[q].x +  val[j].y*_ipk_we[q].y;
          tmp1[idx].y += - val[j].x *_ipk_we[q].y  +  val[j].y*_ipk_we[q].x;  
      }
}
     

static __global__ void  backprojVX(int *vxIdx,
                                   int *onset,
                                   cufftComplex *we, 
                                   int *id,
                                   int *sz, 
                                   cufftComplex *val,
								   cufftComplex *tmp1,
								   int numVox)
                                  
{
      int t = blockIdx.x*blockDim.x + threadIdx.x;
      if (t < numVox)
      {
          int idx = vxIdx[t];
          int ons = onset[t];
          int size = sz[t];
          
          for (int k = 0; k < size; k++)
          {          
              int j = id[ons+k];
              tmp1[idx].x +=   val[j].x * we[ons+k].x +  val[j].y*we[ons+k].y;
              tmp1[idx].y +=   - val[j].x * we[ons+k].y  +  val[j].y*we[ons+k].x;            
          }
      }
}




static __global__ void  dosens(cufftComplex *val,
                               cufftComplex *tmp2,
                               cufftComplex *_ipk_we,
                               int *_the_index,
							   int numP, 
							   int numK)
{     
      int k = blockDim.x * blockIdx.x + threadIdx.x;
      if (k < numK)
      {
          val[k].x = 0;  val[k].y = 0;          
          for (int p = 0; p < numP; p++)
          { 
              int idx = _the_index[numP*k + p];          
              val[k].x += tmp2[idx].x*_ipk_we[numP*k + p].x - tmp2[idx].y*_ipk_we[numP*k + p].y;
              val[k].y += tmp2[idx].x*_ipk_we[numP*k + p].y + tmp2[idx].y*_ipk_we[numP*k + p].x;
          }
      }
}

static __global__ void quadradd(cufftComplex *_r,
                                cufftComplex *tmp2,
                                int w, 
								int h, 
								int d, 
								int w_pad, 
								int h_pad, 
								int d_pad)
{
	int z = threadIdx.x;
    int y = blockIdx.y;
    int x = blockIdx.x;
    int idx_pad = z*w_pad*h_pad+y*w_pad+x;
    int idx = z*w*h + y*w + x;
    
    _r[idx].x = sqrt((_r[idx].x*_r[idx].x) + (tmp2[idx_pad].x*tmp2[idx_pad].x)) ;
    _r[idx].y = sqrt((_r[idx].y*_r[idx].y) + (tmp2[idx_pad].y*tmp2[idx_pad].y)) ;
}

static __global__ void  downwind(cufftComplex *_r,
                                 cufftComplex *tmp2,
                                 cufftComplex *_sens, 
								int w, 
								int h, 
								int d, 
								int w_pad, 
								int h_pad, 
								int d_pad)
{
    int z = threadIdx.x;
    int y = blockIdx.y;
    int x = blockIdx.x;
    int idx_pad = z*w_pad*h_pad+y*w_pad+x;
    int idx = z*w*h + y*w + x;
    
    _r[idx].x +=  tmp2[idx_pad].x*_sens[idx].x + tmp2[idx_pad].y*_sens[idx].y;
    _r[idx].y +=  - tmp2[idx_pad].x*_sens[idx].y + tmp2[idx_pad].y*_sens[idx].x;

}

static __global__ void  upwind(cufftComplex *_r,
                               cufftComplex *tmp2,
                               cufftComplex *_sens, 
							   int w, 
							   int h, 
							   int d, 
							   int w_pad, 
							   int h_pad, 
							   int d_pad)
{
    int z = threadIdx.x;
    int y = blockIdx.y;
    int x = blockIdx.x;
    int idx_pad = z*w_pad*h_pad+y*w_pad+x;
    int idx = z*w*h + y*w + x;
    
    _r[idx_pad].x +=  tmp2[idx].x*_sens[idx].x - tmp2[idx].y*_sens[idx].y;
    _r[idx_pad].y +=  tmp2[idx].x*_sens[idx].y + tmp2[idx].y*_sens[idx].x;

}

static __global__ void  sn_mult(cufftComplex *_r,
                               cufftComplex *tmp2,
                               float *_sn, 
							   int w, 
							   int h, 
							   int d, 
							   int w_pad, 
							   int h_pad, 
							   int d_pad)
{
    int z = threadIdx.x;
    int y = blockIdx.y;
    int x = blockIdx.x;
    int idx_pad = z*w_pad*h_pad+y*w_pad+x;
    int idx = z*w*h + y*w + x;
    
    _r[idx_pad].x =  tmp2[idx].x*_sn[idx];
    _r[idx_pad].y =  tmp2[idx].y*_sn[idx];

}



static __global__ void  scmult(cufftComplex *_a,cufftComplex *_b, float alpha, int n)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t < n)
    {        
         _a[t].x = _b[t].x * alpha;
         _a[t].y = _b[t].y * alpha;
    }
}



static __global__ void  scmultplus(cufftComplex *_a,cufftComplex *_b, float alpha, int n)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t < n)
    {        
         _a[t].x = _a[t].x * alpha + _b[t].x;
         _a[t].y = _a[t].y * alpha + _b[t].y;
    }
}

static __global__ void  addcoiltores(cufftComplex *_res, cufftComplex *_coil, int n, int offset)
{
    int t =  blockIdx.x + blockIdx.y * gridDim.x + threadIdx.x * gridDim.x * gridDim.y;
    if (t+offset < n)
    {        
         _res[t+offset].x += _coil[t].x;
         _res[t+offset].y += _coil[t].y;
    }    
}


static __global__ void  scpm(cufftComplex *_a,cufftComplex *_b, float alpha, int n)
{
    int t =  blockDim.x * blockIdx.x + threadIdx.x;
    if (t < n)
    {        
         _a[t].x += _b[t].x * alpha;
         _a[t].y += _b[t].y * alpha;
    }    
}
// 
// __global__ void Dev_dot(cufftComplex x[], cufftComplex y[], float z[], int n) {
//    /* Use tmp to store products of vector components in each block */
//    /* Can't use variable dimension here                            */
//    __shared__ float tmp[MAX_BLOCK_SZ];
//    int t = blockDim.x * blockIdx.x + threadIdx.x;
//    int loc_t = threadIdx.x;
// 
// 
//    if (t < n) 
//    {
//        tmp[loc_t] = x[t].x*y[t].x + x[t].y*y[t].y;
//    }
//    __syncthreads();
//    
//    /* This uses a tree structure to do the addtions */
//    for (int stride = blockDim.x/2; stride >  0; stride /= 2) {
//       if (loc_t < stride)
//       {
//          tmp[loc_t] += tmp[loc_t + stride];
//       }
//       __syncthreads();
//    }
// 
//    /* Store the result from this cache block in z[blockIdx.x] */
//    if (threadIdx.x == 0) {
//       z[blockIdx.x] = tmp[0];
//    }
// }  /* Dev_dot */    
// 
//  
// float Dot_wrapper(cufftComplex x_d[], cufftComplex y_d[], float z_d[], float z_h[],
//       int n, int blocks, int threads) { 
//    int i;
//    float dot = 0;
// 
//    /* Invoke kernel */
//    Dev_dot<<<blocks, threads>>>(x_d, y_d, z_d, n);
//    cudaThreadSynchronize();
// 
//    /* Note that we don't need to copy z_d back to host */
//    for (i = 0; i < blocks; i++)
//    {
//       dot += z_h[i];
//    }
//    return dot;
// } 
// 

