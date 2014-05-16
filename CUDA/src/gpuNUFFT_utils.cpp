#include "gpuNUFFT_utils.hpp"

long calculateGrid3KernelSize()
{
  return calculateGrid3KernelSize(DEFAULT_OVERSAMPLING_RATIO,DEFAULT_KERNEL_RADIUS);
}

//calculate necessary kernel density (per grid/kernel unit) 
//nearest neighbor with maximum aliasing error
//of 0.001
//
//based on Beatty et al. MRM 24, 2005
long calculateGrid3KernelSize(DType osr, DType kernel_radius)
{
  long kernel_osf = (long)(floor((DType)0.91/(osr * MAXIMUM_ALIASING_ERROR)));

  DType kernel_radius_osr = static_cast<DType>(osr * kernel_radius);

  return (long)(kernel_osf * kernel_radius_osr);
}

long calculateKernelSizeLinInt(double osr, double kernel_radius)
{
  long kernel_osf = (long)(floor(sqrt(0.37/(sqr(osr) * MAXIMUM_ALIASING_ERROR_LIN_INT))));

  DType kernel_radius_osr = static_cast<DType>(osr * kernel_radius);

  return (long)(kernel_osf * kernel_radius_osr);
}

void loadGrid3Kernel(DType *kernTab)
{
  load1DKernel(kernTab,calculateGrid3KernelSize(),DEFAULT_KERNEL_WIDTH,DEFAULT_OVERSAMPLING_RATIO);
}

void loadGrid3Kernel(DType *kernTab,long kernel_entries)
{
  load1DKernel(kernTab,kernel_entries,DEFAULT_KERNEL_WIDTH,DEFAULT_OVERSAMPLING_RATIO);
}

void load1DKernel(DType *kernTab,long kernel_entries, int kernel_width, DType osr)	
{
  /* check input data */
  assert( kernTab != NULL );
  long i;

  double rsqr = 0.0;
  /* load table */
  for (i=1; i<kernel_entries-1; i++)	
  {
    rsqr = (double)sqrt(i/(double)(kernel_entries-1));//*(i/(float)(size-1));
    kernTab[i] = static_cast<DType>(kernel(rsqr,kernel_width,osr)); /* kernel table for radius squared */
    //assert(kernTab[i]!=kernTab[i]); //check is NaN
  }

  /* ensure center point is 1 */
  kernTab[0] = (DType)1.0f;

  /* ensure last point is zero */
  kernTab[kernel_entries-1] = (DType)0.0f;
} /* end loadGrid3Kernel() */

void load2DKernel(DType *kernTab,long kernel_entries, int kernel_width, DType osr)	
{
  /* check input data */
  assert( kernTab != NULL );
  load1DKernel(kernTab,kernel_entries,kernel_width,osr);

  /* load table */
  for (long j=0; j<kernel_entries; j++)	
  { 
    for (long i=0; i<kernel_entries; i++)	
    {
      kernTab[i+j*kernel_entries] = kernTab[j]*kernTab[i];
    }
  }
} // end load2DKernel()

void load3DKernel(DType *kernTab,long kernel_entries, int kernel_width, DType osr)	
{
  /* check input data */
  assert( kernTab != NULL );
  load1DKernel(kernTab,kernel_entries,kernel_width,osr);

  /* load table */
  for (long k=0; k<kernel_entries; k++)	
    for (long j=0; j<kernel_entries; j++)	
    { 
      for (long i=0; i<kernel_entries; i++)	
      {
        kernTab[i+kernel_entries*(j + k*kernel_entries)] = kernTab[j]*kernTab[i]*kernTab[k];
      }
    }
} // end load3DKernel()

