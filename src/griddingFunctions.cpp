#include "griddingFunctions.hpp"

long calculateGrid3KernelSize()
{
	return calculateGrid3KernelSize(DEFAULT_OVERSAMPLING_RATIO,DEFAULT_KERNEL_RADIUS);
}

//calculate kernel density (per grid/kernel unit) 
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

void loadGrid3Kernel(DType *kernTab)
{
	loadGrid3Kernel(kernTab,calculateGrid3KernelSize(),DEFAULT_KERNEL_WIDTH,DEFAULT_OVERSAMPLING_RATIO);
}

void loadGrid3Kernel(DType *kernTab,long kernel_entries)
{
	loadGrid3Kernel(kernTab,kernel_entries,DEFAULT_KERNEL_WIDTH,DEFAULT_OVERSAMPLING_RATIO);
}

void loadGrid3Kernel(DType *kernTab,long kernel_entries, int kernel_width, DType osr)	
{
    /* check input data */
    assert( kernTab != NULL );
	long i;

	DType rsqr = 0.0f;
    /* load table */
	for (i=1; i<kernel_entries-1; i++)	
    {
		rsqr = sqrt(i/(DType)(kernel_entries-1));//*(i/(float)(size-1));
		kernTab[i] = static_cast<DType>(kernel(rsqr,kernel_width,osr)); /* kernel table for radius squared */
		//assert(kernTab[i]!=kernTab[i]); //check is NaN
	}

    /* ensure center point is 1 */
    kernTab[0] = (DType)1.0f;

    /* ensure last point is zero */
    kernTab[kernel_entries-1] = (DType)0.0f;
} /* end loadGrid3Kernel() */

