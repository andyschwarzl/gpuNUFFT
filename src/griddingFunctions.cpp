#include "griddingFunctions.hpp"

/*BEGIN Zwart*/
/**************************************************************************
 *  FROM GRID_UTILS.C
 *
 *  Author: Nick Zwart, Dallas Turley, Ken Johnson, Jim Pipe 
 *  Date: 2011 apr 11
 *  Rev: 2011 aug 21
 * ...
*/

/************************************************************************** KERNEL */

/* 
 *	Summary: Allocates the 3D spherically symmetric kaiser-bessel function 
 *	         for kernel table lookup.
 *  
 *	         This lookup table is with respect to the radius squared.
 *	         and is based on the work described in Beatty et al. MRM 24, 2005
 */
static float i0( float x )
{
	float ax = fabs(x);
	float ans;
	float y;

	if (ax < 3.75f) 
    {
		y=x/3.75f,y=y*y;
		ans=1.0f+y*(3.5156229f+y*(3.0899424f+y*(1.2067492f
			   +y*(0.2659732f+y*(0.360768e-1f+y*0.45813e-2f)))));
	} 
    else 
    {
		y=3.75f/ax;
		ans=(exp(ax)/sqrt(ax))*(0.39894228f+y*(0.1328592e-1f
				+y*(0.225319e-2f+y*(-0.157565e-2f+y*(0.916281e-2f
				+y*(-0.2057706e-1f+y*(0.2635537e-1f+y*(-0.1647633e-1f
				+y*0.392377e-2f))))))));
	}
	return (ans);
}


/* LOADGRID3KERNEL()
 * Loads a radius of the circularly symmetric kernel into a 1-D array, with
 * respect to the kernel radius squared.
 */
#define sqr(__se) ((__se)*(__se))
//#define BETA (M_PI*sqrt(sqr(DEFAULT_KERNEL_WIDTH/DEFAULT_OVERSAMPLING_RATIO*(DEFAULT_OVERSAMPLING_RATIO-0.5))-0.8))
//#define I0_BETA	(i0(BETA))
//#define kernel(__radius) (i0 (BETA * sqrt (1 - sqr(__radius))) / I0_BETA)

#define BETA(__kw,__osr) (M_PI*sqrt(sqr(__kw/__osr*(__osr-0.5f))-0.8f))
#define I0_BETA(__kw,__osr)	(i0(BETA(__kw,__osr)))
#define kernel(__radius,__kw,__osr) (i0 (BETA(__kw,__osr) * sqrt (1 - sqr(__radius))) / I0_BETA(__kw,__osr))

/*END Zwart*/

long calculateGrid3KernelSize()
{
	return calculateGrid3KernelSize(DEFAULT_OVERSAMPLING_RATIO,DEFAULT_KERNEL_RADIUS);
}

long calculateGrid3KernelSize(float osr, float kernel_radius)
{
	//calculate kernel density (per grid/kernel unit) 
	//nearest neighbor with maximum aliasing error
	//of 0.001
	
	long kernel_osf = (long)(floor(0.91f/(osr * MAXIMUM_ALIASING_ERROR)));

	float kernel_radius_osr = static_cast<float>(osr * kernel_radius);

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

void loadGrid3Kernel(DType *kernTab,long kernel_entries, int kernel_width, float osr)	
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
    kernTab[0] = 1.0f;

    /* ensure last point is zero */
    kernTab[kernel_entries-1] = 0.0f;
} /* end loadGrid3Kernel() */


