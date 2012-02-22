#include "griddingFunctions.h"

#ifdef _WIN32 
	#define _USE_MATH_DEFINES	
#endif

#include <math.h>

#include <assert.h>

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
#define BETA (M_PI*sqrt(sqr(KERNEL_WIDTH/OVERSAMPLING_RATIO*(OVERSAMPLING_RATIO-0.5))-0.8))
#define I0_BETA	(i0(BETA))
#define kernel(__radius) (i0 (BETA * sqrt (1 - sqr(__radius))) / I0_BETA)

/*END Zwart*/

#define round(x) floor((x) + 0.5)

void loadGrid3Kernel(float *kernTab, int kernel_entries)	
{
    /* check input data */
    assert( kernTab != NULL );
	long i;
    long size = kernel_entries;
	float rsqr = 0.0f;
    /* load table */
	for (i=1; i<size-1; i++)	
    {
		rsqr = sqrt(i/(float)(size-1));//*(i/(float)(size-1));
		kernTab[i] = static_cast<float>(kernel(rsqr)); /* kernel table for radius squared */
		//assert(kernTab[i]!=kernTab[i]); //check is NaN
	}

    /* ensure center point is 1 */
    kernTab[0] = 1.0;

    /* ensure last point is zero */
    kernTab[size-1] = 0.0;
} /* end loadGrid3Kernel() */


/* set bounds for current data point based on kernel and grid limits */
void set_minmax (double x, int *min, int *max, int maximum, double radius)	
{
	*min = (int) ceil (x - radius);
	printf("x - radius %f - %f => %d ### ",x,radius,*min);
	*max = (int) floor (x + radius);
	printf("x + radius %f - %f => %d\n",x,radius,*max);
	if (*min < 0) *min = 0;
	if (*max >= maximum) *max = maximum-1;
}

inline int getIndex(int x, int y, int z, int gwidth)
{
	return x + gwidth * (y + gwidth * z);
}

void gridding3D(float* data, float* crds, float* gdata, float* kernel, int* sectors, int sector_count, int* sector_centers, int kernel_width, int kernel_count, int width)
{
	int imin, imax, jmin, jmax, kmin, kmax, i, j, k, ind;
	float x, y, z, ix, jy, kz;

    /* kr */
	float dx_sqr, dy_sqr, dz_sqr, dz_sqr_PLUS_dy_sqr, dist_sqr, val;
	int center_x, center_y, center_z, max_x, max_y, max_z;
	
	float radius = static_cast<float>(kernel_width) / width;
	printf("radius %f\n",radius);
	float width_inv = 1.0f / width;
	float radiusSquared = radius * radius;
	float kernelRadius_invSqr = 1 / radiusSquared;

	float dist_multiplier = (kernel_count - 1) * kernelRadius_invSqr;

	for (int sec = 0; sec < sector_count; sec++)
	{
		center_x = sector_centers[sec * 3];
		center_y = sector_centers[sec * 3 + 1];
		center_z = sector_centers[sec * 3 + 2];

		printf("handling center (%d,%d,%d) in sector %d\n",center_x,center_y,center_z,sec);

		for (int data_cnt = sectors[sec]; data_cnt < sectors[sec+1];data_cnt++)
		{
			printf("handling data point %d = %f\n",data_cnt,data[data_cnt]);

			x = crds[3*data_cnt];
			y = crds[3*data_cnt +1];
			z = crds[3*data_cnt +2];
			printf("data k-space coords (%f, %f, %f)\n",x,y,z);
			
			max_x = width;
			max_y = width;
			max_z = width;

			/* set the boundaries of final dataset for gridding this point */
			ix = x * width + center_x;
			set_minmax(ix, &imin, &imax, max_x, KERNEL_WIDTH/2.0f);
			jy = y * width + center_y;
			set_minmax(jy, &jmin, &jmax, max_y, KERNEL_WIDTH/2.0f);
			kz = z * width + center_z;
			set_minmax(kz, &kmin, &kmax, max_z, KERNEL_WIDTH/2.0f);
			printf("grid position of data point: %f,%f,%f\n",ix,jy,kz);
			printf("boundaries: x %d to %d, y %d to %d, z %d to %d\n",imin,imax,jmin,jmax,kmin,kmax);

			/* grid this point onto the neighboring cartesian points */
			for (k=kmin; k<=kmax; k++)	
			{
				kz = (k - center_z) *width_inv;
				dz_sqr = kz - z;
				dz_sqr *= dz_sqr;
				
				for (j=jmin; j<=jmax; j++)	
				{
					jy = (j - center_y) *width_inv;
					dy_sqr = jy - y;
					dy_sqr *= dy_sqr;
					dz_sqr_PLUS_dy_sqr = dz_sqr + dy_sqr;
					if (dz_sqr_PLUS_dy_sqr < radiusSquared)	
					{
						for (i=imin; i<=imax; i++)	
						{
							ix = (i - center_x) *width_inv;
							dx_sqr = ix - x;
							dx_sqr *= dx_sqr;
							dist_sqr = dx_sqr + dz_sqr_PLUS_dy_sqr;
							//printf("dx_sqr=(%f-%f)^2=%f dy_sqr=(%f-%f)^2=%f dz_sqr=(%f-%f)^2=%f -> dist_sqr= %f\n",ix,x,dx_sqr,jy,y,dy_sqr,kz,z,dz_sqr,dist_sqr);
							if (dist_sqr < radiusSquared)	
							{
								/* get kernel value */
								val = kernel[(int) round(dist_sqr * dist_multiplier)];
								//printf("distance sqr %f - kernel-value %f\n",dist_sqr,val);
								
								ind = getIndex(i,j,k,width);
								
								//printf("calculating index for output grid with x=%d, y=%d, z=%d -> %d\n",i,j,k,ind);
								
								/* multiply data by current kernel val */
								
								/* grid complex or scalar */
								gdata[2*ind] = val*data[2*data_cnt];
								gdata[2*ind+1] = val*data[2*data_cnt+1];
							} /* kernel bounds check, spherical support */
						} /* x 	 */
					} /* kernel bounds check, spherical support */
				} /* y */
			} /* z */
		}
	}
}

