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
//#define BETA (M_PI*sqrt(sqr(DEFAULT_KERNEL_WIDTH/DEFAULT_OVERSAMPLING_RATIO*(DEFAULT_OVERSAMPLING_RATIO-0.5))-0.8))
//#define I0_BETA	(i0(BETA))
//#define kernel(__radius) (i0 (BETA * sqrt (1 - sqr(__radius))) / I0_BETA)

#define BETA(__kw,__osr) (M_PI*sqrt(sqr(__kw/__osr*(__osr-0.5))-0.8))
#define I0_BETA(__kw,__osr)	(i0(BETA(__kw,__osr)))
#define kernel(__radius,__kw,__osr) (i0 (BETA(__kw,__osr) * sqrt (1 - sqr(__radius))) / I0_BETA(__kw,__osr))

/*END Zwart*/

#define round(x) floor((x) + 0.5)

long calculateGrid3KernelSize()
{
	return calculateGrid3KernelSize(DEFAULT_OVERSAMPLING_RATIO,DEFAULT_KERNEL_RADIUS);
}

long calculateGrid3KernelSize(float osr, float kernel_radius)
{
	//calculate kernel density (per grid/kernel unit) 
	//nearest neighbor with maximum aliasing error
	//of 0.001
	
	long kernel_osf = floor(0.91f/(osr * MAXIMUM_ALIASING_ERROR));

	float kernel_radius_osr = static_cast<float>(osr * kernel_radius);

    return kernel_osf * kernel_radius_osr;
}

void loadGrid3Kernel(float *kernTab)
{
	loadGrid3Kernel(kernTab,calculateGrid3KernelSize(),DEFAULT_KERNEL_WIDTH,DEFAULT_OVERSAMPLING_RATIO);
}

void loadGrid3Kernel(float *kernTab,long kernel_entries)
{
	loadGrid3Kernel(kernTab,kernel_entries,DEFAULT_KERNEL_WIDTH,DEFAULT_OVERSAMPLING_RATIO);
}

void loadGrid3Kernel(float *kernTab,long kernel_entries, int kernel_width, float osr)	
{
    /* check input data */
    assert( kernTab != NULL );
	long i;

	float rsqr = 0.0f;
    /* load table */
	for (i=1; i<kernel_entries-1; i++)	
    {
		rsqr = sqrt(i/(float)(kernel_entries-1));//*(i/(float)(size-1));
		kernTab[i] = static_cast<float>(kernel(rsqr,kernel_width,osr)); /* kernel table for radius squared */
		//assert(kernTab[i]!=kernTab[i]); //check is NaN
	}

    /* ensure center point is 1 */
    kernTab[0] = 1.0f;

    /* ensure last point is zero */
    kernTab[kernel_entries-1] = 0.0f;
} /* end loadGrid3Kernel() */


/* set bounds for current data point based on kernel and grid limits */
void set_minmax (double x, int *min, int *max, int maximum, double radius)	
{
	*min = (int) ceil (x - radius);
	*max = (int) floor (x + radius);
	//check boundaries
	if (*min < 0) *min = 0;
	if (*max >= maximum) *max = maximum;
	printf("x - radius %f - %f => %d ### ",x,radius,*min);
	printf("x + radius %f - %f => %d\n",x,radius,*max);
}

inline int getIndex(int x, int y, int z, int gwidth)
{
	return x + gwidth * (y + gwidth * z);
}

void gridding3D(float* data, float* crds, float* gdata, float* kernel, int* sectors, int sector_count, int* sector_centers, int sector_width, int kernel_width, int kernel_count, int width)
{
	int imin, imax, jmin, jmax, kmin, kmax, i, j, k, ind;
	float x, y, z, ix, jy, kz;

    /* kr */
	float dx_sqr, dy_sqr, dz_sqr, dz_sqr_PLUS_dy_sqr, dist_sqr, val;
	int center_x, center_y, center_z, max_x, max_y, max_z;
	
	float kernel_radius = static_cast<float>(kernel_width) / 2.0f;
	float radius = kernel_radius / static_cast<float>(width);

	printf("radius rel. to grid width %f\n",radius);
	float width_inv = 1.0f / width;
	float radiusSquared = radius * radius;
	float kernelRadius_invSqr = 1 / radiusSquared;

	float dist_multiplier = (kernel_count - 1) * kernelRadius_invSqr;
	//int sector_width = 10;
	int sector_dim = sector_width * sector_width * sector_width;
	int sector_offset = floor(sector_width / 2.0f);

	printf("sector offset = %d",sector_offset);
	float** sdata =  (float**)malloc(sector_count*sizeof(float*));

	assert(sectors != NULL);

	for (int sec = 0; sec < sector_count; sec++)
	{
		sdata[sec] = (float *) calloc(sector_dim * 2, sizeof(float)); // 5*5*5 * 2
		assert(sdata[sec] != NULL);

		center_x = sector_centers[sec * 3];
		center_y = sector_centers[sec * 3 + 1];
		center_z = sector_centers[sec * 3 + 2];

		printf("\nhandling center (%d,%d,%d) in sector %d\n",center_x,center_y,center_z,sec);

		for (int data_cnt = sectors[sec]; data_cnt < sectors[sec+1];data_cnt++)
		{
			printf("handling %d data point = %f\n",data_cnt+1,data[data_cnt]);

			x = crds[3*data_cnt];
			y = crds[3*data_cnt +1];
			z = crds[3*data_cnt +2];
			printf("data k-space coords (%f, %f, %f)\n",x,y,z);
			
			max_x = sector_width-1;
			max_y = sector_width-1;
			max_z = sector_width-1;

			/* set the boundaries of final dataset for gridding this point */
			ix = (x + 0.5f) * (width) - center_x + sector_offset;
			set_minmax(ix, &imin, &imax, max_x, kernel_radius);
			jy = (y + 0.5f) * (width) - center_y + sector_offset;
			set_minmax(jy, &jmin, &jmax, max_y, kernel_radius);
			kz = (z + 0.5f) * (width) - center_z + sector_offset;
			set_minmax(kz, &kmin, &kmax, max_z, kernel_radius);

			printf("sector grid position of data point: %f,%f,%f\n",ix,jy,kz);
			//printf("boundaries: x %d to %d, y %d to %d, z %d to %d\n",imin,imax,jmin,jmax,kmin,kmax);

			/* grid this point onto the neighboring cartesian points */
			for (k=kmin; k<=kmax; k++)	
			{
				kz = static_cast<float>((k + center_z - sector_offset)) / static_cast<float>((width)) - 0.5f;//(k - center_z) *width_inv;
				dz_sqr = kz - z;
				dz_sqr *= dz_sqr;
				//printf("-----------------------------------------------------------------------------####\n " \
				//	"(%d + %d - %d)/%d - 0.5 = %f\ndz_sqr = %f\n",k,center_z,sector_offset,width,kz,dz_sqr);
				if (dz_sqr < radiusSquared)
				{
					for (j=jmin; j<=jmax; j++)	
					{
						jy = static_cast<float>(j + center_y - sector_offset) / static_cast<float>((width)) - 0.5f;   //(j - center_y) *width_inv;
						dy_sqr = jy - y;
						dy_sqr *= dy_sqr;
						//dz_sqr_PLUS_dy_sqr = dy_sqr;// + dz_sqr;
						//printf("(%d + %d - %d)/%d - 0.5 = %f\ndy_sqr = %f\n",j,center_y,sector_offset,width,jy,dy_sqr);
				
						if (dy_sqr < radiusSquared)	
						{
							for (i=imin; i<=imax; i++)	
							{
								ix = static_cast<float>(i + center_x - sector_offset) / static_cast<float>((width)) - 0.5f;// (i - center_x) *width_inv;
								dx_sqr = ix - x;
								dx_sqr *= dx_sqr;
								//dist_sqr = dx_sqr;// + dz_sqr_PLUS_dy_sqr;
								//printf("(%d + %d - %d)/%d - 0.5 = %f\ndx_sqr = %f\n",i,center_x,sector_offset,width,ix,dx_sqr);
				
								if (dx_sqr < radiusSquared)	
								{
									//printf("dist_sqr = %f\n", dist_sqr);
									/* get kernel value */
									//val = kernel[(int) round(dist_sqr * dist_multiplier)];
									//Berechnung mit Separable Filters 

									val = kernel[(int) round(dz_sqr * dist_multiplier)] *
										  kernel[(int) round(dy_sqr * dist_multiplier)] *
										  kernel[(int) round(dx_sqr * dist_multiplier)];

									//printf("distance sqr %f - kernel-value %f\n",dist_sqr,val);
								
									ind = getIndex(i,j,k,sector_width);
								
									//printf("calculating index for output grid with x=%d, y=%d, z=%d -> %d ",i,j,k,ind);
								
									/* multiply data by current kernel val */
								
									/* grid complex or scalar */
									//gdata[2*ind] = val*data[2*data_cnt];
									sdata[sec][2*ind] = val * data[2*data_cnt];
									//printf("and setting real value %f\n",gdata[2*ind]);
									//gdata[2*ind+1] = val*data[2*data_cnt+1];
									sdata[sec][2*ind+1] = val * data[2*data_cnt+1];
									//printf("sdata at index %d set to = %f\n",2*ind,sdata[2*ind]);
								} /* kernel bounds check x, spherical support */
							} /* x 	 */
						} /* kernel bounds check y, spherical support */
					} /* y */
				} /*kernel bounds check z */
			} /* z */
		} /*data points per sector*/
	
	}/*sectors*/
	
	//TODO copy data from sectors to original grid
	for (int sec = 0; sec < sector_count; sec++)
	{
		printf("DEBUG: showing entries of sector %d in z = 5 plane...\n",sec);
		center_x = sector_centers[sec * 3];
		center_y = sector_centers[sec * 3 + 1];
		center_z = sector_centers[sec * 3 + 2];
		
		int sector_ind_offset = getIndex(center_x - sector_offset,center_y - sector_offset,center_z - sector_offset,width);
		printf("sector index offset in resulting grid: %d\n", sector_ind_offset);
		for (int z = 0; z < sector_width; z++)
			for (int y = 0; y < sector_width; y++)
			{
				for (int x = 0; x < sector_width; x++)
				{
					int s_ind = 2*getIndex(x,y,z,sector_width) ;
					ind = 2*(sector_ind_offset + getIndex(x,y,z,width));
					//if (z==2)
					//	printf("%.4f ",sdata[sec][s_ind]);
					gdata[ind] = sdata[sec][s_ind]; //Re
					gdata[ind+1] = sdata[sec][s_ind+1];//Im
				}
				//if (z==2) printf("\n");
			}
		free(sdata[sec]);
	}
	free(sdata);
}

