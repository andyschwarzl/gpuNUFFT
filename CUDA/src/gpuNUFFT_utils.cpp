#include "gpuNUFFT_utils.hpp"

DType i0(DType x)
{
  DType ax = fabs(x);
  DType ans;
  DType y;

  if (ax < (DType)3.75f)
  {
    y = x / 3.75f, y = y * y;
    ans = 1.0f +
      y * (3.5156229f +
          y * (3.0899424f +
            y * (1.2067492f +
              y * (0.2659732f +
                y * (0.360768e-1f + y * 0.45813e-2f)))));
  }
  else
  {
    y = 3.75f / ax;
    ans = (exp(ax) / sqrt(ax)) *
      (0.39894228f +
       y * (0.1328592e-1f +
         y * (0.225319e-2f +
           y * (-0.157565e-2f +
             y * (0.916281e-2f +
               y * (-0.2057706e-1f +
                 y * (0.2635537e-1f +
                   y * (-0.1647633e-1f +
                     y * 0.392377e-2f))))))));
  }
  return (ans);
}

long calculateGrid3KernelSize()
{
  return calculateGrid3KernelSize(DEFAULT_OVERSAMPLING_RATIO,
                                  DEFAULT_KERNEL_WIDTH);
}

// calculate necessary kernel density (per grid/kernel unit)
// nearest neighbor with maximum aliasing error
// of 0.001
//
// based on Beatty et al. MRM 24, 2005
long calculateGrid3KernelSize(DType osr, IndType kernel_width)
{
  long kernel_osf = (long)(floor((DType)0.91 / (osr * MAXIMUM_ALIASING_ERROR)));

  return (long)(kernel_osf * (DType)kernel_width * 0.5);
}

long calculateKernelSizeLinInt(DType osr, IndType kernel_width)
{
  long kernel_osf =
      (long)(floor(sqrt(0.37 / (sqr(osr) * MAXIMUM_ALIASING_ERROR_LIN_INT))));

  return (long)(kernel_osf * (DType)kernel_width * 0.5);
}

void loadGrid3Kernel(DType *kernTab)
{
  load1DKernel(kernTab, calculateGrid3KernelSize(), DEFAULT_KERNEL_WIDTH,
               DEFAULT_OVERSAMPLING_RATIO);
}

void loadGrid3Kernel(DType *kernTab, long kernel_entries)
{
  load1DKernel(kernTab, kernel_entries, DEFAULT_KERNEL_WIDTH,
               DEFAULT_OVERSAMPLING_RATIO);
}

void load1DKernel(DType *kernTab, long kernel_entries, int kernel_width,
                  DType osr)
{
  /* check input data */
  assert(kernTab != NULL);
  long i;

  double rsqr = 0.0;
  /* load table */
  for (i = 1; i < kernel_entries - 1; i++)
  {
    // dirty fix for kw 1 -> something
    // like nearest neighbor
    if (kernel_width == 1)
    {
      kernTab[i] = 1.0f;
    }
    else
    {
      rsqr = (double)sqrt(
          i / (double)(kernel_entries - 1));  //*(i/(float)(size-1));
      kernTab[i] = static_cast<DType>(
          (DType)kernel((DType)rsqr, kernel_width,
                        osr)); /* kernel table for radius squared */
    }
    //    assert(!isnan(kernTab[i])); //check is NaN
  }

  /* ensure center point is 1 */
  kernTab[0] = (DType)1.0f;

  /* ensure last point is zero */
  kernTab[kernel_entries - 1] = (DType)0.0f;
} /* end loadGrid3Kernel() */

void load2DKernel(DType *kernTab, long kernel_entries, int kernel_width,
                  DType osr)
{
  /* check input data */
  assert(kernTab != NULL);
  load1DKernel(kernTab, kernel_entries, kernel_width, osr);

  /* load table */
  for (long j = 0; j < kernel_entries; j++)
  {
    for (long i = 0; i < kernel_entries; i++)
    {
      kernTab[i + j * kernel_entries] = kernTab[j] * kernTab[i];
    }
  }
}  // end load2DKernel()

void load3DKernel(DType *kernTab, long kernel_entries, int kernel_width,
                  DType osr)
{
  /* check input data */
  assert(kernTab != NULL);
  load1DKernel(kernTab, kernel_entries, kernel_width, osr);

  /* load table */
  for (long k = 0; k < kernel_entries; k++)
    for (long j = 0; j < kernel_entries; j++)
    {
      for (long i = 0; i < kernel_entries; i++)
      {
        kernTab[i + kernel_entries * (j + k * kernel_entries)] =
            kernTab[j] * kernTab[i] * kernTab[k];
      }
    }
}  // end load3DKernel()

