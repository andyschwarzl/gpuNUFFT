#include "gpuNUFFT_cpu.hpp"

void gpuNUFFT_cpu(DType *data, DType *crds, DType *gdata, DType *kernel,
                  int *sectors, int sector_count, int *sector_centers,
                  int sector_width, int kernel_width, int kernel_count,
                  int width)
{
  int imin, imax, jmin, jmax, kmin, kmax, i, j, k, ind;
  DType x, y, z, ix, jy, kz;

  /* kr */
  DType dx_sqr, dy_sqr, dz_sqr, val;
  int center_x, center_y, center_z, max_x, max_y, max_z;

  DType kernel_radius = static_cast<DType>(kernel_width) / 2.0f;
  DType radius = kernel_radius / static_cast<DType>(width);

  if (DEBUG)
    printf("radius rel. to grid width %f\n", radius);
  DType radiusSquared = radius * radius;
  DType kernelRadius_invSqr = 1 / radiusSquared;

  DType dist_multiplier = (kernel_count - 1) * kernelRadius_invSqr;
  // int sector_width = 10;

  int sector_pad_width = sector_width + 2 * (int)floor(kernel_width / 2.0f);
  int sector_dim = sector_pad_width * sector_pad_width * sector_pad_width;
  int sector_offset = (int)floor(sector_pad_width / 2.0f);
  if (DEBUG)
    printf("sector offset = %d", sector_offset);
  DType **sdata = (DType **)malloc(sector_count * sizeof(DType *));

  assert(sectors != NULL);

  for (int sec = 0; sec < sector_count; sec++)
  {
    sdata[sec] = (DType *)calloc(sector_dim * 2, sizeof(DType));  // 5*5*5 * 2
    assert(sdata[sec] != NULL);

    center_x = sector_centers[sec * 3];
    center_y = sector_centers[sec * 3 + 1];
    center_z = sector_centers[sec * 3 + 2];

    if (DEBUG)
      printf("handling center (%d,%d,%d) in sector %d\n", center_x, center_y,
             center_z, sec);

    for (int data_cnt = sectors[sec]; data_cnt < sectors[sec + 1]; data_cnt++)
    {
      if (DEBUG)
        printf("handling %d data point = %f\n", data_cnt + 1,
               data[2 * data_cnt]);

      x = crds[3 * data_cnt];
      y = crds[3 * data_cnt + 1];
      z = crds[3 * data_cnt + 2];
      if (DEBUG)
        printf("data k-space coords (%f, %f, %f)\n", x, y, z);

      max_x = sector_pad_width - 1;
      max_y = sector_pad_width - 1;
      max_z = sector_pad_width - 1;

      /* set the boundaries of final dataset for gpuNUFFT this point */
      ix = (x + 0.5f) * (width)-center_x + sector_offset;
      set_minmax(&ix, &imin, &imax, max_x, kernel_radius);
      if (DEBUG)
        printf("ix=%f, imin = %d, imax = %d, max_x = %d\n", ix, imin, imax,
               max_x);
      jy = (y + 0.5f) * (width)-center_y + sector_offset;
      set_minmax(&jy, &jmin, &jmax, max_y, kernel_radius);
      kz = (z + 0.5f) * (width)-center_z + sector_offset;
      set_minmax(&kz, &kmin, &kmax, max_z, kernel_radius);

      if (DEBUG)
        printf("sector grid position of data point: %f,%f,%f\n", ix, jy, kz);

      /* grid this point onto the neighboring cartesian points */
      for (k = kmin; k <= kmax; k++)
      {
        kz = static_cast<DType>((k + center_z - sector_offset)) /
                 static_cast<DType>((width)) -
             0.5f;  //(k - center_z) *width_inv;
        dz_sqr = kz - z;
        dz_sqr *= dz_sqr;
        if (dz_sqr < radiusSquared)
        {
          for (j = jmin; j <= jmax; j++)
          {
            jy = static_cast<DType>(j + center_y - sector_offset) /
                     static_cast<DType>((width)) -
                 0.5f;  //(j - center_y) *width_inv;
            dy_sqr = jy - y;
            dy_sqr *= dy_sqr;
            if (dy_sqr < radiusSquared)
            {
              for (i = imin; i <= imax; i++)
              {
                ix = static_cast<DType>(i + center_x - sector_offset) /
                         static_cast<DType>((width)) -
                     0.5f;  // (i - center_x) *width_inv;
                dx_sqr = ix - x;
                dx_sqr *= dx_sqr;
                if (dx_sqr < radiusSquared)
                {
                  /* get kernel value */
                  // separable Filters
                  val = kernel[(int)round(dz_sqr * dist_multiplier)] *
                        kernel[(int)round(dy_sqr * dist_multiplier)] *
                        kernel[(int)round(dx_sqr * dist_multiplier)];
                  ind = getIndex(i, j, k, sector_pad_width);

                  /* multiply data by current kernel val */
                  /* grid complex or scalar */
                  sdata[sec][2 * ind] += val * data[2 * data_cnt];
                  sdata[sec][2 * ind + 1] += val * data[2 * data_cnt + 1];
                } /* kernel bounds check x, spherical support */
              }   /* x 	 */
            }     /* kernel bounds check y, spherical support */
          }       /* y */
        }         /*kernel bounds check z */
      }           /* z */
    }             /*data points per sector*/

  } /*sectors*/

  for (int sec = 0; sec < sector_count; sec++)
  {
    // printf("DEBUG: showing entries of sector %d in z = 5 plane...\n",sec);
    center_x = sector_centers[sec * 3];
    center_y = sector_centers[sec * 3 + 1];
    center_z = sector_centers[sec * 3 + 2];

    int sector_ind_offset =
        getIndex(center_x - sector_offset, center_y - sector_offset,
                 center_z - sector_offset, width);

    // printf("sector index offset in resulting grid: %d\n", sector_ind_offset);
    for (int z = 0; z < sector_pad_width; z++)
      for (int y = 0; y < sector_pad_width; y++)
      {
        for (int x = 0; x < sector_pad_width; x++)
        {
          int s_ind = 2 * getIndex(x, y, z, sector_pad_width);
          ind = 2 * (sector_ind_offset + getIndex(x, y, z, width));

          if (isOutlier(x, y, z, center_x, center_y, center_z, width,
                        sector_offset))
            continue;

          gdata[ind] += sdata[sec][s_ind];  // Re
          gdata[ind + 1] += sdata[sec][s_ind + 1];  // Im
        }
      }
    // printf("----------------------------------------------------\n");
    free(sdata[sec]);
  }
  free(sdata);
}

