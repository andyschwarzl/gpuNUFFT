#ifndef GPUNUFFT_TYPES_H_INCLUDED
#define GPUNUFFT_TYPES_H_INCLUDED

#include <cstdlib>
#include <iostream>
#include "config.hpp"

#define DEFAULT_VALUE(a) ((a == 0) ? 1 : a)

#define MAXIMUM_PAYLOAD 256

namespace gpuNUFFT
{
  struct IndPair : std::pair<IndType,IndType>
  {
    IndPair(IndType first, IndType second):
  std::pair<IndType,IndType>(first,second)
  {	
  }

  bool operator<(const IndPair& a) const
  {
    return this->second < a.second;
  }

  bool operator>(const IndPair& a) const
  {
    return this->second > a.second;
  }
  };

  //TODO work on dimensions
  //avoid ambiguity between length (1D) and multidimensional case (2D/3D)
  struct Dimensions
  {
    Dimensions():
  width(0),height(0),depth(0),channels(1),frames(1),length(0)
  {}

  Dimensions(IndType width, IndType height, IndType depth):
  width(width),height(height),depth(depth),channels(1),frames(1),length(0)
  {}

  Dimensions(IndType width, IndType height):
  width(width),height(height),depth(0),channels(1),frames(1),length(0)
  {}

  IndType width  ;
  IndType height ;
  IndType depth  ;

  IndType length; //1D case 

  IndType channels ;
  IndType frames ;

  IndType count()
  {
    return DEFAULT_VALUE(length) * DEFAULT_VALUE(width) * DEFAULT_VALUE(height) * DEFAULT_VALUE(depth) * DEFAULT_VALUE(channels) * DEFAULT_VALUE(frames);
  }

  Dimensions operator*(const DType alpha)
  {
    Dimensions res;
    res.width = (IndType)((*this).width * alpha);
    res.height = (IndType)((*this).height * alpha);
    res.depth = (IndType)((*this).depth * alpha);
    res.length = (IndType)((*this).length * alpha);
    return res;
  }

  Dimensions operator+(const IndType alpha)
  {
    Dimensions res;
    if (this->width > 0)
      res.width = (IndType)((*this).width + alpha);
    if (this->height > 0)
      res.height = (IndType)((*this).height + alpha);
    if (this->depth > 0)
      res.depth = (IndType)((*this).depth + alpha);
    if (this->length > 0)
      res.length = (IndType)((*this).length + alpha);
    return res;
  }
  };

  template <typename T>
  struct Array
  {
    Array():
  data(NULL)
  {}
  T* data;
  Dimensions dim;

  IndType count()
  {
    return dim.count();
  }

  };

  enum InterpolationType
  {
    CONST_LOOKUP,
    TEXTURE_LOOKUP,
    TEXTURE2D_LOOKUP,
    TEXTURE3D_LOOKUP
  };

  enum GpuNUFFTOutput
  {
    CONVOLUTION,
    FFT,
    DEAPODIZATION
  };

  enum FFTShiftDir
  {
    FORWARD,
    INVERSE
  };

  enum OperatorType
  {
    DEFAULT,
    TEXTURE,
    BALANCED,
    BALANCED_TEXTURE
  };

  struct GpuNUFFTInfo 
  {
    int data_count;
    int kernel_width; 
    int kernel_widthSquared;
    DType kernel_widthInvSquared;
    int kernel_count;
    DType kernel_radius;

    int grid_width_dim;  
    int grid_width_offset;
    DType3 grid_width_inv;

    int im_width_dim;
    IndType3 im_width_offset;//used in deapodization

    DType osr;

    int sector_count;
    int sector_width;
    int sector_dim;

    int sector_pad_width;
    int sector_pad_max;
    int sector_offset;
    DType aniso_z_shift;
    DType aniso_z_scale;

    DType radiusSquared;
    DType radiusSquared_inv;
    DType dist_multiplier;

    IndType3 imgDims;
    IndType imgDims_count;

    IndType3 gridDims;
    IndType gridDims_count;

    bool is2Dprocessing;
    int interpolationType;
    int sectorsToProcess;
  };

}

#endif //GPUNUFFT_TYPES_H_INCLUDED
