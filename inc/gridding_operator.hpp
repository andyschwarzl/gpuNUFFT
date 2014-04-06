#ifndef GRIDDING_OPERATOR_H_INCLUDED
#define GRIDDING_OPERATOR_H_INCLUDED

#include "config.hpp"
#include <cstdlib>
#include <iostream>

#define DEFAULT_VALUE(a) ((a == 0) ? 1 : a)

namespace GriddingND
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

  enum GriddingOutput
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

  struct GriddingInfo 
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

    DType radiusSquared;
    DType dist_multiplier;

    IndType3 imgDims;
    IndType imgDims_count;

    IndType3 gridDims;
    IndType gridDims_count;

    bool is2Dprocessing;
  };


  class GriddingOperator 
  {
  public:

    GriddingOperator()
    {
    }

    GriddingOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims): 
    osf(osf), kernelWidth(kernelWidth), sectorWidth(sectorWidth),imgDims(imgDims)
    {
      initKernel();	

      sectorDims.width = sectorWidth;
      sectorDims.height = sectorWidth;
      if (imgDims.depth > 0)
        sectorDims.depth = sectorWidth;
    }

    ~GriddingOperator()
    {
      free(this->kernel.data);
    }

    friend class GriddingOperatorFactory;

    // SETTER 
    void setOsf(DType osf)			{this->osf = osf;}

    void setKSpaceTraj(Array<DType> kSpaceTraj)				{this->kSpaceTraj = kSpaceTraj;}
    void setSectorCenters(Array<IndType> sectorCenters)		{this->sectorCenters = sectorCenters;}
    void setSectorDataCount(Array<IndType> sectorDataCount)	{this->sectorDataCount = sectorDataCount;}
    void setDataIndices(Array<IndType> dataIndices)			{this->dataIndices = dataIndices;}
    void setSens(Array<DType2> sens)						{this->sens = sens;}
    void setDens(Array<DType> dens)							{this->dens = dens;}

    void setImageDims(Dimensions dims)						{this->imgDims = dims;}
    void setGridSectorDims(Dimensions dims)						{this->gridSectorDims = dims;}

    // GETTER
    Array<DType>  getKSpaceTraj()	{return this->kSpaceTraj;}

    Array<DType2>	getSens()			{return this->sens;}
    Array<DType>	getDens()			{return this->dens;}
    Array<DType>    getKernel()			{return this->kernel;}
    Array<IndType>  getSectorDataCount(){return this->sectorDataCount;}

    IndType getKernelWidth()		{return this->kernelWidth;}
    IndType getSectorWidth()		{return this->sectorWidth;}

    Dimensions getImageDims() {return this->imgDims;}
    Dimensions getGridDims() {return this->imgDims * osf;}

    Dimensions getGridSectorDims() {return this->gridSectorDims;}
    Dimensions getSectorDims() {return this->sectorDims;}

    Array<IndType> getSectorCenters() {return this->sectorCenters;}
    IndType* getSectorCentersData() {return reinterpret_cast<IndType*>(this->sectorCenters.data);}

    Array<IndType>  getDataIndices()		{return this->dataIndices;}

    bool is2DProcessing() {return this->imgDims.depth == 0;}
    bool is3DProcessing() {return !is2DProcessing();}

    int getImageDimensionCount() {return (is2DProcessing() ? 2 : 3);}

    // OPERATIONS

    //adjoint gridding
    Array<CufftType> performGriddingAdj(Array<DType2> kspaceData);
    void             performGriddingAdj(Array<DType2> kspaceData, Array<CufftType>& imgData, GriddingOutput griddingOut = DEAPODIZATION);
    Array<CufftType> performGriddingAdj(Array<DType2> kspaceData, GriddingOutput griddingOut);

    //forward gridding
    Array<CufftType> performForwardGridding(Array<DType2> imgData);
    void             performForwardGridding(Array<DType2> imgData,Array<CufftType>& kspaceData, GriddingOutput griddingOut = DEAPODIZATION);
    Array<CufftType> performForwardGridding(Array<DType2> imgData,GriddingOutput griddingOut);

    bool applyDensComp(){return (this->dens.data != NULL && this->dens.count()>1);}

  private:
    IndType getGridWidth() {return (IndType)(this->getGridDims().width);}

    Array<DType> kernel;

    // simple array
    // dimensions: n dimensions * dataCount
    Array<DType> kSpaceTraj;

    // complex array
    // dimensions: kspaceDim * chnCount
    Array<DType2> sens;

    // density compensation
    // dimensions: dataCount
    Array<DType> dens;

    // sector centers
    Array<IndType> sectorCenters;

    // dataCount per sector
    Array<IndType> sectorDataCount;

    // assignment of data index to according sector
    Array<IndType> dataIndices;

    // oversampling factor
    DType osf;

    // width of kernel in grid units
    IndType kernelWidth;

    // sector size in grid units
    IndType sectorWidth;

    Dimensions imgDims;

    // amount of sectors per grid direction
    Dimensions gridSectorDims;

    // size of one sector in grid
    Dimensions sectorDims;

    template <typename T>
    T* selectOrdered(Array<T>& dataArray, int offset=0);

    template <typename T>
    void writeOrdered(Array<T>& destArray, T* sortedArray, int offset=0);

    void initKernel();

    GriddingInfo* initAndCopyGriddingInfo();
  };
}

#endif //GRIDDING_OPERATOR_H_INCLUDED
