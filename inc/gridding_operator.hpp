#ifndef GRIDDING_OPERATOR_H_INCLUDED
#define GRIDDING_OPERATOR_H_INCLUDED

#include "gridding_types.hpp"
#include "config.hpp"
#include <cstdlib>
#include <iostream>

namespace GriddingND
{
  class GriddingOperator 
  {
  public:

    GriddingOperator():
        interpolationType(1)
    {
    }

    GriddingOperator(IndType kernelWidth, IndType sectorWidth, DType osf, Dimensions imgDims): 
    osf(osf), kernelWidth(kernelWidth), sectorWidth(sectorWidth),imgDims(imgDims),interpolationType(1)
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

    int interpolationType;

    const char* getInterpolationTypeName();
  };
}

#endif //GRIDDING_OPERATOR_H_INCLUDED
