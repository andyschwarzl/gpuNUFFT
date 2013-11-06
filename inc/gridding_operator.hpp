#ifndef GRIDDING_OPERATOR_H_INCLUDED
#define GRIDDING_OPERATOR_H_INCLUDED

#include "config.hpp"
#include "gridding_gpu.hpp"
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

        size_t width  ;
        size_t height ;
        size_t depth  ;
		
		size_t length; //1D case 

        size_t channels ;
        size_t frames ;

        size_t count()
        {
           return DEFAULT_VALUE(length) * DEFAULT_VALUE(width) * DEFAULT_VALUE(height) * DEFAULT_VALUE(depth) * DEFAULT_VALUE(channels) * DEFAULT_VALUE(frames);
        }

		Dimensions operator*(DType alpha)
		{
			(*this).width = (size_t)((*this).width * alpha);
			(*this).height = (size_t)((*this).height * alpha);
			(*this).depth = (size_t)((*this).depth * alpha);
			(*this).length = (size_t)((*this).length * alpha);
			return (*this);
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

		size_t count()
		{
			return dim.count();
		}

	};

	class GriddingOperator 
	{
		public:

		GriddingOperator()
		{
        }

		GriddingOperator(size_t kernelWidth, size_t sectorWidth, DType osf): 
		osf(osf), kernelWidth(kernelWidth), sectorWidth(sectorWidth)
		{
			initKernel();	
        }

		~GriddingOperator()
		{
			free(this->kernel.data);
        }

		// SETTER 
        void setOsf(DType osf)			{this->osf = osf;}

        void setKspaceCoords(Array<DType> kSpaceCoords)	{this->kSpaceCoords = kSpaceCoords;}
        void setSens(Array<DType2> sens)		{this->sens = sens;}
		void setDens(Array<DType> dens)		{this->dens = dens;}
        void setSectorCenters(Array<IndType3> sectorCenters)	{this->sectorCenters = sectorCenters;}
        void setSectorDataCount(Array<IndType> sectorDataCount)		{this->sectorDataCount = sectorDataCount;}
		void setDataIndices(Array<IndType> dataIndices)		{this->dataIndices = dataIndices;}
		void setGridDims(Dimensions dims)  {this->gridDims = dims;}
		void setSectorDims(Dimensions dims)  {this->sectorDims = dims;}

		// GETTER
        Array<DType>  getKspaceCoords()	{return this->kSpaceCoords;}

		Array<DType2>	getSens()			{return this->sens;}
        Array<DType>	getDens()			{return this->dens;}
		Array<DType>    getKernel()			{return this->kernel;}
		Array<IndType>  getSectorDataCount(){return this->sectorDataCount;}

        size_t getKernelWidth()		{return this->kernelWidth;}
        size_t getSectorWidth()		{return this->sectorWidth;}
		Dimensions getGridDims() {return this->gridDims;}

		Dimensions getSectorDims() {return this->sectorDims;}

		Array<IndType3> getSectorCenters()	{return this->sectorCenters; }
		Array<IndType>  getDataIndices()		{return this->dataIndices;}


		// OPERATIONS

		//adjoint gridding
		void performGriddingAdj(Array<DType2> kspaceData, Array<CufftType> imgData);
		void performGriddingAdj(Array<DType2> kspaceData, Array<CufftType> imgData, GriddingOutput griddingOut);

		//forward gridding
		void performForwardGridding(Array<DType2> imgData,  GriddingND::Array<CufftType> kspaceData);
		void performForwardGridding(Array<DType2> imgData,  GriddingND::Array<CufftType> kspaceData, GriddingOutput griddingOut);

	private:
		void initKernel()
		{
			this->kernel.dim.length = calculateGrid3KernelSize(osf, kernelWidth/2.0f);
			this->kernel.data = (DType*) calloc(this->kernel.count(),sizeof(DType));
			loadGrid3Kernel(this->kernel.data,(int)this->kernel.count(),(int)kernelWidth,osf);
		}

        size_t getGridWidth() {return (size_t)(kSpaceCoords.dim.width * osf);}
        bool applyDensComp(){return this->dens.data != NULL;}

		Array<DType> kernel;

		// simple array
		// dimensions: n dimensions * dataCount
        Array<DType> kSpaceCoords;

		// complex array
		// dimensions: kspaceDim * chnCount
		Array<DType2> sens;

		// density compensation
		// dimensions: dataCount
		Array<DType> dens;

		// sector centers
		Array<IndType3> sectorCenters;

		// dataCount per sector
		Array<IndType> sectorDataCount;

		// assignment of data index to according sector
		Array<IndType> dataIndices;

		// oversampling factor
		DType osf;
		
		// width of kernel in grid units
		size_t kernelWidth;

		// sector size in grid units
		size_t sectorWidth;

		Dimensions gridDims;

		Dimensions sectorDims;
	};
}

#endif //GRIDDING_OPERATOR_H_INCLUDED
