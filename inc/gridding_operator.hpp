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

		Dimensions operator*(const DType alpha)
		{
			Dimensions res;
			res.width = (size_t)((*this).width * alpha);
			res.height = (size_t)((*this).height * alpha);
			res.depth = (size_t)((*this).depth * alpha);
			res.length = (size_t)((*this).length * alpha);
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
			free(this->dataIndices.data);
			free(this->kSpaceCoords.data);
			free(this->sectorCenters.data);
        }

		// SETTER 
        void setOsf(DType osf)			{this->osf = osf;}

        void setKspaceCoords(Array<DType> kSpaceCoords)	{this->kSpaceCoords = kSpaceCoords;}
        void setSens(Array<DType2> sens)		{this->sens = sens;}
		void setDens(Array<DType> dens)		{this->dens = dens;}
        void setSectorCenters(Array<IndType3> sectorCenters)	{this->sectorCenters = sectorCenters;}
        void setSectorDataCount(Array<IndType> sectorDataCount)		{this->sectorDataCount = sectorDataCount;}
		void setDataIndices(Array<IndType> dataIndices)		{this->dataIndices = dataIndices;}
		void setImageDims(Dimensions dims)  {this->imgDims = dims;}
		void setSectorDims(Dimensions dims)  {this->sectorDims = dims;}

		// GETTER
        Array<DType>  getKspaceCoords()	{return this->kSpaceCoords;}

		Array<DType2>	getSens()			{return this->sens;}
        Array<DType>	getDens()			{return this->dens;}
		Array<DType>    getKernel()			{return this->kernel;}
		Array<IndType>  getSectorDataCount(){return this->sectorDataCount;}

        size_t getKernelWidth()		{return this->kernelWidth;}
        size_t getSectorWidth()		{return this->sectorWidth;}
		
		Dimensions getImageDims() {return this->imgDims;}
		Dimensions getGridDims() {return this->imgDims * osf;}

		Dimensions getSectorDims() {return this->sectorDims;}

		Array<IndType3> getSectorCenters()	{return this->sectorCenters; }

		Array<IndType>  getDataIndices()		{return this->dataIndices;}

		// OPERATIONS

		//adjoint gridding
		Array<CufftType> performGriddingAdj(Array<DType2> kspaceData);
		void             performGriddingAdj(Array<DType2> kspaceData, Array<CufftType>& imgData, GriddingOutput griddingOut = DEAPODIZATION);
		Array<CufftType> performGriddingAdj(Array<DType2> kspaceData, GriddingOutput griddingOut);

		//forward gridding
		Array<CufftType> performForwardGridding(Array<DType2> imgData);
		void             performForwardGridding(Array<DType2> imgData,Array<CufftType>& kspaceData, GriddingOutput griddingOut = DEAPODIZATION);
		Array<CufftType> performForwardGridding(Array<DType2> imgData,GriddingOutput griddingOut);

	private:
		void initKernel()
		{
			this->kernel.dim.length = calculateGrid3KernelSize(osf, kernelWidth/2.0f);
			this->kernel.data = (DType*) calloc(this->kernel.count(),sizeof(DType));
			loadGrid3Kernel(this->kernel.data,(int)this->kernel.count(),(int)kernelWidth,osf);
		}

		size_t getGridWidth() {return (size_t)(this->getGridDims().width);}
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

		Dimensions imgDims;

		Dimensions sectorDims;

		template <typename T>
		T* selectOrdered(Array<T> dataArray);
	};
}

#endif //GRIDDING_OPERATOR_H_INCLUDED
