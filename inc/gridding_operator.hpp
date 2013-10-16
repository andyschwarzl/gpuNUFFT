#ifndef GRIDDING_OPERATOR_H_INCLUDED
#define GRIDDING_OPERATOR_H_INCLUDED

#include "config.hpp"
#include "gridding_gpu.hpp"
#include <iostream>

#define DEFAULT_VALUE(a) ((a == 0) ? 1 : a)

namespace GriddingND
{
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
        void setSectorCenters(Array<size_t> sectorCenters)	{this->sectorCenters = sectorCenters;}
        void setSectors(Array<size_t> sectors)		{this->sectors = sectors;}

		// GETTER
        Array<DType>  getKspaceCoords()	{return this->kSpaceCoords;}

		Array<DType2>	getSens()			{return this->sens;}
        Array<DType>	getDens()			{return this->dens;}
		Array<DType>    getKernel()			{return this->kernel;}

        size_t getKernelWidth()		{return this->kernelWidth;}
        size_t getSectorWidth()		{return this->sectorWidth;}

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
		Array<size_t> sectorCenters;

		// sectors 
		// assignment of data index to according sector
		Array<size_t> sectors;

		// oversampling factor
		DType osf;
		
		// width of kernel in grid units
		size_t kernelWidth;

		// sector size in grid units
		size_t sectorWidth;
	};
}

#endif //GRIDDING_OPERATOR_H_INCLUDED
