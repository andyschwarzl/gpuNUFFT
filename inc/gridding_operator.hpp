#ifndef GRIDDING_OPERATOR_H_INCLUDED
#define GRIDDING_OPERATOR_H_INCLUDED

#include "config.hpp"
#include "gridding_gpu.hpp"

#define DEFAULT_VALUE(a) ((a == 0) ? 1 : a)

namespace GriddingND
{
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
			kernelCount = calculateGrid3KernelSize(osf, kernelWidth/2.0f);
			kernel = (DType*) calloc(kernelCount,sizeof(float));
			loadGrid3Kernel(kernel,(int)kernelCount,(int)kernelWidth,osf);
	
        }

		~GriddingOperator()
		{
        }

		// SETTER 
        void setSectorCount(size_t sectorCount)	{this->sectorCount = sectorCount;}
        void setOsf(DType osf)			{this->osf = osf;}

        void setKspaceCoords(Array<DType> kSpaceCoords)	{this->kSpaceCoords = kSpaceCoords;}
        void setSens(DType2 *sens)		{this->sens = sens;}
        void setDens(DType *dens)		{this->dens = dens;}
        void setSectorCenters(size_t *sectorCenters)	{this->sectorCenters = sectorCenters;}
        void setSectors(size_t *sectors)		{this->sectors = sectors;}

		// GETTER
        Array<DType>  getKspaceCoords()	{return this->kSpaceCoords;}

		DType2* getSens()			{return this->sens;}
        DType*	getDens()			{return this->dens;}

        size_t getKernelWidth()		{return this->kernelWidth;}
        size_t getSectorWidth()		{return this->sectorWidth;}

		// OPERATIONS

		//adjoint gridding
		void performGriddingAdj(Array<DType2> kspaceData, CufftType** imgData);
		void performGriddingAdj(Array<DType2> kspaceData, CufftType** imgData, GriddingOutput griddingOut);

		//forward gridding
		void performForwardGridding(DType2* imgData,  GriddingND::Array<CufftType> kspaceData);
		void performForwardGridding(DType2* imgData,  GriddingND::Array<CufftType> kspaceData, GriddingOutput griddingOut);

	private:
		
        size_t getImgCount() {return kSpaceCoords.count();}
        size_t getGridWidth() {return (size_t)(kSpaceCoords.dim.width * osf);}
        bool applyDensComp(){return this->dens != NULL;}

		// number of sectors
		size_t sectorCount;

		// size of kernel
		size_t kernelCount;

		DType *kernel;

		// simple array
		// dimensions: n dimensions * dataCount
        Array<DType> kSpaceCoords;

		// complex array
		// dimensions: kspaceDim * chnCount
		DType2 *sens;

		// density compensation
		// dimensions: dataCount
		DType *dens;

		// sector centers
		size_t* sectorCenters;

		// sectors 
		// assignment of data index to according sector
		size_t* sectors;

		// oversampling factor
		DType osf;
		

		size_t kernelWidth;

		size_t sectorWidth;
	};
}

#endif //GRIDDING_OPERATOR_H_INCLUDED
