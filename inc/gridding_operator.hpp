#ifndef GRIDDING_OPERATOR_H_INCLUDED
#define GRIDDING_OPERATOR_H_INCLUDED

#include "config.hpp"
#include "gridding_gpu.hpp"

namespace GriddingND
{
    struct Dimension
    {
        Dimension():
            width(0),height(0),depth(0),channels(0),frames(0)
        {}

        size_t width  ;
        size_t height ;
        size_t depth  ;
        size_t channels ;
        size_t frames ;
    };

    template <typename T>
	struct Array
	{
        T* data;
        Dimension dim;

        size_t count()
        {
           return dim.width * dim.height * dim.depth * dim.channels* dim.frames;
        }
	};

	class GriddingOperator 
	{
	public:
		GriddingOperator():
			chnCount(1), frameCount(1)
		{
        }

		GriddingOperator(size_t kernelWidth, size_t sectorWidth, DType osf): 
		chnCount(1), frameCount(1), osf(osf), kernelWidth(kernelWidth), sectorWidth(sectorWidth)
		{
			kernelCount = calculateGrid3KernelSize(osf, kernelWidth/2.0f);
			kernel = (DType*) calloc(kernelCount,sizeof(float));
			loadGrid3Kernel(kernel,(int)kernelCount,(int)kernelWidth,osf);
	
        }

		~GriddingOperator()
		{
        }

		// SETTER 
        void setDataCount(size_t dataCount)	{this->dataCount = dataCount;}
        void setChnCount(size_t chnCount)	{this->chnCount = chnCount;}
        void setFrameCount(size_t frameCount)	{this->frameCount = frameCount;}
        void setSectorCount(size_t sectorCount)	{this->sectorCount = sectorCount;}
        void setOsf(DType osf)			{this->osf = osf;}

        void setKspaceCoords(Array<DType> kSpaceCoords)	{this->kSpaceCoords = kSpaceCoords;}
        void setData(DType2 *data)		{this->data = data;}
        void setSens(DType2 *sens)		{this->sens = sens;}
        void setDens(DType *dens)		{this->dens = dens;}
        void setSectorCenters(size_t *sectorCenters)	{this->sectorCenters = sectorCenters;}
        void setSectors(size_t *sectors)		{this->sectors = sectors;}

		// GETTER
        Array<DType>  getKspaceCoords()	{return this->kSpaceCoords;}
        DType2* getData()			{return this->data;}
        DType2* getSens()			{return this->sens;}
        DType*	getDens()			{return this->dens;}

        size_t getKernelWidth()		{return this->kernelWidth;}
        size_t getSectorWidth()		{return this->sectorWidth;}

		// OPERATIONS

		//adjoint gridding
		void performGriddingAdj(DType2* kspaceData, CufftType** imgData);
		void performGriddingAdj(DType2* kspaceData, CufftType** imgData, GriddingOutput griddingOut);

		//forward gridding
		void performForwardGridding(DType2* imgData, CufftType** kspaceData);
		void performForwardGridding(DType2* imgData, CufftType** kspaceData, GriddingOutput griddingOut);

	private:
		
        size_t getImgCount() {return kSpaceCoords.count();}
        size_t getGridWidth() {return (size_t)(kSpaceCoords.dim.width * osf);}
        bool applyDensComp(){return this->dens != NULL;}

		// n, aka trajectory length
		size_t dataCount;

		// number of channels
		size_t chnCount;

		// number of frames
		size_t frameCount;

		// number of sectors
		size_t sectorCount;

		size_t kernelCount;

		DType *kernel;

		// simple array
		// dimensions: n dimensions * dataCount
        Array<DType> kSpaceCoords;

		// complex array
		// dimensions: dataCount * chnCount * frameCount
		DType2 *data;

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
