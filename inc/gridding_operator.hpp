#ifndef GRIDDING_OPERATOR_H_INCLUDED
#define GRIDDING_OPERATOR_H_INCLUDED

#include "config.hpp"
#include "gridding_gpu.hpp"

namespace GriddingND
{
	struct KSpaceDim
	{
		size_t width;
		size_t height;
		size_t depth;
	};

	class GriddingOperator 
	{
	public:
		GriddingOperator():
			chnCount(1), frameCount(1)
		{
		};


		GriddingOperator(size_t kernelWidth, size_t sectorWidth, DType osf): 
		chnCount(1), frameCount(1), osf(osf), kernelWidth(kernelWidth), sectorWidth(sectorWidth)
		{
			kernelCount = calculateGrid3KernelSize(osf, kernelWidth/2.0f);
			kernel = (DType*) calloc(kernelCount,sizeof(float));
			loadGrid3Kernel(kernel,(int)kernelCount,(int)kernelWidth,osf);
	
		};

		~GriddingOperator()
		{
		};

		// SETTER 
		void setDataCount(size_t dataCount)		{this->dataCount = dataCount;};
		void setChnCount(size_t chnCount)		{this->chnCount = chnCount;};
		void setFrameCount(size_t frameCount)	{this->frameCount = frameCount;};
		void setSectorCount(size_t sectorCount)	{this->sectorCount = sectorCount;};
		void setOsf(DType osf)					{this->osf = osf;};

		void setKspaceCoords(DType *kspaceCoords)	{this->kspaceCoords = kspaceCoords;};
		void setData(DType2 *data)					{this->data = data;};
		void setSens(DType2 *sens)					{this->sens = sens;};
		void setDens(DType *dens)					{this->dens = dens;};
		void setSectorCenters(size_t *sectorCenters)					{this->sectorCenters = sectorCenters;};
		void setSectors(size_t *sectors)					{this->sectors = sectors;};

		void setKSpaceWidth(size_t width)	{this->kspaceDim.width = width;};
		void setKSpaceHeight(size_t height)	{this->kspaceDim.height = height;};
		void setKSpaceDepth(size_t depth)	{this->kspaceDim.depth = depth;};

		// GETTER
		DType*	getKspaceCoords()	{return this->kspaceCoords;};
		DType2* getData()			{return this->data;};
		DType2* getSens()			{return this->sens;};
		DType*	getDens()			{return this->dens;};

		// OPERATIONS

		//adjoint gridding
		void performGriddingAdj(CufftType** imgData);
		void performGriddingAdj(CufftType** imgData, GriddingOutput griddingOut);

	private:
		
		size_t getImgCount() {return kspaceDim.width * kspaceDim.height * kspaceDim.depth;};
		size_t getGridWidth() {return (size_t)(kspaceDim.width * osf);};
		bool applyDensComp(){return this->dens != NULL;};

		// x y [z], image dimension
		KSpaceDim kspaceDim;

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
		DType  *kspaceCoords;

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