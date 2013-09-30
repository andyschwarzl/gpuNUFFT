#ifndef GRIDDING_OPERATOR_H_INCLUDED
#define GRIDDING_OPERATOR_H_INCLUDED

#include "config.hpp"

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

		GriddingOperator()
		{
			this->chnCount = 1;
			this->frameCount = 1;
		};

		~GriddingOperator()
		{
		};

		void performGriddingAdj();

		void setDataCount(size_t dataCount){this->dataCount = dataCount;};
		void setChnCount(size_t chnCount){this->chnCount = chnCount;};
		void setFrameCount(size_t frameCount){this->frameCount = frameCount;};

		void setKspaceCoords(DType *kspaceCoords){this->kspaceCoords = kspaceCoords;};
		void setData(DType2 *data){this->data = data;};
		void setSens(DType2 *sens){this->sens = sens;};
		void setDens(DType *dens){this->dens = dens;};

		void setKSpaceWidth(size_t width){this->kspaceDim.width = width;};
		void setKSpaceHeight(size_t height){this->kspaceDim.height = height;};
		void setKSpaceDepth(size_t depth){this->kspaceDim.depth = depth;};

	private:
		
		// x y [z], image dimension
		KSpaceDim kspaceDim;

		// n, aka trajectory length
		size_t dataCount;

		// number of channels
		size_t chnCount;

		// number of frames
		size_t frameCount;

		// simple array
		// dimensions: n dimensions * dataCount
		DType  *kspaceCoords;

		// complex array
		// dimensions: dataCount * chnCount * frameCount
		DType2 *data;

		// complex array
		// dimensions: kspaceDim * chnCount
		DType2 *sens;

		// simple
		DType *dens;
		
	};
}

#endif //GRIDDING_OPERATOR_H_INCLUDED