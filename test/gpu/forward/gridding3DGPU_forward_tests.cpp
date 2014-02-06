#include <limits.h>

#include "gtest/gtest.h"

#include "gridding_operator_factory.hpp"

#define epsilon 0.0001f

#define get3DC2lin(_x,_y,_z,_width) ((_x) + (_width) * ( (_y) + (_z) * (_width)))

TEST(TestGPUGriddingForwardConv,KernelCall1Sector)
{
	int kernel_width = 5;

	//oversampling ratio
	float osr = 1.25f;

	//Image
	int im_width = 32;

	//Data
	int data_entries = 1;

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	coords[0] = -0.31719f; //should result in 7,7,7 center
	coords[1] = -0.38650f;
	coords[2] = 0;

	DType2* im_data;
	unsigned long dims_g[4];
    dims_g[0] = 2; /* complex */
	dims_g[1] = (unsigned long)(im_width); 
    dims_g[2] = (unsigned long)(im_width);
    dims_g[3] = (unsigned long)(im_width);

	long im_size = dims_g[1]*dims_g[2]*dims_g[3];

	im_data = (DType2*) calloc(im_size,sizeof(DType2));
	
	for (int x=0;x<im_size;x++)
	{
		im_data[x].x = 1.0f;
		im_data[x].y = 1.0f;
	}
	
	//sectors of data, count and start indices
	int sector_width = 8;
	
    GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Array<CufftType> dataArray;

	GriddingND::Array<DType2> im_dataArray;
	im_dataArray.data = im_data;
	im_dataArray.dim.width = im_width;
	im_dataArray.dim.height = im_width;
	im_dataArray.dim.depth = im_width;

	GriddingND::Dimensions imgDims;
	imgDims.width = im_width;
	imgDims.height = im_width;
	imgDims.depth = im_width;

    //GriddingND::GriddingOperator *griddingOp = new GriddingND::GriddingOperator(kernel_width,sector_width,osr);
    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	dataArray = griddingOp->performForwardGridding(im_dataArray);

	for (int j=0; j<dataArray.count(); j++)
	{
		printf("%.4f %.4f \n",dataArray.data[j].x,dataArray.data[j].y);
	}
	EXPECT_NEAR(0.158092f, dataArray.data[0].x,epsilon);

	printf("\n");

	free(coords);
	free(im_data);
	free(dataArray.data);

	delete griddingOp;
}

TEST(TestGPUGriddingForwardConv,KernelCall1Sector2Channels)
{
	int kernel_width = 5;
	float osr = 1.25f;
	int im_width = 32;

	//Data
	int data_entries = 1;

	//Coords
	//Scaled between -0.5 and 0.5
	//in triplets (x,y,z)
    DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
	coords[0] = -0.31719f; //should result in 7,7,7 center
	coords[1] = -0.38650f;
	coords[2] = 0;

	GriddingND::Array<DType2> im_dataArray;
	im_dataArray.data = NULL; 
	im_dataArray.dim.width = im_width;
	im_dataArray.dim.height = im_width;
	im_dataArray.dim.depth = im_width;
	im_dataArray.dim.channels = 2;

	im_dataArray.data = (DType2*) calloc(im_dataArray.count(),sizeof(DType2));
	
	for (int x=0;x<im_dataArray.count();x++)
	{
		im_dataArray.data[x].x = 1.0f;
		im_dataArray.data[x].y = 1.0f;
	}
	
	//sectors of data, count and start indices
	int sector_width = 8;

    GriddingND::Array<DType> kSpaceData;
    kSpaceData.data = coords;
    kSpaceData.dim.length = data_entries;

	GriddingND::Array<CufftType> dataArray;

	//imgDims -> no Channel size here
	GriddingND::Dimensions imgDims;
	imgDims.width = im_dataArray.dim.width;
    imgDims.height = im_dataArray.dim.height;
	imgDims.depth  = im_dataArray.dim.depth;

    GriddingND::GriddingOperator *griddingOp = GriddingND::GriddingOperatorFactory::getInstance().createGriddingOperator(kSpaceData,kernel_width,sector_width,osr,imgDims);

	dataArray = griddingOp->performForwardGridding(im_dataArray);

	std::cout << "result count : " << dataArray.count() << std::endl;

	for (int j=0; j<dataArray.count(); j++)
	{
		printf("%.4f %.4f \n",dataArray.data[j].x,dataArray.data[j].y);
	}
	EXPECT_NEAR(0.158092f, dataArray.data[0].x,epsilon);
	EXPECT_EQ(dataArray.data[1].x, dataArray.data[0].x);
	EXPECT_EQ(dataArray.data[1].y, dataArray.data[0].y);

	printf("\n");

	free(coords);
	free(im_dataArray.data);
	delete griddingOp;
}

