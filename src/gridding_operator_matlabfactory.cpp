
#include "gridding_operator_matlabfactory.hpp"
#include <iostream>

GriddingND::GriddingOperatorMatlabFactory GriddingND::GriddingOperatorMatlabFactory::instance;

GriddingND::GriddingOperatorMatlabFactory& GriddingND::GriddingOperatorMatlabFactory::getInstance()
{
	return instance;
}

