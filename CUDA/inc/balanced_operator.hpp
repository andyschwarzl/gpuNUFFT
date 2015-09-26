#ifndef BALANCED_OPERATOR_H_INCLUDED
#define BALANCED_OPERATOR_H_INCLUDED

#include "gpuNUFFT_types.hpp"

namespace gpuNUFFT
{
/**
  * \brief Interface defined for balanced gpuNUFFT Operators
  *
  * Adds sector processing order getter and setter.
  */
class BalancedOperator
{
 public:
  // Getter and Setter for Processing Order
  virtual Array<IndType2> getSectorProcessingOrder() = 0;
  virtual void
  setSectorProcessingOrder(Array<IndType2> sectorProcessingOrder) = 0;
};
}
#endif  // BALANCED_OPERATOR_H_INCLUDED
