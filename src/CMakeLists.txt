MESSAGE("------start creating project-------")

MESSAGE(${CUDA_SDK_ROOT_DIR})

#add exe to build from gridding3D
CUDA_ADD_LIBRARY(gridding3D gridding_gpu.cu griddingFunctions.cpp ../inc/cuda_utils.hpp ../inc/gridding_gpu.hpp ../inc/config.hpp OPTIONS COMPILE_FLAGS -Xcompiler -fPIC)

CUDA_ADD_CUFFT_TO_TARGET(gridding3D)
CUDA_ADD_CUBLAS_TO_TARGET(gridding3D)

ADD_SUBDIRECTORY(matlab)
