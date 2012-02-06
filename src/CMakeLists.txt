MESSAGE("------start creating project-------")

MESSAGE(${CUDA_SDK_ROOT_DIR})

#add exe to build from gridding3D
CUDA_ADD_EXECUTABLE(gridding3D gridding3D.cu griddingFunctions.cc)

CUDA_ADD_CUFFT_TO_TARGET(gridding3D)
CUDA_ADD_CUBLAS_TO_TARGET(gridding3D)

