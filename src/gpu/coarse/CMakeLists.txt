MESSAGE("Coarse GPU Gridding Library")
#add exe to build from gridding3D

if(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_NAME} coarse.cu ../gridding_gpu.cpp ${GRIDDING_SOURCES} ${GRIDDING_INCLUDE} ${GRIDDING_INC_DIR}/cufft_config.hpp)
else(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_NAME} coarse.cu ../gridding_gpu.cpp ${GRIDDING_SOURCES} ${GRIDDING_INCLUDE} ${GRIDDING_INC_DIR}/cufft_config.hpp SHARED)
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(${GRID_LIB_NAME})
CUDA_ADD_CUBLAS_TO_TARGET(${GRID_LIB_NAME})
