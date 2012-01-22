MESSAGE("------start creating project-------")

MESSAGE(${CUDA_SDK_ROOT_DIR})

find_library(CUFFT_LIB cufft)
find_library(CUBLAS_LIB cublas)

#add exe to build from gridding3D
CUDA_ADD_EXECUTABLE(gridding3D gridding3D.c)

target_link_libraries(gridding3D ${CUFFT_LIB})
target_link_libraries(gridding3D ${CUBLAS_LIB})

