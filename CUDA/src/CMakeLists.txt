MESSAGE("------start creating libraries-------")

SET(GPUNUFFT_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
SET(GPUNUFFT_SOURCES ${GPUNUFFT_SRC_DIR}/gpuNUFFT_utils.cpp
                     ${GPUNUFFT_SRC_DIR}/gpuNUFFT_operator_factory.cpp
                     ${GPUNUFFT_SRC_DIR}/gpuNUFFT_operator.cpp
										 ${GPUNUFFT_SRC_DIR}/texture_gpuNUFFT_operator.cpp
										 ${GPUNUFFT_SRC_DIR}/balanced_gpuNUFFT_operator.cpp
										 ${GPUNUFFT_SRC_DIR}/balanced_texture_gpuNUFFT_operator.cpp)

ADD_SUBDIRECTORY(gpu)

#cpu not necessary
#ADD_SUBDIRECTORY(cpu)
