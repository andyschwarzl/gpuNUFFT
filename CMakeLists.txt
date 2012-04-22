#The name of the project
PROJECT(gridding3D)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)

#for older CMAKE versions these vars have to be set
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(ARCHIVE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#Searching CUDA
FIND_PACKAGE(CUDA)

#Searching MATLAB
FIND_PACKAGE(Matlab)

# For make-based builds, defines make target named test.
# For Visual Studio builds, defines Visual Studio project named RUN_TESTS.
enable_testing()

#Options
#General DEBUG output 
#Matlab DEBUG output
SET (MATLAB_DEBUG false)
OPTION(WITH_MATLAB_DEBUG "Enable DEBUG messages for MATLAB calls" OFF)
if (WITH_MATLAB_DEBUG)
 SET (MATLAB_DEBUG true)
endif()
MESSAGE(STATUS "Setting MATLAB DEBUG Option to ${MATLAB_DEBUG}")

#Enable/Disable GPU double precision 
SET(GPU_DOUBLE_PREC OFF CACHE BOOL "Enable double precision floating point operations on GPU")

if(GPU_DOUBLE_PREC)
  SET(CUDA_NVCC_FLAGS "-arch;sm_20")
else(GPU_DOUBLE_PREC)
  SET(CUDA_NVCC_FLAGS "-arch;sm_11")
endif(GPU_DOUBLE_PREC)

MESSAGE("setting NVCC FLAGS to: ${CUDA_NVCC_FLAGS}")

CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/inc/config.hpp.cmake ${CMAKE_SOURCE_DIR}/inc/config.hpp)

#Include dirs
include_directories (inc)
SET(GRIDDING_INC_DIR ${CMAKE_SOURCE_DIR}/inc)
SET(GRIDDING_INCLUDE ${GRIDDING_INC_DIR}/cuda_utils.hpp 
					 ${GRIDDING_INC_DIR}/cuda_utils.cuh 
					 ${GRIDDING_INC_DIR}/gridding_gpu.hpp 
					 ${GRIDDING_INC_DIR}/config.hpp)

#Adding src directory to the project
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(src)

add_test(
    NAME runUnitTests
    COMMAND runUnitTests
)

add_test(
    NAME runGPUUnitTests
    COMMAND runGPUUnitTests
)

if(GPU_DOUBLE_PREC)
	add_test(
    NAME runGPUATMUnitTests
    COMMAND runGPUATMUnitTests)
endif()
