#The name of the project
PROJECT(gridding3D)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#Searching CUDA
FIND_PACKAGE(CUDA)

#Searching MATLAB
FIND_PACKAGE(MATLAB)

#Options
set (MATLAB_DEBUG FALSE)
OPTION(WITH_MATLAB_DEBUG "Enable DEBUG messages for MATLAB calls" OFF)
if (WITH_MATLAB_DEBUG)
 set (MATLAB_DEBUG TRUE)
endif()
MESSAGE(STATUS "Setting MATLAB DEBUG Option to ${MATLAB_DEBUG}")

set(GPU_DOUBLE_PREC OFF CACHE BOOL "Enable double precision floating point operations on GPU")

if(GPU_DOUBLE_PREC)
  SET(CUDA_NVCC_FLAGS "-arch;sm_20")
else(GPU_DOUBLE_PREC)
  SET(CUDA_NVCC_FLAGS "-arch;sm_11")
endif(GPU_DOUBLE_PREC)

MESSAGE("setting NVCC FLAGS to: ${CUDA_NVCC_FLAGS}")

CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/inc/config.hpp.cmake ${CMAKE_SOURCE_DIR}/inc/config.hpp)

#Include dirs
include_directories (inc)

#Adding src directory to the project
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(src)

# For make-based builds, defines make target named test.
# For Visual Studio builds, defines Visual Studio project named RUN_TESTS.
enable_testing()

add_test(
    NAME runGPUUnitTests
    COMMAND runGPUUnitTests
)

add_test(
    NAME runUnitTests
    COMMAND runUnitTests
)