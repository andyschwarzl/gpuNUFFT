#The name of the project
PROJECT(gpuNUFFT)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)

#for older CMAKE versions these vars have to be set
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(ARCHIVE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#patch output path
SET(PATCH_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/deploy)
SET(PATCH_OUTPUT_PATH ${PATCH_OUTPUT_DIR}/gpuNUFFT.zip)

#mex file export dir
SET(MEX_EXPORT_DIR ${PROJECT_SOURCE_DIR}/../gpuNUFFT/@gpuNUFFT/private CACHE STRING "Folder in which mex files are exported to.")

#Searching CUDA
FIND_PACKAGE(CUDA REQUIRED)

if(WIN32)
  set(CMAKE_CXX_FLAGS "-std=c++11")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC")
endif(WIN32)

#Enable Mex File Generation
#Searching MATLAB
SET(GEN_MEX_FILES OFF CACHE BOOL "Enable generation of Matlab MEX files.")

if(GEN_MEX_FILES)
  if(WIN32)
    SET(MATLAB_ROOT_DIR "C:\\Program Files\\MATLAB\\R2011a" CACHE STRING "MATLAB Installation Directory")
  elseif(UNIX AND APPLE)
    SET(MATLAB_ROOT_DIR "/Applications/MATLAB_R2010b.app" CACHE STRING "MATLAB Installation Directory")
  elseif(UNIX)
    SET(MATLAB_ROOT_DIR "/usr/local/MATLAB/R2010b" CACHE STRING "MATLAB Installation Directory")
  endif()

  SET(MATLAB_DIR = "${MATLAB_ROOT_DIR}")

  FIND_PACKAGE(Matlab)

  if (Matlab_FOUND)
    MESSAGE(STATUS "MATLAB Installation found via script variable Matlab_ROOT_DIR in ${Matlab_ROOT_DIR}")
    SET(MATLAB_ROOT_DIR ${Matlab_ROOT_DIR})
  endif()

  IF (EXISTS "${MATLAB_ROOT_DIR}")
    MESSAGE(STATUS "MATLAB Installation found in ${MATLAB_ROOT_DIR}")
  else()
    MESSAGE(FATAL_ERROR "Please set variable MATLAB_ROOT_DIR correctly! Current value: ${MATLAB_ROOT_DIR}")
  endif()

  #Matlab DEBUG output
  SET (MATLAB_DEBUG false)
  OPTION(WITH_MATLAB_DEBUG "Enable DEBUG messages for MATLAB calls" OFF)
  if (WITH_MATLAB_DEBUG)
   SET (MATLAB_DEBUG true)
  endif()
  MESSAGE(STATUS "Setting MATLAB DEBUG Option to ${MATLAB_DEBUG}")
endif(GEN_MEX_FILES)

#Options
#General DEBUG output 
SET (DEBUG false)
OPTION(WITH_DEBUG "Enable DEBUG messages" OFF)
if (WITH_DEBUG)
 SET (DEBUG true)
endif()
MESSAGE(STATUS "Setting DEBUG Option to ${DEBUG}")

#Enable Google Tests
SET(GEN_TESTS OFF CACHE BOOL "Enable and generate simple GOOGLE test framework unit tests")

#Enable/Disable GPU double precision 
SET(GPU_DOUBLE_PREC OFF CACHE BOOL "Enable double precision floating point operations on GPU (Compute Capability 1.3 needed)")

#Option to enable python bindings
SET(GEN_PYTHON_FILES ON CACHE BOOL "Enable generation of Python binding files.")

if(GPU_DOUBLE_PREC)
  SET(PREC_SUFFIX "_d")
else(GPU_DOUBLE_PREC)
  SET(PREC_SUFFIX "_f")
endif(GPU_DOUBLE_PREC)


set(MY_NVCC_FLAGS -I${CUDA_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS -I${CUDA_INCLUDE_DIRS})

  if(${CUDA_VERSION_MAJOR} GREATER 9)
    message("add CUDA 10 flags: Current version: ${CUDA_VERSION}")
    list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75)
  endif()
  
  # Add support for Ada Lovelace (RTX 40-series / RTX Ada Generation)
  if(${CUDA_VERSION_MAJOR} GREATER 11)
      message("add CUDA 12 flags: Current version: ${CUDA_VERSION}")
      list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_89,code=sm_89)
  endif()

  # --- Ampere (RTX 30-series / A100) ---
  if(${CUDA_VERSION_MAJOR} GREATER 10)
      list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_80,code=sm_80)
      list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_86,code=sm_86)
  endif()
  
  # --- Ada Lovelace (Your RTX 6000 Ada) & Hopper (H100) ---
  if(${CUDA_VERSION_MAJOR} GREATER 11)
      # Your current GPU (Ada)
      list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_89,code=sm_89)
      # Hopper (Data Center H100)
      list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_90,code=sm_90)
  endif()
  
  # --- Blackwell (RTX 50-series / B200) ---
  # Note: Requires CUDA 12.8 or 13.0+
  if(${CUDA_VERSION} VERSION_GREATER_EQUAL "12.8")
      message("Adding Blackwell support")
      list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_100,code=sm_100)
      list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_120,code=sm_120)
  endif()


IF(CMAKE_BUILD_TYPE MATCHES Debug)
  MESSAGE("debug mode")
  list(APPEND CUDA_NVCC_FLAGS ${MY_NVCC_FLAGS} --ptxas-options=-v -G)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -std=c++11")
ELSE(CMAKE_BUILD_TYPE)
  list(APPEND CUDA_NVCC_FLAGS ${MY_NVCC_FLAGS})
ENDIF()

MESSAGE(STATUS "setting NVCC FLAGS to: ${CUDA_NVCC_FLAGS}")

#LIB and MEX-file names
SET(GRID_LIB_ATM_NAME "gpuNUFFT_ATM${PREC_SUFFIX}")

SET(GRID_MEX_FORW_ATM_NAME "mex_gpuNUFFT_forw_atomic${PREC_SUFFIX}")
SET(GRID_MEX_ADJ_ATM_NAME "mex_gpuNUFFT_adj_atomic${PREC_SUFFIX}")

#Precomputation MEX file
SET(GRID_MEX_PRECOMP_NAME "mex_gpuNUFFT_precomp${PREC_SUFFIX}")

MESSAGE(STATUS "creating lib with name: ${GRID_LIB_ATM_NAME}")
SET(WARNING "/* WARNING: Automatically generated file. Please do not modify this file. */")
CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/inc/config.hpp.cmake ${CMAKE_SOURCE_DIR}/inc/config.hpp)
CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/inc/cufft_config.hpp.cmake ${CMAKE_SOURCE_DIR}/inc/cufft_config.hpp)

#Include dirs
include_directories(inc)
message(CUDA_INCLUDE_DIRS : ${CUDA_INCLUDE_DIRS})
SET(GPUNUFFT_INC_DIR ${CMAKE_SOURCE_DIR}/inc)
SET(GPUNUFFT_INCLUDE ${GPUNUFFT_INC_DIR}/cuda_utils.hpp 
										 ${GPUNUFFT_INC_DIR}/cuda_utils.cuh
										 ${GPUNUFFT_INC_DIR}/config.hpp
										 ${GPUNUFFT_INC_DIR}/gpuNUFFT_utils.hpp
										 ${GPUNUFFT_INC_DIR}/gpuNUFFT_types.hpp
										 ${GPUNUFFT_INC_DIR}/gpuNUFFT_kernels.hpp
										 ${GPUNUFFT_INC_DIR}/precomp_kernels.hpp
										 ${GPUNUFFT_INC_DIR}/precomp_utils.hpp
                     ${GPUNUFFT_INC_DIR}/gpuNUFFT_operator.hpp
										 ${GPUNUFFT_INC_DIR}/balanced_operator.hpp
										 ${GPUNUFFT_INC_DIR}/balanced_gpuNUFFT_operator.hpp
                     ${GPUNUFFT_INC_DIR}/gpuNUFFT_operator_factory.hpp)
					 
SET(MATLAB_HELPER_INCLUDE ${GPUNUFFT_INC_DIR}/matlab_helper.h)
SET(CONFIG_INCLUDE ${GPUNUFFT_INC_DIR}/config.hpp ${GPUNUFFT_INC_DIR}/cufft_config.hpp)

#Adding src directory to the project

ADD_SUBDIRECTORY(src)
if(GEN_TESTS)
	# For make-based builds, defines make target named test.
	# For Visual Studio builds, defines Visual Studio project named RUN_TESTS.
	enable_testing()
	ADD_SUBDIRECTORY(test)
	add_test(
		NAME runUnitTests
		COMMAND runUnitTests
	)

	add_test(
		NAME runGPUATMUnitTests
		COMMAND runGPUATMUnitTests)
endif(GEN_TESTS)

ADD_SUBDIRECTORY(doc)

#CREATE zip Archive using JAVA in order to deploy project patch
if (GEN_ZIP)
	find_package(Java)

	MESSAGE(${PATCH_OUTPUT_PATH})
	execute_process(
	    COMMAND 
		"find" "${CMAKE_CURRENT_SOURCE_DIR}/src" "-not" "-name" "CMakeLists.txt" "-not" "-type" "d" 
OUTPUT_VARIABLE _file_list
	)
#MESSAGE(${_file_list})	
	execute_process( 
    COMMAND "${Java_JAR_EXECUTABLE}" "cfM" "${PATCH_OUTPUT_PATH}" "`" 
		"find" "${CMAKE_CURRENT_SOURCE_DIR}/src" "-not" "-name" "CMakeLists.txt" "-not" "-type" "d" "`"
#		"-C" ${CMAKE_CURRENT_SOURCE_DIR} "inc" 
#		"-C" ${CMAKE_CURRENT_SOURCE_DIR} "matlab/gpuNUFFT"
	    RESULT_VARIABLE _result
	)
endif(GEN_ZIP)


