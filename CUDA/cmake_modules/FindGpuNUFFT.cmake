INCLUDE(FindPackageHandleStandardArgs)

  SET(GpuNUFFT_InstallDir /home/aschwarzl/workspace/master_thesis/IMT_ReconTools/CUDA_GPU_NUFFT/gpuNUFFT-2.0.5/gpuNUFFT-2.0.5/CUDA)

  SET(GpuNUFFT_IncludeSearchPaths ${GpuNUFFT_InstallDir}/inc) #/usr/local/include

  SET(GpuNUFFT_LibrarySearchPaths /${GpuNUFFT_InstallDir}/bin) #/usr/local/lib

FIND_PATH(GPUNUFFT_INCLUDE_DIR NAMES gpuNUFFT_operator_factory.hpp PATHS ${GpuNUFFT_IncludeSearchPaths})

FIND_LIBRARY(GPUNUFFT_LIB NAMES gpuNUFFT gpuNUFFT_ATM_f gpuNUFFT_f  PATHS ${GpuNUFFT_LibrarySearchPaths})

#Handle the REQUIRED argument and set the < UPPERCASED_NAME > _FOUND variable
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(
     GpuNUFFT 
      "Could NOT find GpuNUFFT." GPUNUFFT_LIB GPUNUFFT_INCLUDE_DIR)
IF(GpuNUFFT_FOUND)
  FIND_PACKAGE_MESSAGE(
      GpuNUFFT      
      "Found GpuNUFFT ${GPUNUFFT_LIB}"
      "[${GPUNUFFT_LIB}][${GPUNUFFT_INCLUDE_DIR}]")
  ENDIF(GpuNUFFT_FOUND)
MARK_AS_ADVANCED(GPUNUFFT_INCLUDE_DIR GPUNUFFT_LIB)
