MESSAGE("MEXfile build")

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS 1)

macro( matlab_config )
 if(WIN32)
    message( "MATLAB_INCLUDE_DIR = ${MATLAB_INCLUDE_DIR}" )
    #MATLAB_LIBRARIES
    #MATLAB_MEX_LIBRARY
    #MATLAB_MX_LIBRARY
    #MATLAB_ENG_LIBRARY

    if( BUILD_64BIT )
        set( LIB_DIR win64 )
        message( "BUILDING FOR 64 bit Windows" ) 

    else(BUILD_64BIT)
        set( LIB_DIR win32 )
        message( "BUILDING FOR 32 bit Windows" )

    endif(BUILD_64BIT)

    SET( MATLAB_ROOT_DIR "C:\\Program Files\\MATLAB\\R2011a" CACHE STRING "")
    SET( MATLAB_INCLUDE_DIR "${MATLAB_ROOT_DIR}\\extern\\include" CACHE STRING "" FORCE )

    find_library(MATLAB_MX_LIBRARY libmx.lib PATHS ${MATLAB_ROOT_DIR} ${MATLAB_ROOT_DIR}/extern/lib/${LIB_DIR}/microsoft )
    find_library(MATLAB_MEX_LIBRARY libmex.lib PATHS ${MATLAB_ROOT_DIR} ${MATLAB_ROOT_DIR}/extern/lib/${LIB_DIR}/microsoft )

    message("MATLAB_LIB_DIR = ${MATLAB_ROOT_DIR}/extern/lib/${LIB_DIR}/microsoft" )
    message( "MATLAB_LIB_DIR = ${MATLAB_LIB_DIR}" )
    message( "matlab lib dir = ${MATLAB_ROOT_DIR}/extern/lib/${LIB_DIR}" )


    SET( MATLAB_LIBRARIES ${MATLAB_ROOT_DIR}/bin/${LIB_DIR} CACHE STRING "" FORCE )
    #SET( MATLAB_MEX_LIBRARY ${MATLAB_ROOT_DIR}/extern/lib/${LIB_DIR}/microsoft CACHE STRING "" FORCE )
    #SET( MATLAB_MX_LIBRARY ${MATLAB_ROOT_DIR}/extern/lib/${LIB_DIR}/microsoft CACHE STRING "" FORCE )
    SET( MATLAB_ENG_LIBRARY ${MATLAB_ROOT_DIR}/extern/lib/${LIB_DIR}/microsoft CACHE STRING "" FORCE )

    INCLUDE_DIRECTORIES( ${MATLAB_INCLUDE_DIR})
else(WIN32)
    SET (MATLAB_ROOT_DIR "/usr/local/MATLAB/R2010b")
    SET( MATLAB_INCLUDE_DIR "${MATLAB_ROOT_DIR}/extern/include" CACHE STRING "" FORCE )

    SET(LIBMEX "libmex.so")
    SET(LIBMX "libmx.so")
    SET(LIBENG "libeng.so")

    if (CMAKE_SIZEOF_VOID_P MATCHES "8")
        set( MATLAB_LIB_PATHS "${MATLAB_ROOT_DIR}/bin/glnxa64")
        message( "BUILDING FOR 64 bit Linux" ) 
    else(CMAKE_SIZEOF_VOID_P MATCHES "8")
        set( MATLAB_LIB_PATHS "${MATLAB_ROOT_DIR}/bin/glnx86")
        message( "BUILDING FOR 32 bit Linux" )
    endif(CMAKE_SIZEOF_VOID_P MATCHES "8")

    FIND_LIBRARY(MATLAB_MEX_LIBRARY ${LIBMEX} ${MATLAB_LIB_PATHS} NO_DEFAULT_PATH)
    FIND_LIBRARY(MATLAB_MX_LIBRARY ${LIBMX} ${MATLAB_LIB_PATHS} NO_DEFAULT_PATH)
    FIND_LIBRARY(MATLAB_ENG_LIBRARY ${LIBENG} ${MATLAB_LIB_PATHS} NO_DEFAULT_PATH)

    INCLUDE_DIRECTORIES( ${MATLAB_INCLUDE_DIR})
    
    SET(MATLAB_LIBRARIES
        ${MATLAB_MEX_LIBRARY}
        ${MATLAB_MX_LIBRARY}
        ${MATLAB_ENG_LIBRARY}  
        )
  endif(WIN32)
endmacro()

matlab_config()

MESSAGE('MATLAB libraries: ${MATLAB_LIBRARIES}')
MESSAGE('MATLAB inc dir: ${MATLAB_INCLUDE_DIR}')
INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

set(MEX_FILE gridding3D_mex.cpp)
set(CUDA_NVCC_FLAGS -arch sm_13;--ptxas-options=-v)

CUDA_COMPILE(cuda_mex_kernel_generated ${MEX_FILE} SHARED)

CUDA_ADD_LIBRARY(cuda_atomic_mex_kernel SHARED ${cuda_mex_kernel_generated} ${MEX_FILE})

TARGET_LINK_LIBRARIES(cuda_atomic_mex_kernel ${MATLAB_MEX_LIBRARY} ${MATLAB_MX_LIBRARY} ${CUDA_LIBRARIES} gridding3DATM)
SET_TARGET_PROPERTIES(cuda_atomic_mex_kernel PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

if(WIN32)
  if (CMAKE_CL_64)
          MESSAGE("Win 64")
          SET_TARGET_PROPERTIES(cuda_atomic_mex_kernel PROPERTIES
                  SUFFIX .mexw64
                  LINK_FLAGS /export:mexFunction
          )
  else()
          MESSAGE("Win 32")
          SET_TARGET_PROPERTIES(cuda_atomic_mex_kernel PROPERTIES
                  SUFFIX .mexw32
                  LINK_FLAGS /export:mexFunction
          )
  endif()
else(UNIX)
  if (CMAKE_SIZEOF_VOID_P MATCHES "8")
  MESSAGE("Linux 64")
          SET_TARGET_PROPERTIES(cuda_atomic_mex_kernel PROPERTIES SUFFIX .mexa64)
  else()
          MESSAGE("Linux 32")
          SET_TARGET_PROPERTIES(cuda_atomic_mex_kernel PROPERTIES SUFFIX .mexglx)
  endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif()
