MESSAGE("Sparse GPU Gridding Library (Freiburg)")

set(SPARSE_MEX_FILE tikreg_gridding_gpu_f.cu )
set(SPARSE_MEX_FILE_FORW tikreg_gridding_forward_gpu_f.cu )

CUDA_COMPILE( mex_gridding3D_sparse_gen ${SPARSE_MEX_FILE} SHARED)
CUDA_COMPILE( mex_gridding3D_sparse_forw_gen ${SPARSE_MEX_FILE_FORW} SHARED)

CUDA_ADD_LIBRARY(mex_gridding3D_sparse SHARED ${mex_gridding3D_sparse_gen} ${SPARSE_MEX_FILE})
TARGET_LINK_LIBRARIES(mex_gridding3D_sparse ${MATLAB_MEX_LIBRARY} ${MATLAB_MX_LIBRARY} ${CUDA_LIBRARIES})
SET_TARGET_PROPERTIES(mex_gridding3D_sparse PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

CUDA_ADD_CUFFT_TO_TARGET(mex_gridding3D_sparse)

CUDA_ADD_LIBRARY(mex_gridding3D_sparse_forw SHARED ${mex_gridding3D_sparse_forw_gen} ${SPARSE_MEX_FILE_FORW})
TARGET_LINK_LIBRARIES(mex_gridding3D_sparse_forw ${MATLAB_MEX_LIBRARY} ${MATLAB_MX_LIBRARY} ${CUDA_LIBRARIES})
SET_TARGET_PROPERTIES(mex_gridding3D_sparse_forw PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

CUDA_ADD_CUFFT_TO_TARGET(mex_gridding3D_sparse_forw)

if(WIN32)
  SET_TARGET_PROPERTIES(mex_gridding3D_sparse PROPERTIES
						SUFFIX ${MEX_SUFFIX}
						LINK_FLAGS /export:mexFunction)
  SET_TARGET_PROPERTIES(mex_gridding3D_sparse_forw PROPERTIES
						SUFFIX ${MEX_SUFFIX}
						LINK_FLAGS /export:mexFunction)
else(UNIX)
  SET_TARGET_PROPERTIES(mex_gridding3D_sparse PROPERTIES SUFFIX ${MEX_SUFFIX})
  SET_TARGET_PROPERTIES(mex_gridding3D_sparse_forw PROPERTIES SUFFIX ${MEX_SUFFIX})  
endif()