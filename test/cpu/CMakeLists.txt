#setup testframework
MESSAGE("------ generating cpu tests -------")

#cpu test sources
set(CPU_SOURCES gridding3D_tests.cpp 
				gridding3D_kernel_tests.cpp
				gridding3D_precomputation_tests.cpp
				gridding3D_operator_factory_tests.cpp
				../../src/gridding_utils.cpp 
				../../src/cpu/gridding3D_cpu.cpp)

include_directories(${CUDA_INCLUDE_DIRS})
#add source dir
add_executable(runUnitTests ${CPU_SOURCES} ${GRIDDING_SOURCES} ../../inc/gridding_cpu.hpp ../../inc/gridding_utils.hpp ../../inc/gridding_operator_factory.hpp ../../inc/gridding_operator.hpp ../../inc/gridding_kernels.hpp)
target_link_libraries(runUnitTests ${GRID_LIB_NAME} ${GTEST_LIB} ${GTESTMAIN_LIB})
set_target_properties(runUnitTests PROPERTIES LINK_FLAGS -lpthread)
