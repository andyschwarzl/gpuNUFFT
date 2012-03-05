#The name of the project
PROJECT(gridding3D)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#Searching CUDA
FIND_PACKAGE(CUDA)

#Searching MATLAB
FIND_PACKAGE(MATLAB)

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