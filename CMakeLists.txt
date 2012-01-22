#The name of the project
PROJECT(gridding3D)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#Searching CUDA
FIND_PACKAGE(CUDA)

#Adding src directory to the project
ADD_SUBDIRECTORY(src)