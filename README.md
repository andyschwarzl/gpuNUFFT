## gpuNUFFT - GPU Regridding of arbitrary 3-D/2-D MRI data

- Andreas Schwarzl - andreas.schwarzl@student.tugraz.at
- Florian Knoll - florian.knoll@nyumc.org

-------------------------------------------------------------------------------
INFO:
-------------------------------------------------------------------------------
GPU 3D/2D regridding library with MATLAB(R) Mexfile output.
Go to subfolder CUDA to compile mexfiles. 

REQUIREMENTS:
-------------------------------------------------------------------------------

- CUDA capable graphics card and working installation of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [CMake 2.8](https://cmake.org/download/) or higher
- MATLAB 2008 or higher
- [Google test](https://github.com/google/googletest) framework (optional)

CMAKE Options:

- GEN_ATOMIC        : DEFAULT ON, enables fast library using atomic operations
- WITH_DEBUG        : DEFAULT OFF, enables Command-Line DEBUG output
- WITH_MATLAB_DEBUG : DEFAULT OFF, enables MATLAB Console DEBUG output
- GEN_TESTS         : DEFAULT OFF, generate Unit tests

Prior to compilation, the path where MATLAB is installed has to be defined in the top level CMakeLists.txt file, e.g.:

SET(MATLAB_ROOT_DIR "/home/florian/Programs/MATLAB/R2012b" CACHE STRING "MATLAB Installation Directory")

Alternatively, it can be passed as command line argument when calling cmake, e.g.:

```
cmake .. -DMATLAB_ROOT_DIR=/path/to/matlab
```

-------------------------------------------------------------------------------
LINUX, using gcc:
-------------------------------------------------------------------------------

build project via cmake, starting from project root directory:

    > mkdir build
    > cd build
    > cmake ..
    > make
	
Note: This version of gpuNUFFT was tested with CUDA 5.0. NVIDIAs nvcc compiler did not support gcc 4.7 or higher at this time. It is therefore suggested to compile gpuNUFFT with gcc 4.6. One comfortable way to toggle between gcc/g++ versions is to use the update-alternatives tool.

*Screencast*: Alternatively follow the installation and usage instructions for Linux [here](https://vimeo.com/161036817)

-------------------------------------------------------------------------------
WINDOWS, with Visual Studio:
-------------------------------------------------------------------------------
Setup a working version of [Visual Studio Community 2013](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx) or use the Visual Studio professional edition if available. Generate the project solution using the CMake-GUI or from command line by starting from the project root directory:

    > mkdir build 
    > cd build
    > cmake .. -G "Visual Studio 12 2013 Win64"
    > Build created Solution gpuNUFFT.sln using Visual Studio

For the Win64 platform check that all necessary Visual Studio Add-ons are
installed correctly and a Win64 dummy project can be created using VS.

*Screencast*: Alternatively follow the installation and usage instructions for Windows [here](https://vimeo.com/161037263)

-------------------------------------------------------------------------------
Run:
-------------------------------------------------------------------------------
The compiled CUDA-mexfiles will appear in the bin directory and are also copied 
automatically into the gpuNUFFT/@gpuNUFFT/private directory . Include the gpuNUFFT
directory into the matlab search path, in order to run the provided demo example.


-------------------------------------------------------------------------------
Doc:
-------------------------------------------------------------------------------
To generate the source code documentation run 

    > make doc

in the build directory. 

*Note: Requires doxygen to be installed.*
