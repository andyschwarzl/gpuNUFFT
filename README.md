gpuNUFFT
==========

Matlab/CPP GpuNUFFT Library

gpuNUFFT - GPU Regridding of Radial 3-D MRI data

Andreas Schwarzl - andreas.schwarzl@student.tugraz.at
-------------------------------------------------------------------------------
INFO:
-------------------------------------------------------------------------------
GPU 3D regridding library with MATLAB(R) Mexfile output.

REQUIREMENTS:
-------------------------------------------------------------------------------

- CMAKE 2.8
- MATLAB 2008 or higher
- (optional) GOOGLE test framework 1.6

CMAKE Options:

- GEN_ATOMIC        : DEFAULT ON, enables fast library using atomic operations
- WITH_DEBUG        : DEFAULT OFF, enables Command-Line DEBUG output
- WITH_MATLAB_DEBUG : DEFAULT OFF, enables MATLAB Console DEBUG output
- GEN_TESTS         : DEFAULT OFF, enables simple unit tests using GOOGLE Test framework

-------------------------------------------------------------------------------
LINUX:
-------------------------------------------------------------------------------

build project via cmake, starting from project root directory:

    > mkdir build
    > cd build
    > cmake ..
	> make
	
-------------------------------------------------------------------------------
WINDOWS:
-------------------------------------------------------------------------------
build project using cmake or cmake-gui, starting from project root directory:

    > mkdir build 
	> cd build
	> cmake .. -G "Visual Studio 2008 Win64" 
    > Build created Solution gpuNUFFT.sln using Visual Studio

Ensure that at least a Visual Studio 2008 Express build environment is setup correctly.
If the library shall run using Win64 check that all necessary Visual Studio Add-ons are
installed correctly and a Win64 dummy project can be created using VS.

In case of the following compile time error

nvcc fatal   : Visual Studio configuration file '(null)' could not be found for installation at 'C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\x86_amd64/../../..'

you have to manually build a 'vcvars64.bat' file in 'Program Files\Microsoft Visual Studio 10.0\VC\bin\amd64' containing: 

CALL setenv /x64

reference: http://stackoverflow.com/questions/8900617/how-can-i-setup-nvcc-to-use-visual-c-express-2010-x64-from-windows-sdk-7-1

-------------------------------------------------------------------------------
TEST FRAMEWORK:
-------------------------------------------------------------------------------
To enable the tests accordingly download and install the google test framework (https://code.google.com/p/googletest/downloads/list)
as follows:
```
> wget http://googletest.googlecode.com/files/gtest-1.7.0.zip
> unzip gtest-1.7.0.zip
> mv gtest-1.7.0 gtest
> cd gtest
> mkdir build
> cd build
> cmake ..
> make 

Note to keep the following directory structure, e.g.:

work
 | - gpuNUFFT
 | - gtest
```
-------------------------------------------------------------------------------
KNOWN ISSUES:
-------------------------------------------------------------------------------
---------------- RESOLVED -----------------
The software performed stable on the test environment described above. 
However, unresolved problems on different Linux platforms may occur at the use 
of MATLAB 'clear classes' command, leading to a crash of the MATLAB application. 

-> added cudaDeviceReset at mexExit function

----------------  RESOLVED  -----------------
Invalid MEX-file
'/home/florian/Documents/MATLAB/functions/gridding_gpu/mex_gpuNUFFT_adj_atomic_f.mexa64':
/home/florian/Programs/MATLAB/R2012b/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: 
version `GLIBCXX_3.4.15' not found (required by
/usr/local/cuda-5.0/lib64/libcudart.so.5.0)

Zur Info, es ist Ubuntu 12.10, CUDA 5.0, gcc hab ich 4.4, 4.6 und 4.7 ausprobiert, Matlab ist 2012b.

--> so, ich habs jetzt glaube ich geloest. Ziemlich bloeder Fehler, Matlab kommt standardmaessig mit einer (aelteren) libstdc++, und man muss es jetzt dazu bringen die neuere, die mit der Distribution mitkommt zu verwenden, und dazu den Link dorthin umleiten: 

/usr/lib/x86_64-linux-gnu/libstdc++.so.6

