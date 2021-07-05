import os
import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from importlib import import_module
import platform
from pprint import pprint
import subprocess
try:
    from pip._internal.main import main as pip_main
except ImportError:
    from pip._internal import main as pip_main

release_info = {}

class CMakeExtension(Extension):

    def __init__(self, name, sourcedir):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """ Define a cmake build extension.
    """

    @staticmethod
    def _preinstall(package_list, options=[]):
        """ Pre-install PyPi packages before running cmake.
        """

        if not isinstance(package_list, list) or not isinstance(options, list):
            raise TypeError('preinstall inputs must be of type list.')

        pip_main(['install'] + options + package_list)


    def _set_pybind_path(self):
        """ Set path to Pybind11 include directory.
        """
        self.pybind_path = getattr(import_module('pybind11'), 'get_include')()

    def run(self):
        """ Redifine the run method.
        """
        # Set preinstall requirements
        preinstall_list = ["pybind11"]

        # Preinstall packages
        self._preinstall(preinstall_list)

        # Set Pybind11 path
        self._set_pybind_path()

        # Check cmake is installed and is sufficiently new.
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        # Build extensions
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """ Build extension with cmake.
        """
        # Define cmake arguments
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
                      "-DPYTHON_EXECUTABLE=" + sys.executable,
                      "-DGEN_PYTHON_FILES=ON",
                      "-DGEN_MEX_FILES=OFF",
                      "-DPYBIND11_INCLUDE_DIR=" + self.pybind_path]
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            cmake_args += ['-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j8']

        # Call cmake in specific environment
        env = os.environ.copy()
        env["CXXFLAGS"] = '{0} -DVERSION_INFO=\\"{1}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version())
        build_temp_dir = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp_dir):
            os.makedirs(build_temp_dir)
        print("Building " + ext.name + " in {0}...".format(build_temp_dir))
        print("Cmake args:")
        pprint(cmake_args)
        print("Cmake build args:")
        pprint(build_args)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args,
                              cwd=build_temp_dir, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args,
                              cwd=build_temp_dir)
        print()

setup(
    name="gpuNUFFT",
    version="0.2.1",
    description="gpuNUFFT - An open source GPU Library for 3D Gridding and NUFFT",
    package_dir={"": "CUDA/bin"},
    ext_modules=[
        CMakeExtension("gpuNUFFT", sourcedir=os.path.join("CUDA")),
    ],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    zip_safe=False
)
