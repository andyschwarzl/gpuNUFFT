/*
This file holds the python bindings for gpuNUFFT library.
Authors:
Chaithya G R <chaithyagr@gmail.com>
Carole Lazarus <carole.m.lazarus@gmail.com>
*/

#ifndef GPUNUFFT_OPERATOR_PYTHON_FACTORY_H_INCLUDED
#define GPUNUFFT_OPERATOR_PYTHON_FACTORY_H_INCLUDED
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h>
#include <cublas.h>
#include "config.hpp"
#include "gpuNUFFT_operator_factory.hpp"
#include <algorithm>  // std::sort
#include <vector>     // std::vector
#include <string>
#include <cuda.h>
#define CAST_POINTER_VARNAME(x, y)   cast_pointer(x, y, #x)

namespace py = pybind11;

template <typename TType>
gpuNUFFT::Array<TType>
readNumpyArray(py::array_t<TType> data)
{
    py::buffer_info myData = data.request();
    TType *t_data = (TType *) myData.ptr;
    gpuNUFFT::Array<TType> dataArray;
    dataArray.data = t_data;
    return dataArray;
}

gpuNUFFT::Array<DType2>
readNumpyArray(py::array_t<std::complex<DType>> data)
{
    gpuNUFFT::Array<DType2> dataArray;
    py::buffer_info myData = data.request();
    std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
    DType2 *new_data = reinterpret_cast<DType2(&)[0]>(*t_data);
    dataArray.data = new_data;
    return dataArray;
}


void warn_pinned_memory(py::array_t<std::complex<DType>> array, const char * name)
{
    py::buffer_info buffer = array.request();
    cudaPointerAttributes attr;
    if(DEBUG)
        printf("%s => Value of pointer == 0x%x\n", name, buffer.ptr);
    cudaPointerGetAttributes(&attr, buffer.ptr);
    if(DEBUG)
        printf("%s => of attr.cudaMemoryType = %d\n", name, attr.type);
    bool is_pinned_memory = attr.type ==  cudaMemoryTypeHost;
    if(!is_pinned_memory)
        std::cerr<<"WARNING:: The data"<<name<<"is NOT pinned! This will be slow, consider pinning\n";
}

void allocate_pinned_memory(gpuNUFFT::Array<DType2> *lin_array, unsigned long int size)
{
  DType2 *new_data;
  cudaMallocHost((void **)&new_data, size);
  lin_array->data = new_data;
}

void deallocate_pinned_memory(gpuNUFFT::Array<DType2> *lin_array)
{
  cudaFreeHost(lin_array->data);
  lin_array->data = NULL;
}

template <typename TType>
void copyNumpyArray(py::array_t<std::complex<DType>> data, TType *copy_data)
{
    py::buffer_info myData = data.request();
    std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
    TType *my_data = reinterpret_cast<TType(&)[0]>(*t_data);
    memcpy(copy_data, my_data, myData.size*sizeof(TType));
}

template <typename TType>
void cast_pointer(py::array_t<std::complex<DType>> data, gpuNUFFT::Array<TType> &copy_data, const char * name , bool warn=true)
{
    py::buffer_info myData = data.request();
    std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
    TType *my_data = reinterpret_cast<TType(&)[0]>(*t_data);
    copy_data.data = my_data;
    if (warn)
        warn_pinned_memory(data, name);
}




class GpuNUFFTPythonOperator
{
    gpuNUFFT::GpuNUFFTOperatorFactory factory;
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp;
    int trajectory_length, n_coils, dimension;
    bool has_sense_data;
    gpuNUFFT::Dimensions imgDims;
    // sensitivity maps
    gpuNUFFT::Array<DType2> sensArray, kspace_data, image;
    void allocate_memory_kspace()
    {
        allocate_pinned_memory(&kspace_data, n_coils*trajectory_length*sizeof(DType2));
    }

    void allocate_memory_image()
    {
        if(has_sense_data == false)
          allocate_pinned_memory(&image, n_coils * imgDims.count() * sizeof(DType2));
        else
          allocate_pinned_memory(&image, imgDims.count() * sizeof(DType2));
    }
    
    public:
    GpuNUFFTPythonOperator(py::array_t<DType> kspace_loc, py::array_t<int> image_size, int num_coils,
    py::array_t<std::complex<DType>> sense_maps,  std::optional<py::array_t<DType>> density_comp, int kernel_width=3,
    int sector_width=8, int osr=2, bool balance_workload=1) 
    {
        // k-space coordinates
        py::buffer_info sample_loc = kspace_loc.request();
        trajectory_length = sample_loc.shape[1];
        dimension = sample_loc.shape[0];
        gpuNUFFT::Array<DType> kSpaceTraj = readNumpyArray(kspace_loc);
        kSpaceTraj.dim.length = trajectory_length;

        // density compensation weights
        gpuNUFFT::Array<DType> density_compArray;
        if(density_comp.has_value())
        {
            density_compArray = readNumpyArray(density_comp.value());
            density_compArray.dim.length = trajectory_length;
            // No need else as the init is by default with 0 length and density comp is not applied
        }

        // image size
        py::buffer_info img_dim = image_size.request();
        int *dims = (int *) img_dim.ptr;
        imgDims.width = dims[0];
        imgDims.height = dims[1];
        if(dimension==3)
            imgDims.depth = dims[2];
        else
            imgDims.depth = 0;

        n_coils = num_coils;

        // Setup all the sizes
        kspace_data.dim.length = trajectory_length;
        kspace_data.dim.channels = num_coils;
        image.dim = imgDims;

        // sensitivity maps
        py::buffer_info sense_maps_buffer = sense_maps.request();
        if (sense_maps_buffer.shape.size()==0)
        {
            has_sense_data = false;
            sensArray.data = NULL;
        }
        else
        {
            CAST_POINTER_VARNAME(sense_maps, sensArray);
            sensArray.dim = imgDims;
            sensArray.dim.channels = n_coils;
            has_sense_data = true;
        }
        factory.setBalanceWorkload(balance_workload);
        gpuNUFFTOp = factory.createGpuNUFFTOperator(
            kSpaceTraj, density_compArray, sensArray, kernel_width, sector_width,
            osr, imgDims);

        if(has_sense_data == false)
            image.dim.channels = n_coils;
        else
            image.dim.channels = 1;
        cudaDeviceSynchronize();
    }

    py::array_t<std::complex<DType>> op(py::array_t<std::complex<DType>> in_image, py::array_t<std::complex<DType>> out_kspace, bool interpolate_data)
    {
        CAST_POINTER_VARNAME(in_image, image);
        CAST_POINTER_VARNAME(out_kspace, kspace_data);
        if(interpolate_data)
            gpuNUFFTOp->performForwardGpuNUFFT(image, kspace_data, gpuNUFFT::DENSITY_ESTIMATION);
        else
            gpuNUFFTOp->performForwardGpuNUFFT(image, kspace_data);
        cudaDeviceSynchronize();
        std::complex<DType> *ptr = reinterpret_cast<std::complex<DType>(&)[0]>(*kspace_data.data);
        auto capsule = py::capsule(ptr, [](void *ptr) { return;
        });
        return py::array_t<std::complex<DType>>(
            { n_coils, trajectory_length },
            {
                sizeof(DType2) * trajectory_length,
                sizeof(DType2)
            },
            ptr,
            capsule
        );
    }
    py::array_t<std::complex<DType>> adj_op(py::array_t<std::complex<DType>> in_kspace, py::array_t<std::complex<DType>> out_image, bool grid_data)
    {
        CAST_POINTER_VARNAME(in_kspace, kspace_data);
        CAST_POINTER_VARNAME(out_image, image);
        gpuNUFFT::Dimensions myDims = imgDims;
        if(dimension==2)
            myDims.depth = 1;
        if(grid_data)
            gpuNUFFTOp->performGpuNUFFTAdj(kspace_data, image, gpuNUFFT::DENSITY_ESTIMATION);
        else
            gpuNUFFTOp->performGpuNUFFTAdj(kspace_data, image);
        cudaDeviceSynchronize();
        std::complex<DType> *ptr = reinterpret_cast<std::complex<DType>(&)[0]>(*image.data);
        auto capsule = py::capsule(ptr, [](void *ptr) { return;
        });
        if(has_sense_data == false)
          return py::array_t<std::complex<DType>>(
            {
                n_coils,
                (int)myDims.depth,
                (int)myDims.height,
                (int)myDims.width
            },
            {
                sizeof(DType2) * (int)myDims.depth * (int)myDims.height * (int)myDims.width,
                sizeof(DType2) * (int)myDims.height * (int)myDims.width,
                sizeof(DType2) * (int)myDims.width,
                sizeof(DType2),
            },
            ptr,
            capsule
          );
        else
          return py::array_t<std::complex<DType>>(
            {
                (int)myDims.depth,
                (int)myDims.height,
                (int)myDims.width
            },
            {
                sizeof(DType2) * (int)myDims.height * (int)myDims.width,
                sizeof(DType2) * (int)myDims.width,
                sizeof(DType2),
            },
            ptr,
            capsule
      );
    }

    void clean_memory()
    {
       gpuNUFFTOp->clean_memory();
    }

    void set_smaps(py::array_t<std::complex<DType>> sense_maps)
    {
        CAST_POINTER_VARNAME(sense_maps, sensArray);
        has_sense_data = true;
        gpuNUFFTOp->setSens(sensArray);
    }
    ~GpuNUFFTPythonOperator()
    {
        delete gpuNUFFTOp;
    }
};

PYBIND11_MODULE(gpuNUFFT, m) {
    py::class_<GpuNUFFTPythonOperator>(m, "NUFFTOp")
        .def(py::init<py::array_t<DType>, py::array_t<int>, int, py::array_t<std::complex<DType>>, std::optional<py::array_t<DType>>, int, int, int, bool>(),
            py::arg("kspace_loc"), py::arg("image_size"), py::arg("num_coils"), py::arg("sense_maps") = py::none(), py::arg("density_comp") = py::none(), py::arg("kernel_width") = 3, py::arg("sector_width") = 8, py::arg("osr") = 2, py::arg("balance_workload") = true)
        .def("op", &GpuNUFFTPythonOperator::op, py::arg("in_image"), py::arg("out_kspace"), py::arg("interpolate_data") = false)
        .def("adj_op",  &GpuNUFFTPythonOperator::adj_op, py::arg("in_kspace"), py::arg("out_image"),  py::arg("grid_data") = false)
        .def("clean_memory", &GpuNUFFTPythonOperator::clean_memory)
        .def("set_smaps", &GpuNUFFTPythonOperator::set_smaps);
    
}
#endif  // GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED
