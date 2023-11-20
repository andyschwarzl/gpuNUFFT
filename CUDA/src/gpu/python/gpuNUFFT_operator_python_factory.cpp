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
void cast_pointer(py::array_t<std::complex<DType>> data, gpuNUFFT::Array<TType> &copy_data)
{
    py::buffer_info myData = data.request();
    std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
    TType *my_data = reinterpret_cast<TType(&)[0]>(*t_data);
    copy_data.data = my_data;
}

enum MemoryAllocationType{
        NEVER_ALLOCATE_MEMORY = 0,
        ALLOCATE_MEMORY_IN_CONSTRUCTOR = 1,
        ALLOCATE_MEMORY_IN_OP = 2
    };

class GpuNUFFTPythonOperator
{
    gpuNUFFT::GpuNUFFTOperatorFactory factory;
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp;
    int trajectory_length, n_coils, dimension;
    bool has_sense_data;
    MemoryAllocationType when_allocate_memory;
    gpuNUFFT::Dimensions imgDims;
    // sensitivity maps
    gpuNUFFT::Array<DType2> sensArray, kspace_data, image;
    void allocate_memory_kspace()
    {
        allocate_pinned_memory(&kspace_data, n_coils*trajectory_length*sizeof(DType2));
        
    }

    void allocate_memory_image()
    {
        image.dim = imgDims;
        if(has_sense_data == false)
          allocate_pinned_memory(&image, n_coils * imgDims.count() * sizeof(DType2));
        else
          allocate_pinned_memory(&image, imgDims.count() * sizeof(DType2));
    }
    
    public:
    GpuNUFFTPythonOperator(py::array_t<DType> kspace_loc, py::array_t<int> image_size, int num_coils,
    py::array_t<std::complex<DType>> sense_maps,  py::array_t<float> density_comp, int kernel_width=3,
    int sector_width=8, int osr=2, bool balance_workload=1, MemoryAllocationType when_allocate_memory=ALLOCATE_MEMORY_IN_CONSTRUCTOR) : when_allocate_memory(when_allocate_memory)
    {
        // k-space coordinates
        py::buffer_info sample_loc = kspace_loc.request();
        trajectory_length = sample_loc.shape[1];
        dimension = sample_loc.shape[0];
        gpuNUFFT::Array<DType> kSpaceTraj = readNumpyArray(kspace_loc);
        kSpaceTraj.dim.length = trajectory_length;

        // density compensation weights
        gpuNUFFT::Array<DType> density_compArray = readNumpyArray(density_comp);
        density_compArray.dim.length = trajectory_length;

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
        kspace_data.dim.length = trajectory_length;
        kspace_data.dim.channels = n_coils;
        
        // sensitivity maps
        py::buffer_info sense_maps_buffer = sense_maps.request();
        if (sense_maps_buffer.shape.size()==0)
        {
            has_sense_data = false;
            sensArray.data = NULL;
        }
        else
        {
            cudaPointerAttributes attr;
            if(DEBUG)
                printf("Value of sense_maps pointer == 0x%x or %d\n", sense_maps_buffer.ptr, sense_maps_buffer.ptr);
            cudaPointerGetAttributes(&attr, sense_maps_buffer.ptr);
            if(DEBUG)
                printf("Value of attr.cudaMemoryType2 = %d\n", attr.type);
            bool is_pinned_memory = attr.type ==  cudaMemoryTypeHost;
            if(is_pinned_memory)
            {
                if(DEBUG)
                    printf("The smaps data is pinned!, skipping copies\n");
                // Just map the memory to sensArray! We dont need to make a copy if the memory is already pinned
                std::complex<DType> *t_data = (std::complex<DType> *) sense_maps_buffer.ptr;
                sensArray.data = reinterpret_cast<DType2(&)[0]>(*t_data);
            }
            else
            {
                if(DEBUG)
                    printf("The smaps data is NOT pinned!, DOING copies\n");
                allocate_pinned_memory(&sensArray, n_coils * imgDims.count() * sizeof(DType2));
                sensArray.dim = imgDims;
                sensArray.dim.channels = n_coils;
                copyNumpyArray(sense_maps, sensArray.data);
                has_sense_data = true;
            }
        }
        factory.setBalanceWorkload(balance_workload);
        gpuNUFFTOp = factory.createGpuNUFFTOperator(
            kSpaceTraj, density_compArray, sensArray, kernel_width, sector_width,
            osr, imgDims);

        if(has_sense_data == false)
            image.dim.channels = n_coils;
        else
            image.dim.channels = 1;
        if(when_allocate_memory == ALLOCATE_MEMORY_IN_CONSTRUCTOR)
        {
            allocate_memory_kspace();
            allocate_memory_image();
        }
        cudaDeviceSynchronize();
    }

    py::array_t<std::complex<DType>> op(py::array_t<std::complex<DType>> input_image, bool interpolate_data, std::optional<py::array_t<std::complex<DType>>> out_kspace)
    {
        if(when_allocate_memory == ALLOCATE_MEMORY_IN_OP)
        {
            allocate_memory_kspace();
            allocate_memory_image();
            // Copy array to pinned memory for better memory bandwidths!
            copyNumpyArray(input_image, image.data);
        }
        else if(when_allocate_memory == NEVER_ALLOCATE_MEMORY)
        {
            printf("Not allocating memory in op\n");
            cast_pointer(input_image, image);
            printf("Output kspace : %d\n", out_kspace.has_value());
            if(out_kspace.has_value())
                cast_pointer(out_kspace.value(), kspace_data);
            else
            {
                // We dont have out_kspace allocated. Warn and then allocate
                py::print("WARNING: NEVER_ALLOCATE_MEMORY is chosen but no memory is specified, allocating for now!");
                allocate_memory_kspace();
                cudaDeviceSynchronize();
            }
        }
        if(interpolate_data)
            gpuNUFFTOp->performForwardGpuNUFFT(image, kspace_data, gpuNUFFT::DENSITY_ESTIMATION);
        else
            gpuNUFFTOp->performForwardGpuNUFFT(image, kspace_data);
        cudaDeviceSynchronize();
        std::complex<DType> *ptr = reinterpret_cast<std::complex<DType>(&)[0]>(*kspace_data.data);
        auto capsule = py::capsule(ptr, [](void *ptr) { return;
        });
        if (when_allocate_memory == ALLOCATE_MEMORY_IN_OP)
        {
            // Deallocate the memory (only image) to prevent memory leaks!
            deallocate_pinned_memory(&image);
        }
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
    py::array_t<std::complex<DType>> adj_op(py::array_t<std::complex<DType>> input_kspace, bool grid_data, std::optional<py::array_t<std::complex<DType>>> out_image)
    {
        if(when_allocate_memory == ALLOCATE_MEMORY_IN_OP)
        {
            allocate_memory_kspace();
            allocate_memory_image();
            // Copy array to pinned memory for better memory bandwidths!
            copyNumpyArray(input_kspace, kspace_data.data);
        }
        else if(when_allocate_memory == NEVER_ALLOCATE_MEMORY)
        {
            cast_pointer(input_kspace, kspace_data);
            // Check if we have out image allocated
            if (out_image.has_value())
                cast_pointer(out_image.value(), image);
            else
            {
                // We dont have out_image allocated. Warn and then allocate
                py::print("WARNING: NEVER_ALLOCATE_MEMORY is chosen but no memory is specified, allocating for now!");
                allocate_memory_image();
            }
        }
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
        if (when_allocate_memory == ALLOCATE_MEMORY_IN_OP)
        {
            // Deallocate the memory (only k-space) to prevent memory leaks!
            deallocate_pinned_memory(&kspace_data);
        }
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
        py::buffer_info myData = sense_maps.request();
        std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
        DType2 *my_data = reinterpret_cast<DType2(&)[0]>(*t_data);
        memcpy(sensArray.data, my_data, myData.size*sizeof(DType2));
        has_sense_data = true;
        gpuNUFFTOp->setSens(sensArray);
    }
    ~GpuNUFFTPythonOperator()
    {
        cudaFreeHost(kspace_data.data);
        cudaFreeHost(image.data);
        delete gpuNUFFTOp;
    }
};

PYBIND11_MODULE(gpuNUFFT, m) {
    py::enum_<MemoryAllocationType>(m, "MemoryAllocationType")
        .value("NEVER_ALLOCATE_MEMORY", MemoryAllocationType::NEVER_ALLOCATE_MEMORY)
        .value("ALLOCATE_MEMORY_IN_CONSTRUCTOR", MemoryAllocationType::ALLOCATE_MEMORY_IN_CONSTRUCTOR)
        .value("ALLOCATE_MEMORY_IN_OP", MemoryAllocationType::ALLOCATE_MEMORY_IN_OP)
        .export_values();

    py::class_<GpuNUFFTPythonOperator>(m, "NUFFTOp")
        .def(py::init<py::array_t<DType>, py::array_t<int>, int, py::array_t<std::complex<DType>>, py::array_t<float>, int, int, int, bool, MemoryAllocationType>(),
            py::arg("kspace_loc"), py::arg("image_size"), py::arg("num_coils"), py::arg("sense_maps") = py::none(), py::arg("density_comp") = py::none(), py::arg("kernel_width") = 3, py::arg("sector_width") = 8, py::arg("osr") = 2, py::arg("balance_workload") = true, py::arg("when_allocate_memory") = ALLOCATE_MEMORY_IN_CONSTRUCTOR
        )
        .def("op", &GpuNUFFTPythonOperator::op, py::arg("input_image"), py::arg("interpolate_data") = false, py::arg("out_kspace") = py::none())
        .def("adj_op",  &GpuNUFFTPythonOperator::adj_op, py::arg("input_kspace"), py::arg("grid_data") = false, py::arg("out_image") = py::none())
        .def("clean_memory", &GpuNUFFTPythonOperator::clean_memory)
        .def("set_smaps", &GpuNUFFTPythonOperator::set_smaps);
    
}
#endif  // GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED
