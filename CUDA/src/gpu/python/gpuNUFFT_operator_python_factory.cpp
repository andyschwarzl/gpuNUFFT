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
#include <cublas_v2.h>
#include <curand.h>
#include "config.hpp"
#include "gpuNUFFT_operator_factory.hpp"
#include <algorithm>  // std::sort
#include <vector>     // std::vector
#include <string>
#include <cuda.h>
#include <cstdint>

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

template <typename TType>
void allocate_pinned_memory(gpuNUFFT::Array<TType> *lin_array, unsigned long int size)
{
  TType *new_data;
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
    int trajectory_length, n_coils, dimension, gpu_device_id;
    float osr;
    bool has_sense_data;
    gpuNUFFT::Dimensions imgDims;
    // sensitivity maps
    gpuNUFFT::GpuArray<DType2> image_gpu;
    gpuNUFFT::GpuArray<CufftType> kspace_data_gpu;
    gpuNUFFT::Array<CufftType> sensArray, kspace_data, image;
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
    py::array_t<std::complex<DType>> sense_maps,  py::array_t<DType> density_comp, int kernel_width=3,
    int sector_width=8, float osr=2, bool balance_workload=1, int gpu_device_id = 0) : osr(osr)
    {
        cudaError_t err = cudaSetDevice(gpu_device_id);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); 
        // k-space coordinates
        py::buffer_info sample_loc = kspace_loc.request();
        trajectory_length = sample_loc.shape[1];
        dimension = sample_loc.shape[0];
        gpuNUFFT::Array<DType> kSpaceTraj = readNumpyArray(kspace_loc);
        kSpaceTraj.dim.length = trajectory_length;

        // density compensation weights
        gpuNUFFT::Array<DType> density_compArray;
        if(density_comp != Py_None)
        {
            density_compArray = readNumpyArray(density_comp);
            density_compArray.dim.length = trajectory_length;
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
        kspace_data_gpu.dim.length = trajectory_length;
        kspace_data_gpu.dim.channels = num_coils;
        
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
        image_gpu.dim = imgDims;
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

    void op_direct(uintptr_t in_image, uintptr_t out_kspace, bool interpolate_data)
    {
        image_gpu.data = (DType2*) in_image;
        kspace_data_gpu.data = (CufftType*) out_kspace;
        if(interpolate_data)
            gpuNUFFTOp->performForwardGpuNUFFT(image_gpu, kspace_data_gpu, gpuNUFFT::DENSITY_ESTIMATION);
        else
            gpuNUFFTOp->performForwardGpuNUFFT(image_gpu, kspace_data_gpu);
        cudaDeviceSynchronize();
    }

    void adj_op_direct(uintptr_t in_kspace, uintptr_t out_image, bool grid_data)
    {
        kspace_data_gpu.data = (CufftType*) in_kspace;
        image_gpu.data = (DType2*) out_image;
        if(grid_data)
            gpuNUFFTOp->performGpuNUFFTAdj(kspace_data_gpu, image_gpu, gpuNUFFT::DENSITY_ESTIMATION);
        else
            gpuNUFFTOp->performGpuNUFFTAdj(kspace_data_gpu, image_gpu);
        cudaDeviceSynchronize();
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

    void toeplitz_op_direct(uintptr_t in_image, uintptr_t out_image, uintptr_t kernel)
    {
        image_gpu.data = (DType2*) in_image;
        // For now always n_coils_cc = 1
        gpuNUFFTOp->initDeviceMemory(kspace_data.dim.channels, 1);
        gpuNUFFTOp->performForwardGpuNUFFT(image_gpu, kspace_data_gpu, gpuNUFFT::TOEPLITZ);
        cudaDeviceSynchronize();
        // Perform multiplication with kernel
        performSensMul(kspace_data_gpu.data, (DType2*) kernel, gpuNUFFTOp->gi_host, false, true);
        cudaDeviceSynchronize();
        image_gpu.data = (DType2*) out_image;        
        gpuNUFFTOp->performGpuNUFFTAdj(kspace_data_gpu, image_gpu, gpuNUFFT::TOEPLITZ);

        gpuNUFFTOp->freeDeviceMemory();
        cudaDeviceSynchronize();
    }

    void clean_memory()
    {
       gpuNUFFTOp->clean_memory();
    }

    void toggle_grad_mode()
    {
        bool current_mode = gpuNUFFTOp->getGradMode();
        gpuNUFFTOp->setGradMode(!current_mode);
    }

    void set_smaps(py::array_t<std::complex<DType>> sense_maps)
    {
        CAST_POINTER_VARNAME(sense_maps, sensArray);
        has_sense_data = true;
        gpuNUFFTOp->setSens(sensArray);
    }

    void set_pts(py::array_t<DType> kspace_loc, py::array_t<DType> density_comp)
    {
        gpuNUFFT::Array<DType> kSpaceTraj = readNumpyArray(kspace_loc);
        kSpaceTraj.dim.length = trajectory_length;

        // density compensation weights
        gpuNUFFT::Array<DType> density_compArray;
        if(density_comp != Py_None)
        {
            density_compArray = readNumpyArray(density_comp);
            density_compArray.dim.length = trajectory_length;
        }
        factory.set_pts(gpuNUFFTOp, kSpaceTraj, density_compArray);

    }
    py::array_t<DType> estimate_density_comp(int max_iter = 10)
    {
        IndType n_samples = kspace_data.count();
        gpuNUFFT::Array<CufftType> densArray;
        allocate_pinned_memory(&densArray, n_samples * sizeof(CufftType));
        densArray.dim.length = n_samples;

        for (long int cnt = 0; cnt < n_samples; cnt++)
        {
          densArray.data[cnt].x = 1.0;
          densArray.data[cnt].y = 0.0;
        }

        gpuNUFFT::GpuArray<DType2> densArray_gpu;
        densArray_gpu.dim.length = n_samples;
        allocateDeviceMem(&densArray_gpu.data, n_samples);

        copyToDeviceAsync(densArray.data, densArray_gpu.data, n_samples);

        gpuNUFFT::GpuArray<CufftType> densEstimation_gpu;
        densEstimation_gpu.dim.length = n_samples;
        allocateDeviceMem(&densEstimation_gpu.data, n_samples);

        gpuNUFFT::GpuArray<CufftType> image_gpu;
        image_gpu.dim = imgDims;
        allocateDeviceMem(&image_gpu.data, image_gpu.dim.count());

        if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
          printf("error at adj thread synchronization a: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
        for (int cnt = 0; cnt < max_iter; cnt++)
        {
          if (DEBUG)
                printf("### update %i\n", cnt);
          gpuNUFFTOp->performGpuNUFFTAdj(densArray_gpu, image_gpu,
                                         gpuNUFFT::DENSITY_ESTIMATION);
          gpuNUFFTOp->performForwardGpuNUFFT(image_gpu, densEstimation_gpu,
                                             gpuNUFFT::DENSITY_ESTIMATION);
          performUpdateDensityComp(densArray_gpu.data, densEstimation_gpu.data, n_samples);
          if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
                printf("error at adj thread synchronization d: %s\n",
                       cudaGetErrorString(cudaGetLastError()));
        }
        freeDeviceMem(densEstimation_gpu.data);
        freeDeviceMem(image_gpu.data);

        cudaDeviceSynchronize();
        // copy only the real part back to cpu
        DType *tmp_d = (DType *)densArray_gpu.data;

        gpuNUFFT::Array<DType> final_densArray;
        final_densArray.dim.length = n_samples;
        allocate_pinned_memory(&final_densArray, n_samples * sizeof(DType));
        HANDLE_ERROR(cudaMemcpy2DAsync(final_densArray.data, sizeof(DType),
                                       tmp_d, sizeof(DType2), sizeof(DType),
                                       n_samples, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        freeDeviceMem(densArray_gpu.data);
        DType *ptr = reinterpret_cast<DType(&)[0]>(*final_densArray.data);
        auto capsule = py::capsule(ptr, [](void *ptr) { return; });
        return py::array_t<DType>({ trajectory_length }, { sizeof(DType) }, ptr,
                                  capsule);
    }

    


    float get_spectral_radius(int max_iter = 20,float tolerance = 1e-6)
    {
        int im_size = image.count();

        gpuNUFFT::GpuArray<DType2> x_gpu;
        x_gpu.dim = image.dim;
        allocateDeviceMem(&x_gpu.data, im_size);

        gpuNUFFT::GpuArray<DType2> tmp_kspace_gpu;
        tmp_kspace_gpu.dim = kspace_data.dim;
        allocateDeviceMem(&tmp_kspace_gpu.data, kspace_data.count());

        cudaDeviceSynchronize();
        DType norm_old = 1.0;
        DType norm_new = 1.0;
        DType inv_norm = 1.0;
        // initialisation: create a random complex image.
        curandGenerator_t generator;
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
        curandSetPseudoRandomGeneratorSeed(generator, (int)time(NULL));

        // complex value generator by giving twice the size.
        curandGenerateUniform(generator, (DType *)x_gpu.data, 2 * im_size);
        // xold = initialize random x of image size.
        curandDestroyGenerator(generator);
        // Create a handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        cublasScnrm2(handle, im_size, x_gpu.data, 1, &norm_old);
        inv_norm = 1.0 / norm_old;
        cublasCsscal(handle, im_size, &inv_norm, x_gpu.data, 1);

        for (int i = 0; i < max_iter; i++)
        {
          // compute x_new = adj_op(op(x_old))
          gpuNUFFTOp->performForwardGpuNUFFT(x_gpu, tmp_kspace_gpu);
          gpuNUFFTOp->performGpuNUFFTAdj(tmp_kspace_gpu, x_gpu);
          // compute ||x_new||
          cublasScnrm2(handle, im_size, x_gpu.data, 1, &norm_new);
          // x_new <- x_new/ ||x_new||
          inv_norm = 1.0 / norm_new;

          cublasCsscal(handle, im_size, &inv_norm, x_gpu.data, 1);
          if (fabs(norm_new - norm_old) < tolerance)
          {
                break;
          }
          norm_old = norm_new;
        }
        freeTotalDeviceMemory(tmp_kspace_gpu.data, x_gpu.data, NULL);
        cublasDestroy(handle);
        return norm_new;
    }
    ~GpuNUFFTPythonOperator()
    {
        delete gpuNUFFTOp;
    }
};

PYBIND11_MODULE(gpuNUFFT, m) {
    py::class_<GpuNUFFTPythonOperator>(m, "NUFFTOp")
        .def(py::init<py::array_t<DType>, py::array_t<int>, int, py::array_t<std::complex<DType>>, py::array_t<DType>, int, int, float, bool, int>(),
            py::arg("kspace_loc"), py::arg("image_size"), py::arg("num_coils"), py::arg("sense_maps") = py::none(), py::arg("density_comp") = py::none(), py::arg("kernel_width") = 3, py::arg("sector_width") = 8, py::arg("osr") = 2, py::arg("balance_workload") = true, py::arg("gpu_device_id") = 0)
        .def("op", &GpuNUFFTPythonOperator::op, py::arg("in_image"), py::arg("out_kspace"), py::arg("interpolate_data") = false)
        .def("op_direct", &GpuNUFFTPythonOperator::op_direct, py::arg("in_image"), py::arg("out_kspace"), py::arg("interpolate_data") = false)
        .def("adj_op_direct", &GpuNUFFTPythonOperator::adj_op_direct, py::arg("in_kspace"), py::arg("out_image"), py::arg("grid_data") = false)
        .def("adj_op",  &GpuNUFFTPythonOperator::adj_op, py::arg("in_kspace"), py::arg("out_image"),  py::arg("grid_data") = false)
        .def("toeplitz_op", &GpuNUFFTPythonOperator::toeplitz_op_direct, py::arg("in_image"), py::arg("out_image"), py::arg("kernel"))
        .def("clean_memory", &GpuNUFFTPythonOperator::clean_memory)
        .def("estimate_density_comp", &GpuNUFFTPythonOperator::estimate_density_comp, py::arg("max_iter") = 10)
        .def("set_smaps", &GpuNUFFTPythonOperator::set_smaps)
        .def("toggle_grad_mode", &GpuNUFFTPythonOperator::toggle_grad_mode)
        .def("get_spectral_radius", &GpuNUFFTPythonOperator::get_spectral_radius, py::arg("max_iter") = 20, py::arg("tolerance") = 1e-6)
        .def("set_pts", &GpuNUFFTPythonOperator::set_pts, py::arg("kspace_loc"), py::arg("density_comp") = py::none());
}
#endif  // GPUNUFFT_OPERATOR_PYTHONFACTORY_H_INCLUDED