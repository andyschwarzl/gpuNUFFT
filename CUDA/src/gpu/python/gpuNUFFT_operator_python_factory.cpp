#ifndef GPUNUFFT_OPERATOR_PYTHON_FACTORY_H_INCLUDED
#define GPUNUFFT_OPERATOR_PYTHON_FACTORY_H_INCLUDED
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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


class GpuNUFFTPythonOperator
{
    gpuNUFFT::GpuNUFFTOperatorFactory factory;
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp;
    int trajectory_length, n_coils, dimension;
    gpuNUFFT::Dimensions imgDims;
    public:
    GpuNUFFTPythonOperator(py::array_t<DType> kspace_loc, py::array_t<int> image_size, py::array_t<std::complex<DType>> sense_maps,
    py::array_t<float> density_comp, int kernel_width=3, int sector_width=8, int osr=2, bool balance_workload=1)
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

        // sensitivity maps
        gpuNUFFT::Array<DType2> sensArray;
        DType2 *sensData = NULL;
        py::buffer_info sense_maps_buffer = sense_maps.request();

        if (sense_maps_buffer.size==0)
            n_coils = 1;
        else
        {
            n_coils = sense_maps_buffer.shape[0];
            sensData = (DType2 *) sense_maps_buffer.ptr;
        }

        sensArray.data = sensData;
        sensArray.dim = imgDims;
        sensArray.dim.channels = n_coils;

        factory.setBalanceWorkload(balance_workload);
        gpuNUFFTOp = factory.createGpuNUFFTOperator(
            kSpaceTraj, density_compArray, sensArray, kernel_width, sector_width,
            osr, imgDims);
        cudaThreadSynchronize();
    }

    py::array_t<std::complex<DType>> op(py::array_t<std::complex<DType>> image)
    {
        py::array_t<std::complex<DType>> out_result({n_coils, trajectory_length});
        py::buffer_info out = out_result.request();
        std::complex<DType> *t_data = (std::complex<DType> *) out.ptr;
        DType2 *new_data = reinterpret_cast<DType2(&)[0]>(*t_data);
        gpuNUFFT::Array<CufftType> dataArray;
        dataArray.data = new_data;
        dataArray.dim.length = trajectory_length;
        dataArray.dim.channels = n_coils;

        gpuNUFFT::Array<DType2> imdataArray = readNumpyArray(image);
        imdataArray.dim = imgDims;
        imdataArray.dim.channels = n_coils;
        gpuNUFFTOp->performForwardGpuNUFFT(imdataArray, dataArray);
        cudaThreadSynchronize();
        return out_result;
    }
    py::array_t<std::complex<DType>> adj_op(py::array_t<std::complex<DType>> kspace_data)
    {
        int depth = imgDims.depth;
        if(dimension==2)
            depth = 1;
        py::array_t<std::complex<DType>> out_result({depth, (int)imgDims.height, (int)imgDims.width});
        py::buffer_info out = out_result.request();
        std::complex<DType> *t_data = (std::complex<DType> *) out.ptr;
        DType2 *new_data = reinterpret_cast<DType2(&)[0]>(*t_data);
        gpuNUFFT::Array<DType2> imdataArray;
        imdataArray.data = new_data;
        imdataArray.dim = imgDims;

        gpuNUFFT::Array<CufftType> dataArray = readNumpyArray(kspace_data);
        dataArray.dim.length = trajectory_length;
        gpuNUFFTOp->performGpuNUFFTAdj(dataArray, imdataArray);
        cudaThreadSynchronize();
        return out_result;
    }
    ~GpuNUFFTPythonOperator()
    {
        delete gpuNUFFTOp;
    }
};
PYBIND11_MODULE(gpuNUFFT, m) {
    py::class_<GpuNUFFTPythonOperator>(m, "NUFFTOp")
        .def(py::init<py::array_t<float>, py::array_t<int>, py::array_t<std::complex<float>>, py::array_t<float>, int, int, int, bool>())
        .def("op", &GpuNUFFTPythonOperator::op)
        .def("adj_op",  &GpuNUFFTPythonOperator::adj_op);
}
#endif  // GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED
