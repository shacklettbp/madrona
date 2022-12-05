#include <madrona/python.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

#include <iostream>

namespace nb = nanobind;

namespace madrona {
namespace py {

MADRONA_EXPORT ExternalSync::ExternalSync(cudaExternalSemaphore_t sema)
    : sema_(sema)
{}

MADRONA_EXPORT void ExternalSync::wait(uint64_t strm)
{
    // Get the current CUDA stream from pytorch and force it to wait
    // on an external semaphore to finish
    cudaStream_t cuda_strm = (cudaStream_t)strm;
    cudaExternalSemaphoreWaitParams params {};
    cudaError_t res =
        cudaWaitExternalSemaphoresAsync(&sema_, &params, 1, cuda_strm);
    if (res != cudaSuccess) {
        std::cerr << "Madrona Pytorch Integration: failed to wait on external semaphore"
             << std::endl;
        abort();
    }
}

MADRONA_EXPORT GPUTensor::GPUTensor(void *dev_ptr, ElementType type,
                                    Span<const int64_t> dimensions, int gpu_id)
    : dev_ptr_(dev_ptr),
      type_(type),
      gpu_id_(gpu_id),
      num_dimensions_(dimensions.size()),
      dimensions_()
{
    assert(num_dimensions_ <= maxDimensions);
    memcpy(dimensions_.data(), dimensions.data(),
           num_dimensions_ * sizeof(int64_t));
}

static nb::dlpack::dtype toDLPackType(GPUTensor::ElementType type)
{
    switch (type) {
        case GPUTensor::ElementType::UInt8:
            return nb::dtype<uint8_t>();
        case GPUTensor::ElementType::Int8:
            return nb::dtype<int8_t>();
        case GPUTensor::ElementType::Int16:
            return nb::dtype<int16_t>();
        case GPUTensor::ElementType::Int32:
            return nb::dtype<int32_t>();
        case GPUTensor::ElementType::Int64:
            return nb::dtype<int64_t>();
        case GPUTensor::ElementType::Float16:
            return nb::dlpack::dtype {
                static_cast<uint8_t>(nb::dlpack::dtype_code::Float),
                sizeof(int16_t) * 8,
                1,
            };
        case GPUTensor::ElementType::Float32:
            return nb::dtype<float>();
    }
}

static GPUTensor::ElementType fromDLPackType(nb::dlpack::dtype dtype)
{
    using ET = GPUTensor::ElementType;

    if (nb::dlpack::dtype_code(dtype.code) == nb::dlpack::dtype_code::Int) {
        switch (dtype.bits) {
        case 8:
            return ET::Int8;
        case 16:
            return ET::Int16;
        case 32:
            return ET::Int32;
        case 64:
            return ET::Int64;
        default:
            break;
        }
    } else if (nb::dlpack::dtype_code(dtype.code) ==
               nb::dlpack::dtype_code::UInt && dtype.bits == 8) {
        return ET::UInt8;
    } else if (nb::dlpack::dtype_code(dtype.code) ==
               nb::dlpack::dtype_code::Float) {
        if (dtype.bits == 16) {
            return ET::Float16;
        }
        if (dtype.bits == 32) {
            return ET::Float32;
        }
    }

    std::cerr << "GPUTensor: Invalid tensor dtype" << std::endl;
    abort();
}

NB_MODULE(madrona_python, m) {
    nb::class_<GPUTensor>(m, "GPUTensor")
        .def("__init__", ([](GPUTensor *dst, nb::tensor<> torch_tensor) {
            if (torch_tensor.device_type() != nb::device::cuda::value) {
                std::cerr << "Cannot construct GPUTensor from a non CUDA tensor"
                    << std::endl;
                abort();
            }

            std::array<int64_t, GPUTensor::maxDimensions> dims;

            if (torch_tensor.ndim() > GPUTensor::maxDimensions) {
                std::cerr << "Cannot construct GPUTensor with more than "
                    << GPUTensor::maxDimensions << " dimensions" << std::endl;
                abort();
            }

            for (size_t i = 0; i < torch_tensor.ndim(); i++) {
                dims[i] = torch_tensor.shape(i);
            }

            new (dst) GPUTensor(torch_tensor.data(),
                                fromDLPackType(torch_tensor.dtype()),
                                Span(dims.data(), torch_tensor.ndim()),
                                torch_tensor.device_id());
        }))
        .def("to_torch", [](const GPUTensor &tensor) {
            nb::dlpack::dtype type = toDLPackType(tensor.type());

            return nb::tensor<nb::pytorch, void> {
                tensor.devicePtr(),
                (size_t)tensor.numDims(),
                (const size_t *)tensor.dims(),
                nb::handle(),
                nullptr,
                type,
                nb::device::cuda::value,
                tensor.gpuID(),
            };
        });

    nb::class_<ExternalSync>(m, "ExternalSync")
        .def("wait", &ExternalSync::wait);
}

}
}
