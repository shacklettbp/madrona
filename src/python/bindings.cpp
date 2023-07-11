#include <madrona/py/bindings.hpp>

#include <iostream>

namespace nb = nanobind;

namespace madrona::py {

static nb::dlpack::dtype toDLPackType(Tensor::ElementType type)
{
    switch (type) {
        case Tensor::ElementType::UInt8:
            return nb::dtype<uint8_t>();
        case Tensor::ElementType::Int8:
            return nb::dtype<int8_t>();
        case Tensor::ElementType::Int16:
            return nb::dtype<int16_t>();
        case Tensor::ElementType::Int32:
            return nb::dtype<int32_t>();
        case Tensor::ElementType::Int64:
            return nb::dtype<int64_t>();
        case Tensor::ElementType::Float16:
            return nb::dlpack::dtype {
                static_cast<uint8_t>(nb::dlpack::dtype_code::Float),
                sizeof(int16_t) * 8,
                1,
            };
        case Tensor::ElementType::Float32:
            return nb::dtype<float>();
        default: MADRONA_UNREACHABLE();
    }
}

static Tensor::ElementType fromDLPackType(nb::dlpack::dtype dtype)
{
    using ET = Tensor::ElementType;

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

    std::cerr << "Tensor: Invalid tensor dtype" << std::endl;
    exit(1);
}

void setupMadronaSubmodule(nb::module_ parent_mod)
{
    auto m = parent_mod.def_submodule("madrona");

      nb::class_<madrona::py::PyExecMode>(m, "ExecMode")
        .def_prop_ro_static("CPU", [](nb::handle) {
            return madrona::py::PyExecMode(madrona::ExecMode::CPU);
        })
        .def_prop_ro_static("CUDA", [](nb::handle) {
            return madrona::py::PyExecMode(madrona::ExecMode::CUDA);
        });

    nb::class_<Tensor>(m, "Tensor")
        .def("__init__", ([](Tensor *dst, nb::ndarray<> torch_tensor) {
            Optional<int> gpu_id = Optional<int>::none();
            if (torch_tensor.device_type() == nb::device::cuda::value) {
                gpu_id = torch_tensor.device_id();
            } else if (torch_tensor.device_type() != nb::device::cpu::value) {
                std::cerr <<
                    "madrona::Tensor: failed to import unknown tensor type" <<
                    std::endl;
                abort();
            }

            std::array<int64_t, Tensor::maxDimensions> dims;

            if (torch_tensor.ndim() > Tensor::maxDimensions) {
                std::cerr << "Cannot construct Tensor with more than "
                    << Tensor::maxDimensions << " dimensions" << std::endl;
                abort();
            }

            for (size_t i = 0; i < torch_tensor.ndim(); i++) {
                dims[i] = torch_tensor.shape(i);
            }

            new (dst) Tensor(torch_tensor.data(),
                                fromDLPackType(torch_tensor.dtype()),
                                { dims.data(), (CountT)torch_tensor.ndim() },
                                gpu_id);
        }))
        .def("to_torch", [](const Tensor &tensor) {
            nb::dlpack::dtype type = toDLPackType(tensor.type());

            return nb::ndarray<nb::pytorch, void> {
                tensor.devicePtr(),
                (size_t)tensor.numDims(),
                (const size_t *)tensor.dims(),
                nb::handle(),
                nullptr,
                type,
                tensor.isOnGPU() ?
                    nb::device::cuda::value :
                    nb::device::cpu::value,
                tensor.isOnGPU() ? tensor.gpuID() : 0,
            };
        }, nb::rv_policy::automatic_reference);

#ifdef MADRONA_CUDA_SUPPORT
    nb::class_<CudaSync>(m, "CudaSync")
        .def("wait", &CudaSync::wait);
#endif
}

}
