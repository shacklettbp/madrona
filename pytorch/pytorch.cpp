#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.hpp>
#include <cuda_runtime.h>

using namespace std;
namespace py = pybind11;

namespace madrona {
namespace pytorch {

enum class ElementType {
    UInt8,
    Int8,
    Int16,
    Int32,
    Float16,
    Float32,
};

class ExternalSync {
public:
    ExternalSync(const py::capsule &cap) : sema_(cudaExternalSemaphore_t(cap))
    {}

    void wait()
    {
        // Get the current CUDA stream from pytorch and force it to wait
        // on an external semaphore to finish
        cudaStream_t cuda_strm = at::cuda::getCurrentCUDAStream().stream();
        cudaExternalSemaphoreWaitParams params {};
        cudaError_t res =
            cudaWaitExternalSemaphoresAsync(&sema_, &params, 1, cuda_strm);
        if (res != cudaSuccess) {
            cerr << "Madrona Pytorch Integration: failed to wait on external semaphore"
                 << endl;
            abort();
        }
    }

private:
    cudaExternalSemaphore_t sema_;
};

at::Tensor makeTensor(const py::capsule &ptr_capsule,
                      int dev_id,
                      const vector<int64_t> &dimensions,
                      ElementType type)
{
    void *dev_ptr(ptr_capsule);

    at::ScalarType dtype;

    switch (type) {
        case ElementType::UInt8:
            dtype = torch::kUInt8;
            break;
        case ElementType::Int8:
            dtype = torch::kInt8;
            break;
        case ElementType::Int16:
            dtype = torch::kInt16;
            break;
        case ElementType::Int32:
            dtype = torch::kInt16;
            break;
        case ElementType::Float16:
            dtype = torch::kFloat16;
            break;
        case ElementType::Float32:
            dtype = torch::kFloat32;
            break;
    }

    auto options = torch::TensorOptions()
                       .dtype(dtype)
                       .device(torch::kCUDA, (short)dev_id);

    c10::ArrayRef<int64_t> dims_ref(dimensions.data(), dimensions.size());

    return torch::from_blob(dev_ptr, dims_ref, options);
}

py::capsule tensorData(const at::Tensor &tensor)
{
    return py::capsule(tensor.data_ptr());
}

}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<madrona::pytorch::ElementType>(m, "ElementType")
        .value("u8", madrona::pytorch::ElementType::UInt8)
        .value("i8", madrona::pytorch::ElementType::Int8)
        .value("i16", madrona::pytorch::ElementType::Int16)
        .value("i32", madrona::pytorch::ElementType::Int32)
        .value("f16", madrona::pytorch::ElementType::Float16)
        .value("f32", madrona::pytorch::ElementType::Float32)
        .export_values();

    py::class_<madrona::pytorch::ExternalSync>(m, "ExternalSync")
        .def(py::init<const py::capsule &>())
        .def("wait", &madrona::pytorch::ExternalSync::wait);

    m.def("make_tensor", &madrona::pytorch::makeTensor);
    m.def("tensor_data", &madrona::pytorch::makeTensor);
}
