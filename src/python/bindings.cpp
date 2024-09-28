#include <madrona/py/bindings.hpp>
#include <madrona/crash.hpp>

#include <nanobind/eval.h>

namespace nb = nanobind;

namespace madrona::py {

namespace {

struct JAXModule {
    nb::object mod;
    nb::object typeU8;
    nb::object typeI8;
    nb::object typeI16;
    nb::object typeI32;
    nb::object typeI64;
    nb::object typeF16;
    nb::object typeF32;
    nb::object typeShapeDtypeStruct;

    static inline JAXModule imp();
    inline nb::object getDType(TensorElementType type) const;
};

nb::dlpack::dtype toDLPackType(TensorElementType type)
{
    switch (type) {
        case TensorElementType::UInt8:
            return nb::dtype<uint8_t>();
        case TensorElementType::Int8:
            return nb::dtype<int8_t>();
        case TensorElementType::Int16:
            return nb::dtype<int16_t>();
        case TensorElementType::Int32:
            return nb::dtype<int32_t>();
        case TensorElementType::Int64:
            return nb::dtype<int64_t>();
        case TensorElementType::Float16:
            return nb::dlpack::dtype {
                static_cast<uint8_t>(nb::dlpack::dtype_code::Float),
                sizeof(int16_t) * 8,
                1,
            };
        case TensorElementType::Float32:
            return nb::dtype<float>();
        default: MADRONA_UNREACHABLE();
    }
}

auto tensor_to_pytorch(const Tensor &tensor)
{
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
}

auto tensor_to_jax(const Tensor &tensor)
{
    nb::dlpack::dtype type = toDLPackType(tensor.type());

    return nb::ndarray<nb::jax, void> {
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
}


JAXModule JAXModule::imp()
{
    nb::object mod = nb::module_::import_("jax");
    nb::object jnp = mod.attr("numpy");

    nb::object type_u8 = jnp.attr("uint8");
    nb::object type_i8 = jnp.attr("int8");
    nb::object type_i16 = jnp.attr("int16");
    nb::object type_i32 = jnp.attr("int32");
    nb::object type_i64 = jnp.attr("int64");
    nb::object type_f16 = jnp.attr("float16");
    nb::object type_f32 = jnp.attr("float32");

    nb::object type_shapedtype = mod.attr("ShapeDtypeStruct");

    return JAXModule {
        .mod = mod,
        .typeU8 = type_u8,
        .typeI8 = type_i8,
        .typeI16 = type_i16,
        .typeI32 = type_i32,
        .typeI64 = type_i64,
        .typeF16 = type_f16,
        .typeF32 = type_f32,
        .typeShapeDtypeStruct = type_shapedtype,
    };
}

nb::object JAXModule::getDType(TensorElementType type) const
{
    switch (type) {
        case TensorElementType::UInt8:
            return typeU8;
        case TensorElementType::Int8:
            return typeI8;
        case TensorElementType::Int16:
            return typeI16;
        case TensorElementType::Int32:
            return typeI32;
        case TensorElementType::Int64:
            return typeI64;
        case TensorElementType::Float16:
            return typeF16;
        case TensorElementType::Float32:
            return typeF32;
        default: MADRONA_UNREACHABLE();
    }
}

nb::object tensor_iface_to_jax(
    const JAXModule &jax_mod,
    TensorInterface iface)
{
    nb::list shape;

    for (int64_t d : iface.dimensions) {
        shape.append(d);
    }

    return jax_mod.typeShapeDtypeStruct(
        nb::arg("dtype") = jax_mod.getDType(iface.type),
        nb::arg("shape") = shape);
}

nb::dict train_interface_inputs_to_pytree(
    const JAXModule &jax_mod,
    const TrainInterface &iface)
{
    const TrainStepInputInterface &inputs = iface.stepInputs();

    nb::dict d;

    d["actions"] = tensor_iface_to_jax(jax_mod, inputs.actions);
    d["resets"] = tensor_iface_to_jax(jax_mod, inputs.resets);

    nb::dict pbt;
    for (const NamedTensorInterface &t : inputs.pbt) {
        pbt[t.name] = tensor_iface_to_jax(jax_mod, t.interface);
    }

    d["pbt"] = pbt;
    
    return d;
}

nb::dict train_interface_outputs_to_pytree(
    const JAXModule &jax_mod,
    const TrainInterface &iface)
{
    const TrainStepOutputInterface &outputs = iface.stepOutputs();

    nb::dict d;

    nb::dict obs;
    for (const NamedTensorInterface &t : outputs.observations) {
        obs[t.name] = tensor_iface_to_jax(jax_mod, t.interface);
    }

    d["obs"] = obs;

    d["rewards"] = tensor_iface_to_jax(jax_mod, outputs.rewards);
    d["dones"] = tensor_iface_to_jax(jax_mod, outputs.dones);

    nb::dict pbt;
    for (const NamedTensorInterface &t : outputs.pbt) {
        pbt[t.name] = tensor_iface_to_jax(jax_mod, t.interface);
    }

    d["pbt"] = pbt;

    return d;
}

nb::dict train_interface_checkpointing_to_pytree(
    const JAXModule &jax_mod,
    const TrainInterface &iface)
{
    nb::dict d;

    d["data"] = tensor_iface_to_jax(
        jax_mod, iface.checkpointing()->checkpointData);

    return d;
}

}

nb::dict JAXInterface::setup(const TrainInterface &iface,
                             nb::object sim_obj,
                             void *sim_ptr,
                             void *init_fn,
                             void *step_fn,
                             void *save_ckpts_fn,
                             void *restore_ckpts_fn,
                             bool xla_gpu)
{
    JAXModule jax_mod = JAXModule::imp();

    nb::capsule init_fn_capsule(init_fn, "xla._CUSTOM_CALL_TARGET");
    nb::capsule step_fn_capsule(step_fn, "xla._CUSTOM_CALL_TARGET");

    auto sim_encode = nb::bytes((char *)&sim_ptr, sizeof(char *));

    nb::dict input_iface = train_interface_inputs_to_pytree(jax_mod, iface);
    nb::dict output_iface = train_interface_outputs_to_pytree(jax_mod, iface);

    nb::dict scope;
    scope["sim_obj"] = sim_obj;
    scope["sim_ptr"] = (uint64_t)sim_ptr;
    scope["sim_encode"] = sim_encode;
    scope["step_inputs_iface"] = input_iface;
    scope["step_outputs_iface"] = output_iface;
    scope["init_custom_call_capsule"] = init_fn_capsule;
    scope["step_custom_call_capsule"] = step_fn_capsule;
    scope["custom_call_platform"] = xla_gpu ? "gpu" : "cpu";

    if (iface.checkpointing().has_value()) {
        assert(save_ckpts_fn != nullptr && restore_ckpts_fn != nullptr);

        nb::dict checkpointing_iface = train_interface_checkpointing_to_pytree(
            jax_mod, iface);

        nb::capsule save_ckpts_fn_capsule(
            save_ckpts_fn, "xla._CUSTOM_CALL_TARGET");
        nb::capsule restore_ckpts_fn_capsule(
            restore_ckpts_fn, "xla._CUSTOM_CALL_TARGET");

        scope["ckpt_iface"] = checkpointing_iface;
        scope["save_ckpts_custom_call_capsule"] = save_ckpts_fn_capsule;
        scope["restore_ckpts_custom_call_capsule"] = restore_ckpts_fn_capsule;
    } else {
        scope["ckpt_iface"] = nb::none();
    }

    nb::exec(
#include "jax_register.py"
    , scope);

    nb::dict fns;
    fns["init"] = scope["init_func"];
    fns["step"] = scope["step_func"];

    if (iface.checkpointing().has_value()) {
        fns["save_ckpts"] = scope["save_ckpts_func"];
        fns["restore_ckpts"] = scope["restore_ckpts_func"];
    }

    return fns;
}

static TensorElementType fromDLPackType(nb::dlpack::dtype dtype)
{
    using ET = TensorElementType;

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

    FATAL("Tensor: Invalid tensor dtype");
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
                FATAL("madrona::Tensor: failed to import unknown tensor type");
            }

            std::array<int64_t, Tensor::maxDimensions> dims;

            if (torch_tensor.ndim() > Tensor::maxDimensions) {
                FATAL("Cannot construct Tensor with more than %ld dimensions\n",
                      Tensor::maxDimensions);
            }

            for (size_t i = 0; i < torch_tensor.ndim(); i++) {
                dims[i] = torch_tensor.shape(i);
            }

            new (dst) Tensor(torch_tensor.data(),
                                fromDLPackType(torch_tensor.dtype()),
                                { dims.data(), (CountT)torch_tensor.ndim() },
                                gpu_id);
        }))
        .def("to_torch", tensor_to_pytorch, nb::rv_policy::automatic_reference)
        .def("to_jax", tensor_to_jax, nb::rv_policy::automatic_reference)
    ;

#ifdef MADRONA_CUDA_SUPPORT
    nb::class_<CudaSync>(m, "CudaSync")
        .def("wait", &CudaSync::wait)
    ;
#endif

    nb::class_<TrainInterface>(m, "TrainInterface")
        .def("step_inputs", [](const TrainInterface &iface) {
            return train_interface_inputs_to_pytree(
                JAXModule::imp(), iface);
        })
        .def("step_outputs", [](const TrainInterface &iface) {
            return train_interface_outputs_to_pytree(
                JAXModule::imp(), iface);
        })
    ;
}

}
