#pragma once

#include <madrona/py/utils.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace madrona::py {

class JAXInterface {
public:
    // Returns a function that registers custom_call_name with XLA
    // to call step_fn (or async_step_fn in GPU mode) and returns
    // the python function implementing the custom call.
    template <auto iface_fn, auto cpu_step_fn, auto gpu_step_fn = nullptr>
    static auto buildEntry();

private:
    template <typename SimT, auto cpu_step_fn>
    static void cpuStepFn(void *out, void **in);

#ifdef MADRONA_CUDA_SUPPORT
    template <typename SimT, auto gpu_step_fn>
    static void gpuStepFn(cudaStream_t strm, void **buffers,
                          const char *opaque, size_t opaque_len);
#endif

    static nb::object setup(const TrainInterface &iface,
                            nb::object sim_obj,
                            void *sim_ptr,
                            void *fn,
                            bool xla_gpu);
};

void setupMadronaSubmodule(nb::module_ parent_mod);

}

#include "bindings.inl"
