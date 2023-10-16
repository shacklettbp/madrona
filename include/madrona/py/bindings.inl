#include <madrona/template_helpers.hpp>

#include <bit>
#include <utility>
#include <string>

namespace madrona::py {

template <auto iface_fn, auto cpu_step_fn, auto gpu_step_fn>
auto JAXInterface::buildEntry()
{
    using SimT =
        typename utils::ExtractClassFromMemberPtr<decltype(cpu_step_fn)>::type;

    return [](nb::object sim, bool xla_gpu) {
        void *fn;
        if (xla_gpu) {
            assert(gpu_step_fn != nullptr);
#ifdef MADRONA_CUDA_SUPPORT
            auto fn_wrapper =
                &JAXInterface::gpuStepFn<SimT, gpu_step_fn>;
            fn = std::bit_cast<void *>(fn_wrapper);
#else
            fn = nullptr;
#endif
        } else {
            auto fn_wrapper = &JAXInterface::cpuStepFn<SimT, cpu_step_fn>;
            fn = std::bit_cast<void *>(fn_wrapper);
        }
        assert(fn != nullptr);

        SimT *sim_ptr = nb::inst_ptr<SimT>(sim);
        TrainInterface iface = std::invoke(iface_fn, *sim_ptr);
        return setup(iface, sim, (void *)sim_ptr, fn, xla_gpu);
    };
}

template <typename SimT, auto cpu_step_fn>
void JAXInterface::cpuStepFn(void *, void **in)
{
    SimT *sim = *(SimT **)in[0];
    // FIXME: currently_broken, need to pass args
    std::invoke(cpu_step_fn, *sim);
}

#ifdef MADRONA_CUDA_SUPPORT
template <typename SimT, auto gpu_step_fn>
void JAXInterface::gpuStepFn(cudaStream_t strm, void **buffers,
                             const char *opaque, size_t)
{
    SimT *sim = *(SimT **)opaque;
    std::invoke(gpu_step_fn, *sim, strm, buffers);
}
#endif

}
