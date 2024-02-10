#include <madrona/template_helpers.hpp>

#include <bit>
#include <utility>
#include <string>

namespace madrona::py {

template <auto iface_fn,
          auto cpu_init_fn,
          auto cpu_step_fn,
          auto gpu_init_fn,
          auto gpu_step_fn>
auto JAXInterface::buildEntry()
{
    using SimT =
        typename utils::ExtractClassFromMemberPtr<decltype(cpu_step_fn)>::type;

    return [](nb::object sim, bool xla_gpu) {
        void *init_fn;
        void *step_fn;
        if (xla_gpu) {
            static_assert(gpu_init_fn != nullptr && gpu_step_fn != nullptr);
#ifdef MADRONA_CUDA_SUPPORT
            auto init_wrapper =
                &JAXInterface::gpuEntryFn<SimT, gpu_init_fn>;
            init_fn = std::bit_cast<void *>(init_wrapper);

            auto step_wrapper =
                &JAXInterface::gpuEntryFn<SimT, gpu_step_fn>;
            step_fn = std::bit_cast<void *>(step_wrapper);
#else
            init_fn = nullptr;
            step_fn = nullptr;
#endif
        } else {
            auto init_wrapper =
                &JAXInterface::cpuEntryFn<SimT, cpu_init_fn>;
            init_fn = std::bit_cast<void *>(init_wrapper);

            auto step_wrapper =
                &JAXInterface::cpuEntryFn<SimT, cpu_step_fn>;
            step_fn = std::bit_cast<void *>(step_wrapper);
        }
        assert(init_fn != nullptr && step_fn != nullptr);

        SimT *sim_ptr = nb::inst_ptr<SimT>(sim);
        TrainInterface iface = std::invoke(iface_fn, *sim_ptr);
        return setup(iface, sim, (void *)sim_ptr, init_fn, step_fn, xla_gpu);
    };
}

template <typename SimT, auto fn>
void JAXInterface::cpuEntryFn(void *, void **in)
{
    SimT *sim = *(SimT **)in[0];
    // FIXME: currently_broken, need to pass args
    std::invoke(fn, *sim);
}

#ifdef MADRONA_CUDA_SUPPORT
template <typename SimT, auto fn>
void JAXInterface::gpuEntryFn(cudaStream_t strm, void **buffers,
                              const char *opaque, size_t)
{
    SimT *sim = *(SimT **)opaque;
    std::invoke(fn, *sim, strm, buffers);
}
#endif

}
