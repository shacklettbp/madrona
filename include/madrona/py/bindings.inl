#include <madrona/template_helpers.hpp>

#include <bit>
#include <utility>
#include <string>

namespace madrona::py {

template <auto iface_fn,
          auto cpu_init_fn,
          auto cpu_step_fn,
          auto gpu_init_fn,
          auto gpu_step_fn,
          auto cpu_save_ckpts_fn,
          auto cpu_restore_ckpts_fn,
          auto gpu_save_ckpts_fn,
          auto gpu_restore_ckpts_fn>
auto JAXInterface::buildEntry()
{
    using SimT =
        typename utils::ExtractClassFromMemberPtr<decltype(cpu_step_fn)>::type;

    return [](nb::object sim, bool xla_gpu) {
        void *init_fn;
        void *step_fn;
        void *save_ckpts_fn;
        void *restore_ckpts_fn;
        if (xla_gpu) {
#ifdef MADRONA_CUDA_SUPPORT
            if constexpr (gpu_init_fn != nullptr &&
                          gpu_step_fn != nullptr) {
                auto init_wrapper =
                    &JAXInterface::gpuEntryFn<SimT, gpu_init_fn>;
                init_fn = std::bit_cast<void *>(init_wrapper);

                auto step_wrapper =
                    &JAXInterface::gpuEntryFn<SimT, gpu_step_fn>;
                step_fn = std::bit_cast<void *>(step_wrapper);
            } else {
                init_fn = nullptr;
                step_fn = nullptr;
            }

            if constexpr (gpu_save_ckpts_fn != nullptr &&
                    gpu_restore_ckpts_fn != nullptr) {
                auto gpu_save_ckpts_wrapper =
                    &JAXInterface::gpuEntryFn<
                        SimT, gpu_save_ckpts_fn>;
                save_ckpts_fn = std::bit_cast<void *>(
                    gpu_save_ckpts_wrapper);

                auto gpu_restore_ckpts_wrapper =
                    &JAXInterface::gpuEntryFn<
                        SimT, gpu_restore_ckpts_fn>;
                restore_ckpts_fn = std::bit_cast<void *>(
                    gpu_restore_ckpts_wrapper);
            } else {
                save_ckpts_fn = nullptr;
                restore_ckpts_fn = nullptr;
            }
#else
            init_fn = nullptr;
            step_fn = nullptr;
            save_ckpts_fn = nullptr;
            restore_ckpts_fn = nullptr;
#endif
        } else {
            auto init_wrapper =
                &JAXInterface::cpuEntryFn<SimT, cpu_init_fn>;
            init_fn = std::bit_cast<void *>(init_wrapper);

            auto step_wrapper =
                &JAXInterface::cpuEntryFn<SimT, cpu_step_fn>;
            step_fn = std::bit_cast<void *>(step_wrapper);

            if constexpr (cpu_save_ckpts_fn != nullptr &&
                    cpu_restore_ckpts_fn != nullptr) {
                auto cpu_save_ckpts_wrapper =
                    &JAXInterface::cpuEntryFn<
                        SimT, cpu_save_ckpts_fn>;
                save_ckpts_fn = std::bit_cast<void *>(
                    cpu_save_ckpts_wrapper);

                auto cpu_restore_ckpts_wrapper =
                    &JAXInterface::cpuEntryFn<
                        SimT, cpu_restore_ckpts_fn>;
                restore_ckpts_fn = std::bit_cast<void *>(
                    cpu_restore_ckpts_wrapper);
            } else {
                save_ckpts_fn = nullptr;
                restore_ckpts_fn = nullptr;
            }
        }
        assert(init_fn != nullptr && step_fn != nullptr);

        SimT *sim_ptr = nb::inst_ptr<SimT>(sim);
        TrainInterface iface = std::invoke(iface_fn, *sim_ptr);
        return setup(iface, sim, (void *)sim_ptr, init_fn, step_fn,
                     save_ckpts_fn, restore_ckpts_fn, xla_gpu);
    };
}

template <typename SimT, auto fn>
void JAXInterface::cpuEntryFn(void **out, void **in)
{
    SimT *sim = *(SimT **)in[0];
    std::invoke(fn, *sim, in + 2, out);
}

#ifdef MADRONA_CUDA_SUPPORT
template <typename SimT, auto fn>
void JAXInterface::gpuEntryFn(cudaStream_t strm, void **buffers,
                              const char *opaque, size_t)
{
    SimT *sim = *(SimT **)opaque;

    // The first buffer entry is used by the CPU backend, skip
    // The scond buffer entry is a token JAX uses for ordering, skip
    std::invoke(fn, *sim, strm, buffers + 2);
}
#endif

}
