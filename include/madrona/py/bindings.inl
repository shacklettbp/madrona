#include <madrona/template_helpers.hpp>

#include <bit>
#include <utility>
#include <string>

namespace madrona::py {

template <auto iface_fn, auto step_fn, auto async_step_fn>
auto XLAInterface::buildEntry()
{
    using SimT =
        typename utils::ExtractClassFromMemberPtr<decltype(step_fn)>::type;

    return [](nb::object sim, bool xla_gpu) {
        void *fn;
        if (xla_gpu) {
            auto fn_wrapper =
                &XLAInterface::gpuStepFn<SimT, step_fn, async_step_fn>;
            fn = std::bit_cast<void *>(fn_wrapper);
        } else {
            auto fn_wrapper = &XLAInterface::cpuStepFn<SimT, step_fn>;
            fn = std::bit_cast<void *>(fn_wrapper);
        }

        SimT *sim_ptr = nb::inst_ptr<SimT>(sim);
        TrainInterface iface = std::invoke(iface_fn, *sim_ptr);
        return setup(iface, sim, (void *)sim_ptr, fn, xla_gpu);
    };
}

template <typename SimT, auto step_fn>
void XLAInterface::cpuStepFn(void *, void **in)
{
    SimT *sim = *(SimT **)in[0];
    std::invoke(step_fn, *sim);
}

#ifdef MADRONA_CUDA_SUPPORT
template <typename SimT, auto step_fn, auto async_step_fn>
void XLAInterface::gpuStepFn(cudaStream_t strm, void **,
                             const char *opaque, size_t)
{
    SimT *sim = *(SimT **)opaque;

    if constexpr (async_step_fn == nullptr) {
        REQ_CUDA(cudaStreamSynchronize(strm));
        std::invoke(step_fn, *sim);
    } else {
        std::invoke(async_step_fn, *sim);
    }
}
#endif

}
