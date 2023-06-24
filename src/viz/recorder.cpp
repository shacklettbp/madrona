#include <madrona/viz/recorder.hpp>
#include "interop.hpp"

#include <madrona/utils.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace madrona::viz {

struct Recorder::Impl {
    VizECSBridge bridge;
    bool gpuCopyRequired;
    uint32_t numWorlds;
    
    uint32_t *curEpisodeLens;

    static Impl * init(const Config &cfg);

    void record();
};

Recorder::Impl * Recorder::Impl::init(const Config &cfg)
{
    bool gpu_copy_required = cfg.execMode == ExecMode::CUDA;

#ifdef MADRONA_CUDA_SUPPORT
    auto alloc = gpu_copy_required ? cu::allocGPU : malloc;
#else
    auto alloc = malloc;
#endif

    PerspectiveCameraData *view_data_buffer = (PerspectiveCameraData *)alloc(
        sizeof(PerspectiveCameraData) * cfg.maxViewsPerWorld * cfg.numWorlds);

    InstanceData *instance_data_buffer = (InstanceData *)alloc(
        sizeof(InstanceData) * cfg.maxInstancesPerWorld * cfg.numWorlds);

    bool *dones = (bool *)alloc(sizeof(bool) * cfg.numWorlds);

    PerspectiveCameraData **view_ptrs = (PerspectiveCameraData **)alloc(
        sizeof(PerspectiveCameraData *) * cfg.numWorlds);

    uint32_t *num_views = (uint32_t *)alloc(
        sizeof(uint32_t) * cfg.numWorlds);

    InstanceData **instance_ptrs = (InstanceData **)alloc(
        sizeof(InstanceData *) * cfg.numWorlds);

    uint32_t *num_instances = (uint32_t *)alloc(
        sizeof(uint32_t) * cfg.numWorlds);

    if (gpu_copy_required) {
#ifdef MADRONA_CUDA_SUPPORT
        auto view_ptrs_staging = (PerspectiveCameraData *)cu::allocStaging(
        sizeof(PerspectiveCameraData *) * cfg.numWorlds);

        auto instance_ptrs_staging = (InstanceData *)cu::allocStaging(
        sizeof(InstanceData *) * cfg.numWorlds);

        for (uint32_t i = 0; i < cfg.numWorlds; i++) {
            view_ptrs_staging[i] = view_data_buffer + i * cfg.maxViewsPerWorld;
            instance_ptrs_staging[i] =
                instance_data_buffer + i * cfg.maxInstancesPerWorld;
        }

        cudaMemcpy(view_ptrs, view_ptrs_staging,
                   sizeof(PerspectiveCameraData *) * cfg.numWorlds,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(instance_ptrs, instance_ptrs_staging,
                   sizeof(InstanceData *) * cfg.numWorlds,
                   cudaMemcpyHostToDevice);
#endif
    } else {
        for (uint32_t i = 0; i < cfg.numWorlds; i++) {
            view_ptrs[i] = view_data_buffer + i * cfg.maxViewsPerWorld;
            instance_ptrs[i] =
                instance_data_buffer + i * cfg.maxInstancesPerWorld;
        }
    }

    VizECSBridge bridge {
        .views = view_ptrs,
        .numViews = num_views,
        .instances = instance_ptrs,
        .numInstances = num_instances,
        .renderWidth = (int32_t)cfg.renderWidth,
        .renderHeight = (int32_t)cfg.renderHeight,
        .episodeDone = dones,
    };

    uint32_t *cur_episode_lens = (uint32_t *)malloc(
        sizeof(uint32_t) * cfg.numWorlds);
    utils::zeroN<uint32_t>(cur_episode_lens, cfg.numWorlds);

    return new Impl {
        bridge,
        gpu_copy_required,
        cfg.numWorlds,
        cur_episode_lens,
    };
}

void Recorder::Impl::record()
{
    for (uint32_t i = 0; i < numWorlds; i++) {
        curEpisodeLens[i]++;
    }
}

Recorder::Recorder(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

Recorder::Recorder(Recorder &&o) = default;
Recorder::~Recorder() = default;

const VizECSBridge * Recorder::bridge() const
{
    return &impl_->bridge;
}

void Recorder::record()
{
    impl_->record();
}

}
