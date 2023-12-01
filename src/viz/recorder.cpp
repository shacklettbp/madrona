#include <madrona/viz/recorder.hpp>
#include <madrona/viz/interop.hpp>

#include <madrona/utils.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace madrona::viz {

struct Recorder::Impl {
    RenderECSBridge bridge;
    bool gpuCopyRequired;
    uint32_t numWorlds;
    uint64_t numInstanceCopyBytes;
    
    uint32_t * curEpisodeLengths;
    bool * doneReadback;
    InstanceData * curEpisodeInstanceData;
    PerspectiveCameraData * curEpisodeViewData;

    InstanceData * recordedInstances;
    PerspectiveCameraData * recordedViews;
    uint32_t * recordedEpisodeLengths;
    uint32_t numRecordedEpisodes;

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

    PerspectiveCameraData *interop_view_data = (PerspectiveCameraData *)alloc(
        sizeof(PerspectiveCameraData) * cfg.maxViewsPerWorld * cfg.numWorlds);

    uint64_t max_instances_per_step = cfg.maxInstancesPerWorld * cfg.numWorlds;
    uint64_t num_instance_data_bytes_per_step =
        sizeof(InstanceData) * (uint64_t)max_instances_per_step;

    InstanceData *interop_instance_data = (InstanceData *)alloc(
        num_instance_data_bytes_per_step);

    bool *dones = (bool *)alloc(sizeof(bool) * cfg.numWorlds);

    PerspectiveCameraData **interop_view_ptrs =
        (PerspectiveCameraData **)alloc(
            sizeof(PerspectiveCameraData *) * cfg.numWorlds);

    uint32_t *interop_num_views = (uint32_t *)alloc(
        sizeof(uint32_t) * cfg.numWorlds);

    InstanceData **interop_instance_ptrs = (InstanceData **)alloc(
        sizeof(InstanceData *) * cfg.numWorlds);

    uint32_t *interop_num_instances = (uint32_t *)alloc(
        sizeof(uint32_t) * cfg.numWorlds);

    bool *done_readback;
    if (gpu_copy_required) {
#ifdef MADRONA_CUDA_SUPPORT
        auto interop_view_ptrs_staging =
            (PerspectiveCameraData *)cu::allocStaging(
                sizeof(PerspectiveCameraData *) * cfg.numWorlds);

        auto interop_instance_ptrs_staging = (InstanceData *)cu::allocStaging(
            sizeof(InstanceData *) * cfg.numWorlds);

        for (uint32_t i = 0; i < cfg.numWorlds; i++) {
            view_ptrs_staging[i] =
                interop_view_data + i * cfg.maxViewsPerWorld;
            instance_ptrs_staging[i] =
                interop_instance_data + i * cfg.maxInstancesPerWorld;
        }

        cudaMemcpy(interop_view_ptrs, interop_view_ptrs_staging,
                   sizeof(PerspectiveCameraData *) * cfg.numWorlds,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(interop_instance_ptrs, interop_instance_ptrs_staging,
                   sizeof(InstanceData *) * cfg.numWorlds,
                   cudaMemcpyHostToDevice);

        done_readback = (bool *)cu::allocReadback(
            sizeof(bool) * cfg.numWorlds);
#endif
    } else {
        for (uint32_t i = 0; i < cfg.numWorlds; i++) {
            interop_view_ptrs[i] =
                interop_view_data + i * cfg.maxViewsPerWorld;
            interop_instance_ptrs[i] =
                interop_instance_data + i * cfg.maxInstancesPerWorld;
        }

        done_readback = nullptr;
    }

    RenderECSBridge bridge {
        .views = interop_view_ptrs,
        .numViews = interop_num_views,
        .instances = interop_instance_ptrs,
        .numInstances = interop_num_instances,
        .renderWidth = (int32_t)cfg.renderWidth,
        .renderHeight = (int32_t)cfg.renderHeight,
        .episodeDone = dones,
    };

    uint32_t *cur_episode_lens = (uint32_t *)malloc(
        sizeof(uint32_t) * cfg.numWorlds);
    utils::zeroN<uint32_t>(cur_episode_lens, cfg.numWorlds);

    RenderECSBridge bridge;
    bool gpuCopyRequired;
    uint32_t numWorlds;
    uint64_t numInstanceCopyBytes;
    
    uint32_t * curEpisodeLens;
    bool * doneReadback;
    InstanceData * curEpisodeInstanceData;
    PerspectiveCameraData * curEpisodeViewData;

    InstanceData * recordedInstances;
    PerspectiveCameraData * recordedViews;
    uint32_t * recordedEpisodeLengths;
    uint32_t numRecordedEpisodes;

    return new Impl {
        .bridge = bridge,
        .gpuCopyRequired = gpu_copy_required,
        .numWorlds = cfg.numWorlds,
        .numInstanceCopyBytes = num_instance_data_bytes_per_step,
        .curEpisodeLengths = cur_episode_lens,
        .doneReadback = done_readback,
        .curEpisodeInstance
    };
}

void Recorder::Impl::record()
{
    bool *episode_dones;
    if (gpuCopyRequired) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(curEpisodeInstanceData, bridge.instancePtrs[0],
                   numInstanceCopyBytes, cudaMemcpyDeviceToHost);

        cudaMemcpy(doneReadback, bridge.episodeDones,
                   sizeof(bool) * numWorlds, cudaMemcpyDeviceToHost);

        episode_dones = doneReadback;
#endif
    } else {
        memcpy(cudaEpisodeInstanceData, bridge.instancePtrs[0],
               numInstanceCopyBytes);

        episode_dones = bridge.episodeDones;
    }

    for (uint32_t i = 0; i < numWorlds; i++) {
        uint32_t episode_len = ++curEpisodeLengths[i];

        if (episode_dones[i]) {

        }
    }
}

Recorder::Recorder(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

Recorder::Recorder(Recorder &&o) = default;
Recorder::~Recorder() = default;

const RenderECSBridge * Recorder::bridge() const
{
    return &impl_->bridge;
}

void Recorder::record()
{
    impl_->record();
}

}
