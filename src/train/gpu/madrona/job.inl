#include <type_traits>

namespace madrona {

namespace gpuTrain {

namespace ICfg {

static constexpr uint32_t numWarpThreads = 32;
static constexpr uint32_t numJobLaunchKernelThreads = 512;

}

JobBase::JobBase(uint32_t world_id)
    : worldID(world_id)
{}

template <typename Fn>
JobContainer<Fn>::JobContainer(uint32_t world_id, Fn &&func)
    : JobBase(world_id),
      fn(std::forward<Fn>(func))
{}

template <typename Fn>
__global__ void jobEntry(gpuTrain::JobBase *job_data,
                         uint32_t num_launches,
                         uint32_t grid_id)
{
    uint32_t lane_id = threadIdx.x % ICfg::numWarpThreads;

    uint32_t invocation_idx =
        threadIdx.x + blockIdx.x * ICfg::numJobLaunchKernelThreads;

    if (invocation_idx >= num_launches) {
        return;
    }

    using JobContainer = gpuTrain::JobContainer<Fn>;
    JobContainer &job_container =
        static_cast<JobContainer *>(job_data)[invocation_idx];

    Context ctx(grid_id, job_container.worldID, lane_id);

    (*(job_container.fn))(ctx);

    // Calls the destructor for the functor
    job_container.~JobContainer();

    uint32_t num_block_launches = min(
        num_launches - blockIdx.x * ICfg::numJobLaunchKernelThreads,
        ICfg::numJobLaunchKernelThreads);

    ctx.markJobFinished(num_block_launches);
}

}

Context::Context(uint32_t grid_id, uint32_t world_id, uint32_t lane_id)
    : grid_id_(grid_id),
      world_id_(world_id),
      lane_id_(lane_id)
{}

template <typename Fn>
void Context::queueJob(Fn &&fn)
{
    auto func_ptr = gpuTrain::jobEntry<Fn>;

    using JobContainer = gpuTrain::JobContainer<Fn>;

    auto wave_info = computeWaveInfo();

    gpuTrain::JobBase *base_store;
    if (lane_id_ == wave_info.leaderLane) {
        base_store = (gpuTrain::JobBase *)allocJob(
            sizeof(JobContainer) * wave_info.numActive);
    }

    base_store = (gpuTrain::JobBase *)__shfl_sync(wave_info.activeMask,
        (uintptr_t)base_store, wave_info.leaderLane);

    void *store = (char *)base_store +
        sizeof(JobContainer) * wave_info.coalescedIDX;

    new (store) JobContainer(world_id_, std::forward<Fn>(fn));

    if (lane_id_ == wave_info.leaderLane) {
        queueJob(func_ptr, base_store, wave_info.numActive,
                 sizeof(JobContainer));
    }

    __syncwarp();
}

}
