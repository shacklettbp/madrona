#include <type_traits>

namespace madrona {

namespace gpuTrain {

namespace ICfg {

static constexpr uint32_t numWarpThreads = 32;
static constexpr uint32_t numJobLaunchKernelThreads = 512;

}

JobBase::JobBase(uint32_t world_id, uint32_t job_id, uint32_t num_deps)
    : worldID(world_id),
      jobID(job_id),
      numDependencies(num_deps)
{}

template <size_t N>
template <typename... Args>
JobDependenciesBase<N>::JobDependenciesBase(uint32_t world_id, uint32_t job_id,
                                            Args && ...args)
    : JobBase(world_id, job_id, N),
      deps {
          std::forward<Args>(args)...
      }
{}

template <typename Fn, size_t N>
template <typename... Args>
JobContainer<Fn, N>::JobContainer(uint32_t world_id, uint32_t job_id, Fn &&func,
                                  Args && ...args)
    : JobDependenciesBase<N>(world_id, job_id, std::forward<Args>(args)...),
      fn(std::forward<Fn>(func))
{}

template <typename Fn, size_t N>
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

    using JobContainer = gpuTrain::JobContainer<Fn, N>;
    JobContainer &job_container =
        static_cast<JobContainer *>(job_data)[invocation_idx];

    Context ctx(job_container.jobID, grid_id, job_container.worldID, lane_id);

    (*(job_container.fn))(ctx);

    // Calls the destructor for the functor
    job_container.~JobContainer();

    uint32_t num_block_launches = min(
        num_launches - blockIdx.x * ICfg::numJobLaunchKernelThreads,
        ICfg::numJobLaunchKernelThreads);

    ctx.markJobFinished(num_block_launches);
}

}

Context::Context(uint32_t job_id, uint32_t grid_id, uint32_t world_id,
                 uint32_t lane_id)
    : job_id_(job_id),
      grid_id_(grid_id),
      world_id_(world_id),
      lane_id_(lane_id)
{}

template <typename Fn, typename... Args>
JobID Context::queueJob(Fn &&fn, bool is_child, Args && ...dependencies)
{
    constexpr std::size_t num_deps = sizeof...(Args);

    auto func_ptr = gpuTrain::jobEntry<Fn, num_deps>;

    using JobContainer = gpuTrain::JobContainer<Fn, num_deps>;

    auto wave_info = computeWaveInfo();

    gpuTrain::JobBase *base_store;
    JobID queue_job_id;
    if (lane_id_ == wave_info.leaderLane) {
        base_store = (gpuTrain::JobBase *)allocJob(
            sizeof(JobContainer) * wave_info.numActive);

        queue_job_id = getNewJobID(wave_info.numActive, is_child);
    }

    // Sync store point & id with wave
    base_store = (gpuTrain::JobBase *)__shfl_sync(wave_info.activeMask,
        (uintptr_t)base_store, wave_info.leaderLane);
    queue_job_id.gen = __shfl_sync(wave_info.activeMask, queue_job_id.gen,
                                   wave_info.leaderLane);
    queue_job_id.id = __shfl_sync(wave_info.activeMask, queue_job_id.id,
                                   wave_info.leaderLane);

    void *store = (char *)base_store +
        sizeof(JobContainer) * wave_info.coalescedIDX;

    new (store) JobContainer(world_id_, queue_job_id.id, std::forward<Fn>(fn),
                             std::forward<Args>(dependencies)...);

    if (lane_id_ == wave_info.leaderLane) {
        addToWaitList(func_ptr, base_store, wave_info.numActive,
                      sizeof(JobContainer));
    }

    return queue_job_id;
}

}
