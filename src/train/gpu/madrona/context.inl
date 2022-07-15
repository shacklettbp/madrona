#pragma once

#include <madrona/gpu_train/const.hpp>

namespace madrona {
namespace gpuTrain {

template <typename Fn, size_t N>
__global__ void jobEntry(gpuTrain::JobBase *job_data,
                        uint32_t num_invocations,
                        uint32_t grid_id)
{
    uint32_t lane_id = threadIdx.x % ICfg::numWarpThreads;

    uint32_t invocation_idx =
        threadIdx.x + blockIdx.x * ICfg::numJobLaunchKernelThreads;

    if (invocation_idx >= num_invocations) {
        return;
    }

    using JobContainer = gpuTrain::JobContainer<Fn, N>;
    JobContainer &job_container =
        static_cast<JobContainer *>(job_data)[invocation_idx];

    Context ctx(job_container.jobID, grid_id, job_container.worldID, lane_id);

    (job_container.fn)(ctx);

    // Calls the destructor for the functor
    job_container.~JobContainer();

    uint32_t num_block_invocations = min(
        num_invocations - blockIdx.x * ICfg::numJobLaunchKernelThreads,
        ICfg::numJobLaunchKernelThreads);

    ctx.markJobFinished(num_block_invocations);
}

}

Context::Context(uint32_t job_id, uint32_t grid_id, uint32_t world_id,
                 uint32_t lane_id)
    : job_id_(job_id),
      grid_id_(grid_id),
      world_id_(world_id),
      lane_id_(lane_id)
{}

StateManager & Context::state()
{
    return *(StateManager *)((char *)gpuTrain::GPUImplConsts::get().baseAddr +
         gpuTrain::GPUImplConsts::get().stateManagerOffset);
}

template <typename Fn, typename... Deps>
JobID Context::queueJob(Fn &&fn, bool is_child,
                        Deps && ...dependencies)
{
    return queueMultiJob(std::forward<Fn>(fn), 1, is_child,
                         std::forward<Deps>(dependencies)...);
}

template <typename Fn, typename... Deps>
JobID Context::queueMultiJob(Fn &&fn, uint32_t num_invocations,
                             bool is_child, Deps && ...dependencies)
{
    constexpr std::size_t num_deps = sizeof...(Deps);

    auto func_ptr = gpuTrain::jobEntry<Fn, num_deps>;

    using FnJobContainer = gpuTrain::JobContainer<Fn, num_deps>;

    auto wave_info = computeWaveInfo();

    JobID queue_job_id = getNewJobID(is_child);

    auto store = allocJob(sizeof(FnJobContainer), wave_info);
    new (store) FnJobContainer(world_id_, queue_job_id.id,
                               std::forward<Fn>(fn),
                               std::forward<Deps>(dependencies)...);

    __syncwarp(wave_info.activeMask);

    addToWaitList(func_ptr, store, num_invocations,
                  sizeof(FnJobContainer), lane_id_, wave_info);

    __syncwarp(wave_info.activeMask);

    return queue_job_id;
}

}
