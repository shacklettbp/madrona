/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include "mw_gpu/const.hpp"

namespace madrona {
namespace mwGPU {

template <typename ContextT, typename Fn, size_t N>
__global__ void jobEntry(JobContainerBase *job_data,
                         uint32_t *data_indices,
                         uint32_t *invocation_offsets,
                         uint32_t num_invocations,
                         uint32_t grid_id)
{
    uint32_t lane_id = threadIdx.x % consts::numWarpThreads;

    uint32_t invocation_idx =
        threadIdx.x + blockIdx.x * consts::numJobLaunchKernelThreads;

    if (invocation_idx >= num_invocations) {
        return;
    }

    uint32_t data_idx = data_indices[invocation_idx];
    uint32_t local_offset = invocation_offsets[invocation_idx];

    using JobContainer = JobContainer<Fn, N>;
    static_assert(std::is_trivially_destructible_v<JobContainer>);

    JobContainer &job_container =
        static_cast<JobContainer *>(job_data)[data_idx];

    ContextT ctx = JobManager::makeContext<ContextT>(job_container.jobID,
        grid_id, job_container.worldID, lane_id);

    (job_container.fn)(ctx, local_offset);

    uint32_t num_block_invocations = min(
        num_invocations - blockIdx.x * consts::numJobLaunchKernelThreads,
        consts::numJobLaunchKernelThreads);

    ctx.markJobFinished(num_block_invocations);
}

}

Context::Context(WorldBase *world_data, WorkerInit &&init)
    : data_(world_data),
      job_id_(init.jobID),
      grid_id_(init.gridID),
      world_id_(init.worldID),
      lane_id_(init.laneID)
{}

StateManager & Context::state()
{
    return *(StateManager *)mwGPU::GPUImplConsts::get().stateManagerAddr;
}

template <typename Fn, typename... DepTs>
JobID Context::submit(Fn &&fn, bool is_child, DepTs && ... dependencies)
{
    return submitImpl<Context>(std::forward<Fn>(fn), 1, is_child,
                               std::forward<Deps>(dependencies)...);
}

template <typename Fn, typename... DepTs>
JobID Context::submitN(Fn &&fn, uint32_t num_invocations,
                       bool is_child, DepTs && ... dependencies)
{
    return submitImpl<Context>(std::forward<Fn>(fn), num_invocations, is_child,
                               std::forward<DepTs>(dependencies)...);
}

template <typename... ColTypes, typename Fn, typename... DepTs>
JobID Context::parallelFor(const Query<ColTypes...> &query, Fn &&fn,
                           bool is_child, DepTs && ... dependencies)
{
    return parallelForImpl<Context>(query, std::forward<Fn>(fn), is_child,
                                    std::forward<DepTs>(dependencies)...);
}


template <typename ContextT, typename Fn, typename... DepTs>
JobID Context::submitImpl(Fn &&fn, bool is_child, DepTs && ... dependencies)
{
    auto wrapper = [fn = std::forward<Fn>(fn)](ContextT &ctx, uint32_t) {
        fn(ctx);
    };

    return submitNImpl<ContextT>(std::move(wrapper), 1, is_child,
                                 std::forward<DepTs>(dependencies)...);
}

// FIXME: implement is_child, dependencies, num_invocations
template <typename ContextT, typename Fn, typename... DepTs>
JobID Context::submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                           DepTs &&... dependencies)
{
    constexpr std::size_t num_deps = sizeof...(DepTs);

    auto func_ptr = mwGPU::jobEntry<ContextT, Fn, num_deps>;

    using FnJobContainer = JobContainer<Fn, num_deps>;

    auto wave_info = computeWaveInfo();

    JobID queue_job_id = getNewJobID(is_child, num_invocations);

    auto store = allocJob(sizeof(FnJobContainer), wave_info);
    new (store) FnJobContainer(queue_job_id, world_id_,
                               num_invocations,
                               std::forward<Fn>(fn),
                               std::forward<DepTs>(dependencies)...);

    __syncwarp(wave_info.activeMask);

    addToWaitList(func_ptr, store, num_invocations, sizeof(FnJobContainer),
                  lane_id_, wave_info);

    __syncwarp(wave_info.activeMask);

    return queue_job_id;
}

}
