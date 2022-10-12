/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/mw_gpu/const.hpp>

namespace madrona {
namespace mwGPU {

template <typename ContextT, typename Fn, size_t N>
__global__ void jobEntry(mwGPU::JobBase *job_data,
                        uint32_t num_invocations,
                        uint32_t grid_id)
{
    uint32_t lane_id = threadIdx.x % consts::numWarpThreads;

    uint32_t invocation_idx =
        threadIdx.x + blockIdx.x * consts::numJobLaunchKernelThreads;

    if (invocation_idx >= num_invocations) {
        return;
    }

    using JobContainer = mwGPU::JobContainer<Fn, N>;
    JobContainer &job_container =
        static_cast<JobContainer *>(job_data)[invocation_idx];

    WorkerInit worker_init {
        .jobID = job_container.jobID,
        .gridID = grid_id,
        .worldID = job_container.worldID,
        .laneID = lane_id,
    };

    constexpr bool is_generic_context = std::is_same_v<ContextT, Context>;

    using DataT = std::conditional_t<is_generic_context, WorldBase,
        typename ContextT::WorldDataT>;

    DataT *world_data;

    // If this is being called with the generic Context base class,
    // we need to look up the size of the world data in constant memory
    if constexpr (is_generic_context) {
        char *world_data_base = 
            (char *)mwGPU::GPUImplConsts::get().worldDataAddr;
        world_data = (DataT *)(world_data_base + worker_init.worldID *
            mwGPU::GPUImplConsts::get().numWorldDataBytes);
    } else {
        DataT *world_data_base =
            (DataT *)mwGPU::GPUImplConsts::get().worldDataAddr;

        world_data = world_data_base + worker_init.worldID;
    }

    ContextT ctx(world_data, std::move(worker_init));

    (job_container.fn)(ctx);

    // Calls the destructor for the functor
    job_container.~JobContainer();

    uint32_t num_block_invocations = min(
        num_invocations - blockIdx.x * ICfg::numJobLaunchKernelThreads,
        ICfg::numJobLaunchKernelThreads);

    ctx.markJobFinished(num_block_invocations);
}

}

Context::Context(WorkerInit &&init)
    : job_id_(init.jobID),
      grid_id_(init.gridID),
      world_id_(init.worldID),
      lane_id_(init.laneID)
{}

StateManager & Context::state()
{
    return *(StateManager *)mwGPU::GPUImplConsts::get().stateManagerAddr;
}

template <typename Fn, typename... Deps>
JobID Context::submit(Fn &&fn, bool is_child, Deps && ... dependencies)
{
    return submitImpl<Context>(std::forward<Fn>(fn), 1, is_child,
                               std::forward<Deps>(dependencies)...);
}

template <typename Fn, typename... Deps>
JobID Context::submitN(Fn &&fn, uint32_t num_invocations,
                       bool is_child, Deps && ... dependencies)
{
    return submitImpl<Context>(std::forward<Fn>(fn), num_invocations, is_child,
                               std::forward<Deps>(dependencies)...);
}

template <typename... ColTypes, typename Fn, typename... Deps>
JobID Context::forAll(const Query<ColTypes...> &query, Fn &&fn,
             bool is_child, Deps && ... dependencies)
{
    return forallImpl<Context>(query, std::forward<Fn>(fn), is_child,
                               std::forward<Deps>(dependencies)...);
}

// FIXME: implement is_child, dependencies, num_invocations
template <typename ContextT, typename Fn, typename... Deps>
JobID Context::submitImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                          Deps &&... dependencies)
{
    constexpr std::size_t num_deps = sizeof...(Deps);

    auto func_ptr = mwGPU::jobEntry<ContextT, Fn, num_deps>;

    using FnJobContainer = mwGPU::JobContainer<Fn, num_deps>;

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
