/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include "mw_gpu/const.hpp"
#include "mw_gpu/cu_utils.hpp"

namespace madrona {

Context::Context(WorldBase *world_data, const WorkerInit &init)
    : data_(world_data),
      world_id_(init.worldID)
{}

template <typename ArchetypeT>
Entity Context::makeEntity()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    return makeEntity(archetype_id);
}

Entity Context::makeEntity(uint32_t archetype_id)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    return state_mgr->makeEntityNow(world_id_, archetype_id);
}

template <typename ArchetypeT>
Loc Context::makeTemporary()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    return makeTemporary(archetype_id);
}

Loc Context::makeTemporary(uint32_t archetype_id)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    return state_mgr->makeTemporary(world_id_, archetype_id);
}

void Context::destroyEntity(Entity e)
{
    return mwGPU::getStateManager()->destroyEntityNow(e);
}

Loc Context::loc(Entity e) const
{
    return mwGPU::getStateManager()->getLoc(e);
}

template <typename ComponentT>
ComponentT & Context::get(Entity e)
{
    return mwGPU::getStateManager()->getUnsafe<ComponentT>(e);
}

template <typename ComponentT>
ComponentT & Context::get(Loc l)
{
    return mwGPU::getStateManager()->getUnsafe<ComponentT>(l);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getSafe(Entity e)
{
    return mwGPU::getStateManager()->get<ComponentT>(e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getCheck(Entity e)
{
    return mwGPU::getStateManager()->get<ComponentT>(e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getCheck(Loc l)
{
    return mwGPU::getStateManager()->get<ComponentT>(l);
}

template <typename ComponentT>
ComponentT & Context::getDirect(int32_t column_idx, Loc l)
{
    return mwGPU::getStateManager()->getDirect<ComponentT>(column_idx, l);
}

template <typename SingletonT>
SingletonT & Context::singleton()
{
    return mwGPU::getStateManager()->getSingleton<SingletonT>(world_id_);
}

inline void * Context::tmpAlloc(uint64_t num_bytes)
{
    return mwGPU::TmpAllocator::get().alloc(num_bytes);
}

template <typename... ComponentTs>
inline Query<ComponentTs...> Context::query() 
{
    return mwGPU::getStateManager()->query<ComponentTs...>();
}
    
template <typename... ComponentTs, typename Fn>
inline void Context::iterateQuery(Query<ComponentTs...> &query, Fn&& fn) 
{
    mwGPU::getStateManager()->iterateQuery<sizeof...(ComponentTs)>(
        world_id_.idx,
        query.getSharedRef(),
    [&](int32_t offset, auto ...raw_ptrs){
        // offset is a global offset computed from world offset and 
        // a count of the current entity within the archetype table.
        cuda::std::tuple typed_ptrs {
            (ComponentTs *)raw_ptrs
            ...
        };
        
        std::apply([&](auto ...ptrs) {
            fn(ptrs[offset] ...);
        }, typed_ptrs);
    });
}

#if 0

class Context {
    template <typename Fn, typename... DepTs>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        DepTs && ...dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
        bool is_child = true, DepTs && ...dependencies);

    template <typename... ColTypes, typename Fn, typename... DepTs>
    inline JobID parallelFor(const Query<ColTypes...> &query, Fn &&fn,
        bool is_child = true, DepTs && ... dependencies);

#if 0
    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitImpl(Fn &&fn, bool is_child,
                            DepTs && ... dependencies);

    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                             DepTs && ... dependencies);

    template <typename ContextT, typename... ComponentTs,
              typename Fn, typename... DepTs>
    inline JobID parallelForImpl(const Query<ComponentTs...> &query, Fn &&fn,
                                 bool is_child, DepTs && ... dependencies);
#endif

    void markJobFinished();

    inline JobID currentJobID() const { return job_id_; }

    inline StateManager & state();

    WaveInfo computeWaveInfo();

    JobID waveSetupNewJob(uint32_t func_id, bool link_parent,
                          uint32_t num_invocations, uint32_t bytes_per_job,
                          void **thread_data_store);

    JobContainerBase * allocJob(uint32_t bytes_per_job, WaveInfo wave_info);

    inline void stageChildJob(uint32_t func_id, uint32_t num_combined_jobs,
                              uint32_t bytes_per_job, void *containers);

    JobID job_id_;

    uint32_t lane_id_;
}

namespace mwGPU {

// This function is executed at the thread-block granularity.
// num_invocations <= consts::numMegakernelThreads
template <typename ContextT, typename ContainerT>
__attribute__((used, always_inline))
inline void jobEntry(JobContainerBase *job_data,
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

    static_assert(std::is_trivially_destructible_v<ContainerT>);

    ContainerT &job_container =
        static_cast<ContainerT *>(job_data)[data_idx];

    ContextT ctx = JobManager::makeContext<ContextT>(job_container.jobID,
        grid_id, job_container.worldID, lane_id);

    (job_container.fn)(ctx, local_offset);

    ctx.markJobFinished(num_invocations);
}

template <typename ContextT, typename ContainerT>
struct JobFuncIDBase {
    static uint32_t id;
};

template <typename ContextT, typename ContainerT,
          decltype(jobEntry<ContextT, ContainerT>) =
            jobEntry<ContextT, ContainerT>>
struct JobFuncID : JobFuncIDBase<ContextT, ContainerT> {};

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

template <typename ContextT, typename Fn, typename... DepTs>
JobID Context::submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                           DepTs &&... dependencies)
{
    using namespace mwGPU;

    constexpr std::size_t num_deps = sizeof...(DepTs);

    using ContainerT = JobContainer<Fn, num_deps>;

    UserJobTracker *user_trackers =
        JobManager::get()->getUserJobTrackers();

    auto wave_info = computeWaveInfo();

    // Consolidate dependencies across warp
    // num_deps is guaranteed to be the same across activemask here
#if 0
    uint32_t num_unique_dependencies = 0;
    auto countUniqueDeps = [&](JobID dep) {
        UserJobTracker &user_tracker = user_trackers[dep.id];

        uint32_t shared_id = dep.gen == user_tracker.gen ?
            user_tracker.sharedID : ~0_u32;

        uint32_t match_mask = __match_any_sync(wave_info.activeMask,
                                               shared_id);

        uint32_t top_match = getHighestSetBit(match_mask);

        uint32_t num_add = (top_match == lane_id_ && shared_id != ~0_u32) ?
            1 : 0;
        num_unique_dependencies +=
            __reduce_add_sync(wave_info.activeMask, num_add);
    };
#endif

    uint32_t func_id = mwGPU::JobFuncID<ContextT, ContainerT>::id;

    void *thread_data_store;
    JobID job_id = waveSetupNewJob(func_id, is_child, num_invocations,
        sizeof(ContainerT), &thread_data_store);

    new (thread_data_store) ContainerT(job_id, world_id_,
        num_invocations,
        std::forward<Fn>(fn),
        std::forward<DepTs>(dependencies)...);

    return queue_job_id;
}
#endif

}
