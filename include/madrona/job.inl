/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <type_traits>

namespace madrona {

constexpr JobID JobID::none()
{
    return JobID {
        0xFFFF'FFFF_u32,
        0xFFFF'FFFF_i32,
    };
}

template <size_t N>
struct JobContainerBase::DepsArray {
    JobID dependencies[N];

    template <typename... DepTs>
    inline DepsArray(DepTs ...deps)
        : dependencies { deps ... }
    {}
};

template <> struct JobContainerBase::DepsArray<0> {
    template <typename... DepTs>
    inline DepsArray(DepTs...) {}
};

template <typename Fn, size_t N>
template <typename... DepTs>
JobContainer<Fn, N>::JobContainer(uint32_t job_size,
                                  MADRONA_MW_COND(uint32_t world_id,)
                                  Fn &&func,
                                  DepTs ...deps)
    : JobContainerBase {
          .id = JobID::none(), // Assigned in JobManager::queueJob
          .jobSize = job_size,
          MADRONA_MW_COND(.worldID = world_id,)
          .numDependencies = N,
      },
      dependencies(deps...),
      fn(std::forward<Fn>(func))
{}

bool JobManager::isQueueEmpty(uint32_t head,
                              uint32_t correction,
                              uint32_t tail) const
{
    auto checkGEWrapped = [](uint32_t a, uint32_t b) {
        return a - b <= (1u << 31u);
    };

    return checkGEWrapped(head - correction, tail);
}

template <typename StartFn, typename UpdateFn>
struct JobManager::EntryConfig {
    uint32_t numUserdataBytes;
    uint32_t userdataAlignment;
    void (*ctxInitCB)(void *, void *, WorkerInit &&);
    uint32_t numCtxBytes;
    uint32_t ctxAlignment;
    StartFn startFnData;
    void (*startWrapper)(Context *, void *);
    UpdateFn updateFnData;
    void (*updateLoop)(Context *, void *);
};

template <typename ContextT, typename StartFn>
JobManager::EntryConfig<StartFn, void (*)(Context *, void *)>
    JobManager::makeEntry(StartFn &&start_fn)
{
    return makeEntry<ContextT, StartFn, void (*)(Context *, void *)>(
        std::forward<StartFn>(start_fn), nullptr);
}

template <typename ContextT, typename StartFn,
          typename UpdateFn>
JobManager::EntryConfig<StartFn, UpdateFn> JobManager::makeEntry(
    StartFn &&start_fn, UpdateFn &&update_fn)
{
    static_assert(std::is_trivially_destructible_v<ContextT>,
                  "Context types with custom destructors are not supported");

    void (*start_wrapper)(Context *, void *);
    if constexpr (!std::is_same_v<StartFn, void (*)(Context *, void *)>) {
        start_wrapper = [](Context *ctx_base, void *data) {
            auto &ctx = *static_cast<ContextT *>(ctx_base);
            auto fn_ptr = (StartFn *)data;

            ctx.submit([fn = StartFn(*fn_ptr)](ContextT &ctx) {
                fn(ctx);
            }, false, ctx.currentJobID());
        };
    } else {
        start_wrapper = start_fn;
        start_fn = nullptr;
    }

    void (*update_wrapper)(Context *, void *);
    if constexpr (!std::is_same_v<UpdateFn, void (*)(Context *, void *)>) {
        update_wrapper = [](Context *ctx_base, void *data) {
            auto &ctx = *static_cast<ContextT *>(ctx_base);
            auto fn_ptr = (UpdateFn *)data;

            ctx.submit([fn = UpdateFn(*fn_ptr)](ContextT &ctx) {
                fn(ctx);
            }, false, ctx.currentJobID());
        };
    } else {
        update_wrapper = update_fn;
        update_fn = nullptr;
    }

    using DataT = typename ContextT::WorldDataT;

    return {
        sizeof(DataT),
        alignof(DataT),
        [](void *ctx, void *data, WorkerInit &&init) {
            new (ctx) ContextT((DataT *)data, std::forward<WorkerInit>(init));
        },
        sizeof(ContextT),
        std::alignment_of_v<ContextT>,
        std::forward<StartFn>(start_fn),
        start_wrapper,
        std::forward<UpdateFn>(update_fn),
        update_wrapper,
    };
}

template <typename StartFn, typename UpdateFn>
JobManager::JobManager(const EntryConfig<StartFn, UpdateFn> &entry_cfg,
                       int desired_num_workers,
                       int num_io,
                       StateManager *state_mgr,
                       bool pin_workers)
    : JobManager(entry_cfg.numUserdataBytes,
                 entry_cfg.userdataAlignment,
                 entry_cfg.ctxInitCB,
                 entry_cfg.numCtxBytes,
                 entry_cfg.ctxAlignment,
                 entry_cfg.startWrapper,
                 [&entry_cfg]() {
                     if constexpr (std::is_same_v<StartFn,
                            void (*)(Context *, void *)>) {
                         (void)entry_cfg;
                         return nullptr;
                     } else {
                         return (void *)&entry_cfg.startFnData;
                     }
                 }(),
                 entry_cfg.updateLoop,
                 [&entry_cfg]() {
                     if constexpr (std::is_same_v<UpdateFn,
                            void (*)(Context *, void *)>) {
                         (void)entry_cfg;
                         return nullptr;
                     } else {
                         return (void *)&entry_cfg.updateFnData;
                     }
                 }(),
                 desired_num_workers,
                 num_io,
                 state_mgr,
                 pin_workers)
{}

JobID JobManager::reserveProxyJobID(int thread_idx, JobID parent_id)
{
    return reserveProxyJobID(thread_idx, parent_id.id);
}

void JobManager::relinquishProxyJobID(int thread_idx, JobID job_id)
{
    return markInvocationsFinished(thread_idx, nullptr, job_id.id, 1);
}

bool JobManager::shouldSplitJob(RunQueue *queue) const
{
    uint32_t cur_tail = queue->tail.load_relaxed();
    uint32_t cur_correction = queue->correction.load_relaxed();
    uint32_t cur_head = queue->head.load_relaxed();

    return isQueueEmpty(cur_head, cur_correction, cur_tail);
}

template <typename ContextT, typename ContainerT>
void JobManager::singleInvokeEntry(Context *ctx_base,
                                   JobContainerBase *data)
{
    ContextT &ctx = *static_cast<ContextT *>(ctx_base);
    auto container = static_cast<ContainerT *>(data);
    JobManager *job_mgr = ctx.job_mgr_;
    
    container->fn(ctx);

    job_mgr->markInvocationsFinished(ctx.worker_idx_, data, data->id.id, 1);
}

template <typename ContextT, typename ContainerT>
void JobManager::multiInvokeEntry(Context *ctx_base,
                                  JobContainerBase *data,
                                  uint64_t invocation_offset,
                                  uint64_t num_invocations,
                                  RunQueue *thread_queue)
{
    ContextT &ctx = *static_cast<ContextT *>(ctx_base);
    auto container = static_cast<ContainerT *>(data);
    JobManager *job_mgr = ctx.job_mgr_;

    // This loop is never called with num_invocations == 0
    uint64_t invocation_idx = invocation_offset;
    uint64_t remaining_invocations = num_invocations;
    do {
        uint64_t cur_invocation = invocation_idx++;
        remaining_invocations -= 1;

        if (remaining_invocations > 0 &&
                job_mgr->shouldSplitJob(thread_queue)) {
            job_mgr->splitJob(&multiInvokeEntry<ContextT, ContainerT>, data,
                invocation_idx, remaining_invocations, thread_queue);
            remaining_invocations = 0;
        }

        container->fn(ctx, cur_invocation);
    } while (remaining_invocations > 0);

    job_mgr->markInvocationsFinished(ctx.worker_idx_, data, data->id.id,
        invocation_idx - invocation_offset);
}

template <typename ContextT, bool single_invoke, typename Fn,
          typename... DepTs>
JobID JobManager::queueJob(int thread_idx,
                           Fn &&fn,
                           uint32_t num_invocations,
                           JobID parent_id,
                           MADRONA_MW_COND(uint32_t world_id,)
                           JobPriority prio,
                           DepTs ...deps)
{
    static constexpr uint32_t num_deps = sizeof...(DepTs);
    using ContainerT = JobContainer<Fn, num_deps>;
    static_assert(std::is_trivially_destructible_v<ContainerT>);

#ifdef MADRONA_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
    static_assert(num_deps == 0 ||
        offsetof(ContainerT, dependencies) == sizeof(JobContainerBase),
        "Dependencies at incorrect offset in container type");
#ifdef MADRONA_GCC
#pragma GCC diagnostic pop
#endif

    static constexpr uint64_t job_size = sizeof(ContainerT);
    static constexpr uint64_t job_alignment = alignof(ContainerT);
    static_assert(job_size <= JobManager::Alloc::maxJobSize,
                  "Job lambda capture is too large");
    static_assert(job_alignment <= JobManager::Alloc::maxJobAlignment,
        "Job lambda capture has too large an alignment requirement");
    static_assert(utils::isPower2(job_alignment));

    void *store = allocJob(thread_idx, job_size, job_alignment);

    auto container = new (store) ContainerT(
        job_size, MADRONA_MW_COND(world_id,) std::forward<Fn>(fn), deps...);

    void (*entry)();
    if constexpr (single_invoke) {
        SingleInvokeFn fn_ptr = &singleInvokeEntry<ContextT, ContainerT>;
        entry = (void (*)())fn_ptr;
    } else {
        MultiInvokeFn fn_ptr = &multiInvokeEntry<ContextT, ContainerT>;
        entry = (void (*)())fn_ptr;
    }

    return queueJob(thread_idx, entry, container, num_invocations,
                    parent_id.id, prio);
}

void * JobManager::allocJob(int worker_idx, uint32_t num_bytes,
                            uint32_t alignment)
{
    return job_allocs_[worker_idx].alloc(alloc_state_, num_bytes,
                                         alignment);
}

void JobManager::deallocJob(int worker_idx, void *ptr, uint32_t num_bytes)
{
    job_allocs_[worker_idx].dealloc(alloc_state_, ptr, num_bytes);
}

}
