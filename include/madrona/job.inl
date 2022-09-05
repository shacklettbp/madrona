#pragma once

#include <type_traits>

namespace madrona {

JobID JobID::none()
{
    return JobID {
        ~0u,
        ~0u,
    };
}


template <typename Fn, size_t N>
template <typename... DepTs>
JobContainer<Fn, N>::JobContainer(Fn &&func, DepTs ...deps)
    : JobContainerBase {
          .id = JobID::none(), // Assigned in JobManager::queueJob
          .numDependencies = N,
      },
      dependencies { deps ... },
      fn(std::forward<Fn>(func))
{}

template <typename DataT, typename StartFn>
struct JobManager::EntryConfig {
    DataT ctxData;
    void (*ctxInitCB)(void *, void *, WorkerInit &&);
    uint32_t numCtxBytes;
    uint32_t ctxAlignment;
    StartFn startData;
    void (*startCB)(Context &, void *);
};

template <typename ContextT, typename DataT, typename StartFn>
JobManager::EntryConfig<DataT, StartFn> JobManager::makeEntry(
    DataT &&ctx_data, StartFn &&start_fn)
{
    static_assert(std::is_trivially_destructible_v<ContextT>,
                  "Context types with custom destructors are not supported");

    static_assert(std::is_trivially_copyable_v<DataT>,
                  "Context data must be trivially copyable");

    return {
        std::forward<DataT>(ctx_data),
        [](void *ctx, void *data, WorkerInit &&init) {
            new (ctx) ContextT(data, std::forward<WorkerInit>(init));
        },
        sizeof(ContextT),
        std::alignment_of_v<ContextT>,
        std::forward<StartFn>(start_fn),
        [](Context &ctx_base, void *data) {
            auto fn_ptr = (StartFn *)data;
            ContextT &ctx = static_cast<ContextT &>(ctx_base);
            (*fn_ptr)(ctx);
            fn_ptr->~StartFn();
        },
    };
}

template <typename DataT, typename StartFn>
JobManager::JobManager(const EntryConfig<DataT, StartFn> &entry_cfg,
                       int desired_num_workers,
                       int num_io,
                       StateManager *state_mgr,
                       bool pin_workers)
    : JobManager((void *)&entry_cfg.ctxData,
                 sizeof(DataT),
                 alignof(DataT),
                 entry_cfg.ctxInitCB,
                 entry_cfg.numCtxBytes,
                 entry_cfg.ctxAlignment,
                 entry_cfg.startCB,
                 (void *)&entry_cfg.startData,
                 desired_num_workers,
                 num_io,
                 state_mgr,
                 pin_workers)
{}

JobID JobManager::getProxyJobID(JobID parent_id)
{
    return getProxyJobID(parent_id.idx);
}

template <typename ContextT, typename Fn, typename... DepTs>
JobID JobManager::queueJob(int thread_idx, Fn &&fn, uint32_t num_invocations,
                           JobID parent_id, JobPriority prio, DepTs ...deps)
{
    static constexpr uint32_t num_deps = sizeof...(DepTs);
    using ContainerT = JobContainer<Fn, num_deps>;
    static_assert(num_deps == 0 ||
        offsetof(ContainerT, dependencies) == sizeof(JobContainerBase),
        "Dependencies at incorrect offset in container type");

    static constexpr size_t job_size = sizeof(ContainerT);
    static constexpr size_t job_alignment = std::alignment_of_v<ContainerT>;
    static_assert(job_size <= JobManager::Alloc::maxJobSize,
                  "Job lambda capture is too large");
    static_assert(job_alignment <= JobManager::Alloc::maxJobAlignment,
        "Job lambda capture has too large an alignment requirement");
    static_assert(utils::isPower2(job_alignment));

    void *store = allocJob(thread_idx, job_size, job_alignment);

    auto container = new (store) ContainerT(std::forward<Fn>(fn), num_invocations,
                                            deps...);

    Job::EntryPtr stateless_ptr = [](Context &ctx_base, JobContainerBase *data,
                                     uint32_t invocation_idx) {
        ContextT &ctx = static_cast<ContextT &>(ctx_base);

        auto container = static_cast<ContainerT *>(data);
        container->fn(ctx, invocation_idx);
        container->~ContainerT();

        ctx.job_mgr_->markJobFinished(ctx.worker_idx_, container, job_size);
    };

    return queueJob(thread_idx, stateless_ptr, container, num_invocations,
                    parent_id.idx, prio);
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
