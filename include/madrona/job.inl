#pragma once

#include <type_traits>

namespace madrona {

template <typename DataT, typename StartFn>
struct JobManager::Init {
    StartFn startData;
    Job::EntryPtr startCB;
    void (*ctxInitCB)(void *, void *, WorkerInit &&);
    DataT ctxData;
    StateManager *stateMgr;
    int desiredNumWorkers;
    int numIOWorkers;
    uint32_t numCtxBytes;
    uint32_t ctxAlignment;
    bool pinWorkers;
};

template <typename ContextT, typename DataT, typename StartFn>
JobManager::Init<DataT, StartFn> JobManager::makeInit(
    int desired_num_workers, int num_io, StateManager &state_mgr,
    const DataT &ctx_data, StartFn &&start_fn, bool pin_workers)
{
    static_assert(std::is_trivially_destructible_v<ContextT>,
                  "Context types with custom destructors are not supported");

    static_assert(std::is_trivially_copyable_v<DataT>,
                  "Context data must be trivially copyable");

    return {
        std::move(start_fn),
        [](Context &ctx_base, void *data, uint32_t invocation_idx) {
            auto fn_ptr = (StartFn *)data;
            ContextT &ctx = static_cast<ContextT &>(ctx_base);
            (*fn_ptr)(ctx, invocation_idx);
            fn_ptr->~StartFn();
        },
        [](void *ctx, void *data, WorkerInit &&init) {
            new (ctx) ContextT(data, std::forward<WorkerInit>(init));
        },
        ctx_data,
        &state_mgr,
        desired_num_workers,
        num_io,
        sizeof(ContextT),
        std::alignment_of_v<ContextT>,
        pin_workers,
    };
}

template <typename DataT, typename StartFn>
JobManager::JobManager(const Init<DataT, StartFn> &init)
    : JobManager(init.desiredNumWorkers, init.numIOWorkers,
                 init.stateMgr, init.numCtxBytes, init.ctxAlignment,
                 init.ctxInitCB, (void *)&init.ctxData, sizeof(DataT),
                 std::alignment_of_v<DataT>, init.startCB,
                 (void *)&init.startData, init.pinWorkers)
{}

JobID JobManager::queueJob(int thread_idx, Job job,
                           const JobID *deps, uint32_t num_dependencies,
                           JobPriority prio)
{
    return queueJob(thread_idx, job.func_, job.data_, deps,
                    num_dependencies, prio);
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
