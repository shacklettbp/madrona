#pragma once

#include <type_traits>

namespace madrona {

template <typename StartFn>
struct JobManager::Init {
    StartFn startData;
    Job::EntryPtr startCB;
    void (*ctxInitCB)(void *, void *, WorkerInit &&);
    void *ctxData;
    StateManager *stateMgr;
    int desiredNumWorkers;
    int numIOWorkers;
    uint32_t numCtxBytes;
    uint32_t ctxAlignment;
    bool pinWorkers;
};

template <typename ContextT, typename DataT, typename StartFn>
JobManager::Init<StartFn> JobManager::makeInit(
    int desired_num_workers, int num_io, StateManager &state_mgr,
    DataT *ctx_data, StartFn &&start_fn, bool pin_workers)
{
    static_assert(std::is_trivially_destructible_v<ContextT>,
                  "Context types with custom destructors are not supported");

    return {
        std::move(start_fn),
        [](Context &ctx_base, void *data) {
            auto fn_ptr = (StartFn *)data;
            ContextT &ctx = static_cast<ContextT &>(ctx_base);
            (*fn_ptr)(ctx);
            fn_ptr->~StartFn();
        },
        [](void *ctx, void *data, WorkerInit &&init) {
            new (ctx) ContextT((DataT *)data, std::forward<WorkerInit>(init));
        },
        (void *)ctx_data,
        &state_mgr,
        desired_num_workers,
        num_io,
        sizeof(ContextT),
        std::alignment_of_v<ContextT>,
        pin_workers,
    };
}

template <typename StartFn>
JobManager::JobManager(const Init<StartFn> &init)
    : JobManager(init.desiredNumWorkers, init.numIOWorkers,
                 init.stateMgr, init.numCtxBytes, init.ctxAlignment,
                 init.ctxInitCB, init.ctxData, init.startCB,
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
