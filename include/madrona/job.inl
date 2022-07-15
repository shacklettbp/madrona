#pragma once

namespace madrona {

template <typename Fn>
JobManager::JobManager(int desired_num_workers, int num_io,
                       StateManager &state_mgr, void *world_data,
                       Fn &&fn, bool pin_workers)
    : JobManager(desired_num_workers, num_io,
        [](Context &ctx, void *data) {
            auto fn_ptr = (Fn *)data;
            (*fn_ptr)(ctx);
            fn_ptr->~Fn();
        },
        &fn,
        state_mgr,
        world_data,
        pin_workers)
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
