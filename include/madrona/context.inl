#pragma once

namespace madrona {

inline StateManager & Context::state() { return *state_mgr_; }
 
// FIXME: implement is_child and dependencies
template <typename Fn, typename... Deps>
inline JobID Context::queueJob(Fn &&fn, bool is_child,
                               Deps &&... dependencies)
{
    Job job = makeJob(std::forward<Fn>(fn));
    (void)is_child;

    ( (void)dependencies, ... );

    return job_mgr_->queueJob(worker_idx_, job, nullptr, 0,
                              JobPriority::Normal);
}

template <typename Fn>
Job Context::makeJob(Fn &&fn)
{
    Job job;

    if constexpr (std::is_empty_v<Fn>) {
        job.func_ = [](Context &ctx, void *) {
            Fn()(ctx);
        };
        job.data_ = nullptr;
    } else {
        // Make job_size bigger to fit metadata
        static constexpr size_t job_size = sizeof(Fn);

        static constexpr size_t job_alignment = std::alignment_of_v<Fn>;
        static_assert(job_size <= JobManager::Alloc::maxJobSize,
                      "Job lambda capture is too large");
        static_assert(job_alignment <= JobManager::Alloc::maxJobAlignment,
            "Job lambda capture has too large an alignment requirement");
        static_assert(utils::isPower2(job_alignment));

        void *store = job_mgr_->allocJob(worker_idx_, job_size, job_alignment);

        new (store) Fn(std::forward<Fn>(fn));

        job.func_ = [](Context &ctx, void *data) {
            auto fn_ptr = (Fn *)data;
            (*fn_ptr)(ctx);
            fn_ptr->~Fn();

            // Important note: jobs may be freed by different threads
            ctx.job_mgr_->deallocJob(ctx.worker_idx_, fn_ptr, job_size);
        };

        job.data_ = store;
    }

    return job;
}


}
