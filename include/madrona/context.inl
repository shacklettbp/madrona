#pragma once

namespace madrona {

StateManager & Context::state() { return *state_mgr_; }

template <typename ArchetypeT>
ArchetypeRef<ArchetypeT> Context::archetype()
{
    return state_mgr_->archetype<ArchetypeT>();
}

template <typename... ComponentTs>
Query<ComponentTs...> Context::query()
{
    return state_mgr_->query<ComponentTs...>();
}

template <typename Fn, typename... Deps>
JobID Context::submit(Fn &&fn, bool is_child, Deps && ... dependencies)
{
    return submitImpl<Context>(std::forward<Fn>(fn), is_child,
                               std::forward<Deps>(dependencies)...);
}

template <typename Fn, typename... Deps>
JobID Context::submitN(Fn &&fn, uint32_t num_invocations,
                       bool is_child, Deps && ... dependencies)
{
    return submitNImpl<Context>(std::forward<Fn>(fn), num_invocations,
        is_child, std::forward<Deps>(dependencies)...);
}

template <typename... ComponentTs, typename Fn, typename... Deps>
JobID Context::forAll(Query<ComponentTs...> query, Fn &&fn,
                      bool is_child, Deps && ... dependencies)
{
    return forallImpl<Context>(query, std::forward<Fn>(fn), is_child,
                               std::forward<Deps>(dependencies)...);
}

template <typename Fn, typename... Deps>
inline JobID Context::ioRead(const char *path, Fn &&fn,
                             bool is_child, Deps && ... dependencies)
{
    IOPromise promise = io_mgr_->makePromise();
    Job job = makeJob([promise, fn=std::move(fn), io_mgr=io_mgr_](
            Context &ctx) {
        fn(ctx, io_mgr->getBuffer(promise));
    });

    io_mgr_->load(promise, path, job);

    (void)is_child;
    ( (void)dependencies, ... );
}

// FIXME: implement is_child, dependencies, num_invocations
template <typename ContextT, typename Fn, typename... Deps>
JobID Context::submitImpl(Fn &&fn, bool is_child,
                          Deps &&... dependencies)
{
    auto wrapper = [fn = std::forward<Fn>(fn)](ContextT &ctx, uint32_t) {
        fn(ctx);
    };

    submitNImpl(std::forward<Fn>(wrapper), 1, is_child,
                std::forward<Deps>(dependencies)...);
}

// FIXME: implement is_child, dependencies, num_invocations
template <typename ContextT, typename Fn, typename... Deps>
JobID Context::submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                           Deps &&... dependencies)
{
    Job job = makeJob<ContextT>(std::forward<Fn>(fn));
    (void)is_child;

    ( (void)dependencies, ... );
    (void)num_invocations;

    return job_mgr_->queueJob(worker_idx_, job, nullptr, 0,
                              JobPriority::Normal);
}

template <typename ContextT, typename... ComponentTs, typename Fn,
          typename... Deps>
JobID Context::forAllImpl(Query<ComponentTs...> query, Fn &&fn,
                          bool is_child, Deps && ... dependencies)
{
    state_mgr_->iterateArchetypes(query, [fn = std::forward<Fn>(fn)](
            int num_rows, auto ...ptrs) {
        if (num_rows == 0) {
            return;
        }

        auto wrapper = [fn = std::forward<Fn>(fn), ptrs ...](
                Context &ctx, uint32_t idx) {
            fn(ctx, ptrs[idx]...);
        };

        submitImpl<ContextT>(std::move(wrapper), num_rows, is_child,
                             std::forward<Deps>(dependencies)...);
    });
}

template <typename ContextT, typename Fn>
Job Context::makeJob(Fn &&fn, uint32_t num_invocations)
{
    Job job;
    job.invocation_offset_ = 0;
    job.num_invocations_ = num_invocations;

    if constexpr (std::is_empty_v<Fn>) {
        job.func_ = [](Context &ctx_base, void *, uint32_t invocation_idx) {
            ContextT &ctx = static_cast<ContextT &>(ctx_base);
            Fn()(ctx, invocation_idx);
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

        job.func_ = [](Context &ctx_base, void *data,
                       uint32_t invocation_idx) {
            ContextT &ctx = static_cast<ContextT &>(ctx_base);

            auto fn_ptr = (Fn *)data;
            (*fn_ptr)(ctx, invocation_idx);
            fn_ptr->~Fn();

            // Important note: jobs may be freed by different threads
            ctx.job_mgr_->deallocJob(ctx.worker_idx_, fn_ptr, job_size);
        };

        job.data_ = store;
    }

    return job;
}

}
