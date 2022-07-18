#pragma once

namespace madrona {

StateManager & Context::state() { return *state_mgr_; }

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
    Job job = makeJob<ContextT>(std::forward<Fn>(fn));
    (void)is_child;

    ( (void)dependencies, ... );
    (void)num_invocations;

    return job_mgr_->queueJob(worker_idx_, job, nullptr, 0,
                              JobPriority::Normal);
}

template <typename ContextT, typename... ColTypes, typename Fn,
          typename... Deps>
JobID Context::forAllImpl(const Query<ColTypes...> &query, Fn &&fn,
                          bool is_child, Deps && ... dependencies)
{
    const uint32_t num_entities = query.size();

    auto query_loop = [fn=std::move(fn), query, num_entities](
        ContextT &ctx, Entity entity) {
        fn(ctx, query.template get<ColTypes>(entity)...);
    };

    return submitImpl<ContextT>(std::move(query_loop), num_entities, is_child,
                                std::forward<Deps>(dependencies)...);
}

template <typename ContextT, typename Fn>
Job Context::makeJob(Fn &&fn)
{
    Job job;

    if constexpr (std::is_empty_v<Fn>) {
        job.func_ = [](Context &ctx_base, void *) {
            ContextT &ctx = static_cast<ContextT &>(ctx_base);
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

        job.func_ = [](Context &ctx_base, void *data) {
            ContextT &ctx = static_cast<ContextT &>(ctx_base);

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

template <typename ContextT>
CustomContext<ContextT>::CustomContext(WorkerInit &&worker_init)
    : Context(std::forward<WorkerInit>(worker_init))
{}

template <typename ContextT>
template <typename Fn, typename... Deps>
JobID CustomContext<ContextT>::submit(Fn &&fn, bool is_child,
                                      Deps && ... dependencies)
{
    return submitImpl<ContextT>(std::forward<Fn>(fn), 1, is_child,
                                std::forward<Deps>(dependencies)...);
}

template <typename ContextT>
template <typename Fn, typename... Deps>
JobID CustomContext<ContextT>::submitN(Fn &&fn, uint32_t num_invocations,
                                       bool is_child, Deps && ... dependencies)
{
    return submitImpl<ContextT>(std::forward<Fn>(fn), num_invocations,
                                is_child, std::forward<Deps>(dependencies)...);
}

template <typename ContextT>
template <typename... ColTypes, typename Fn, typename... Deps>
JobID CustomContext<ContextT>::forAll(const Query<ColTypes...> &query, Fn &&fn,
                                      bool is_child, Deps && ... dependencies)
{
    return forallImpl<ContextT>(query, std::forward<Fn>(fn), is_child,
                                std::forward<Deps>(dependencies)...);
}

}
