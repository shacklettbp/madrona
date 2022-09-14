#pragma once

namespace madrona {

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
    return forAllImpl<Context>(query, std::forward<Fn>(fn), is_child,
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

    return JobID::none();
}

StateManager & Context::state() { return *state_mgr_; }

// FIXME: implement is_child, dependencies, num_invocations
template <typename ContextT, typename Fn, typename... Deps>
JobID Context::submitImpl(Fn &&fn, bool is_child,
                          Deps &&... dependencies)
{
    auto wrapper = [fn = std::forward<Fn>(fn)](ContextT &ctx, uint32_t) {
        fn(ctx);
    };

    return submitNImpl<ContextT>(std::forward<decltype(wrapper)>(wrapper), 1,
        is_child, std::forward<Deps>(dependencies)...);
}

template <typename ContextT, typename Fn, typename... Deps>
JobID Context::submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                           Deps && ...dependencies)
{
    JobID parent_id = is_child ? cur_job_id_ : JobID::none();

    return submitNImpl<ContextT>(std::forward<Fn>(fn), num_invocations,
        parent_id, std::forward<Deps>(dependencies)...);
}

template <typename ContextT, typename... ComponentTs, typename Fn,
          typename... Deps>
JobID Context::forAllImpl(Query<ComponentTs...> query, Fn &&fn,
                          bool is_child, Deps && ... dependencies)
{
    if (query.numMatchingArchetypes() == 0) {
        return JobID::none();
    }

    JobID parent_id = is_child ? cur_job_id_ : JobID::none();

    JobID proxy_id = job_mgr_->reserveProxyJobID(parent_id);

    state_mgr_->iterateArchetypes(query,
            [this, proxy_id, fn = std::forward<Fn>(fn), dependencies ...](
            int num_rows, auto ...ptrs) {
        if (num_rows == 0) {
            return;
        }

        // FIXME, is there a better solution here?
        Fn fn_copy(fn);

        auto wrapper = [fn = std::forward<Fn>(fn_copy), ptrs ...](
                ContextT &ctx, uint32_t idx) {
            fn(ctx, ptrs[idx]...);
        };

        // For some reason clang warns that 'this' isn't used without the
        // explicit this->
        this->submitNImpl<ContextT>(std::move(wrapper), num_rows, proxy_id,
                                    dependencies ...);
    });

    // Note that even though we "relinquish" the id here, it is still safe
    // to return the ID, since the generation stored in the ID will simply
    // be invalid if the entire forall job finishes, just like a normal job id.
    job_mgr_->relinquishProxyJobID(proxy_id);

    return proxy_id;
}

template <typename ContextT, typename Fn, typename... Deps>
JobID Context::submitNImpl(Fn &&fn, uint32_t num_invocations, JobID parent_id,
                           Deps && ...dependencies)
{
    return job_mgr_->queueJob<ContextT>(worker_idx_, std::forward<Fn>(fn),
                                        num_invocations, parent_id,
#ifdef MADRONA_MW_MODE
                                        cur_world_id_,
#endif
                                        JobPriority::Normal,
                                        std::forward<Deps>(dependencies)...);
}

JobID Context::currentJobID() const
{
    return cur_job_id_;
}

#ifdef MADRONA_MW_MODE
uint32_t Context::worldID() const
{
    return cur_world_id_;
}
#endif

}
