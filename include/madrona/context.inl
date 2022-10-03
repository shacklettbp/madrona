#pragma once

namespace madrona {

template <typename ComponentT>
void Context::registerComponent()
{
    state_mgr_->registerComponent<ComponentT>();
}

template <typename ArchetypeT>
void Context::registerArchetype()
{
    state_mgr_->registerArchetype<ArchetypeT>();
}

template <typename ArchetypeT>
ArchetypeRef<ArchetypeT> Context::archetype()
{
    return state_mgr_->archetype<ArchetypeT>(
        MADRONA_MW_COND(cur_world_id_));
}


template <typename ArchetypeT, typename... Args>
Entity Context::makeEntity(Transaction &txn, Args && ...args)
{
    return state_mgr_->makeEntity<ArchetypeT>(
        MADRONA_MW_COND(cur_world_id_,) txn, *state_cache_,
        std::forward<Args>(args)...);
}

template <typename ArchetypeT, typename... Args>
Entity Context::makeEntityNow(Args && ...args)
{
    return state_mgr_->makeEntityNow<ArchetypeT>(
        MADRONA_MW_COND(cur_world_id_,) *state_cache_,
        std::forward<Args>(args)...);
}

void Context::destroyEntity(Transaction &txn, Entity e)
{
    state_mgr_->destroyEntity(MADRONA_MW_COND(cur_world_id_,)
                              txn, *state_cache_, e);
}

void Context::destroyEntityNow(Entity e)
{
    state_mgr_->destroyEntityNow(MADRONA_MW_COND(cur_world_id_,)
                                 *state_cache_, e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::get(Entity e)
{
    return state_mgr_->get<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) e);
}

template <typename ArchetypeT>
void Context::clearArchetype()
{
    state_mgr_->clear<ArchetypeT>(MADRONA_MW_COND(cur_world_id_,)
                                  *state_cache_);
}

template <typename... ComponentTs>
Query<ComponentTs...> Context::query()
{
    return state_mgr_->query<ComponentTs...>();
}

template <typename... ComponentTs, typename Fn>
void Context::forEach(const Query<ComponentTs...> &query, Fn &&fn)
{
    state_mgr_->iterateEntities(MADRONA_MW_COND(cur_world_id_,) query,
                                std::forward<Fn>(fn));
}

template <typename... ComponentTs>
uint32_t Context::numMatches(const Query<ComponentTs...> &query)
{
    uint32_t num_entities = 0;
    state_mgr_->iterateArchetypes(MADRONA_MW_COND(cur_world_id_,) query,
            [&](int num_rows, auto ...) {
        num_entities += num_rows;
    });

    return num_entities;
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
    assert(num_invocations > 0);
    return submitNImpl<Context>(std::forward<Fn>(fn), num_invocations,
        is_child, std::forward<Deps>(dependencies)...);
}

template <typename... ComponentTs, typename Fn, typename... Deps>
JobID Context::parallelFor(const Query<ComponentTs...> &query, Fn &&fn,
                            bool is_child, Deps && ... dependencies)
{
    return parallelForImpl<Context>(query, std::forward<Fn>(fn), is_child,
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

// FIXME: implement is_child, dependencies, num_invocations
template <typename ContextT, typename Fn, typename... Deps>
JobID Context::submitImpl(Fn &&fn, bool is_child,
                          Deps &&... dependencies)
{
    JobID parent_id = is_child ? cur_job_id_ : JobID::none();

    return job_mgr_->queueJob<ContextT, true>(worker_idx_,
        std::forward<Fn>(fn), 0, parent_id,
        MADRONA_MW_COND(cur_world_id_, ) JobPriority::Normal,
        std::forward<Deps>(dependencies)...);
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
JobID Context::parallelForImpl(const Query<ComponentTs...> &query, Fn &&fn,
                               bool is_child, Deps && ... dependencies)
{
    if (query.numMatchingArchetypes() == 0) {
        return JobID::none();
    }

    JobID parent_id = is_child ? cur_job_id_ : JobID::none();

    JobID proxy_id = job_mgr_->reserveProxyJobID(worker_idx_, parent_id);
    JobID debug_parent = job_mgr_->getParent(proxy_id);
    assert(debug_parent.id == parent_id.id &&
           debug_parent.gen == parent_id.gen);

    state_mgr_->iterateArchetypes(MADRONA_MW_COND(cur_world_id_,) query,
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

    debug_parent = job_mgr_->getParent(proxy_id);
    assert(debug_parent.id == parent_id.id &&
           debug_parent.gen == parent_id.gen);

    // Note that even though we "relinquish" the id here, it is still safe
    // to return the ID, since the generation stored in the ID will simply
    // be invalid if the entire parallelFor job finishes, just like a normal
    // job id.
    job_mgr_->relinquishProxyJobID(worker_idx_, proxy_id);

    return proxy_id;
}

template <typename ContextT, typename Fn, typename... Deps>
JobID Context::submitNImpl(Fn &&fn, uint32_t num_invocations, JobID parent_id,
                           Deps && ...dependencies)
{
    return job_mgr_->queueJob<ContextT, false>(worker_idx_,
        std::forward<Fn>(fn), num_invocations, parent_id,
        MADRONA_MW_COND(cur_world_id_, ) JobPriority::Normal,
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
