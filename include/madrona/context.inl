/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace madrona {

template <typename ArchetypeT>
Entity Context::makeEntity()
{
    return state_mgr_->makeEntityNow<ArchetypeT>(
        MADRONA_MW_COND(cur_world_id_,) *state_cache_);
}

Entity Context::makeEntity(uint32_t archetype_id)
{
    return state_mgr_->makeEntityNow(
        MADRONA_MW_COND(cur_world_id_,) *state_cache_, archetype_id);
}

template <typename ArchetypeT>
Loc Context::makeTemporary()
{
    return state_mgr_->makeTemporary<ArchetypeT>(
        MADRONA_MW_COND(cur_world_id_));
}

Loc Context::makeTemporary(uint32_t archetype_id)
{
    return state_mgr_->makeTemporary(MADRONA_MW_COND(cur_world_id_,)
                                     archetype_id);
}

void Context::destroyEntity(Entity e)
{
    state_mgr_->destroyEntityNow(MADRONA_MW_COND(cur_world_id_,)
                                 *state_cache_, e);
}

Loc Context::loc(Entity e) const
{
    return state_mgr_->getLoc(e);
}

template <typename ComponentT>
ComponentT & Context::get(Entity e)
{
    return state_mgr_->getUnsafe<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) e.id);
}

template <typename ComponentT>
ComponentT & Context::get(Loc l)
{
    return state_mgr_->getUnsafe<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) l);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getSafe(Entity e)
{
    return state_mgr_->get<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getCheck(Entity e)
{
    return state_mgr_->get<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getCheck(Loc l)
{
    return state_mgr_->get<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) l);
}

template <typename ComponentT>
ComponentT & Context::getDirect(int32_t column_idx, Loc loc)
{
    return state_mgr_->getDirect<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) column_idx, loc);
}

template <typename SingletonT>
SingletonT & Context::singleton()
{
    return state_mgr_->getSingleton<SingletonT>(MADRONA_MW_COND(cur_world_id_));
}

void * Context::tmpAlloc(uint64_t num_bytes)
{
    return state_mgr_->tmpAlloc(MADRONA_MW_COND(cur_world_id_,) num_bytes);
}

template <typename... ComponentTs>
Query<ComponentTs...> Context::query()
{
    return state_mgr_->query<ComponentTs...>();
}

template <typename Fn, typename... ComponentTs>
inline void Context::iterateQuery(const Query<ComponentTs...> &query, Fn &&fn)
{
    state_mgr_->iterateQuery(MADRONA_MW_COND(cur_world_id_, ) query,
        std::forward<Fn>(fn));
}

#ifdef MADRONA_USE_JOB_SYSTEM
JobID Context::currentJobID() const
{
    return cur_job_id_;
}
#endif

#ifdef MADRONA_MW_MODE
WorldID Context::worldID() const
{
    return WorldID { (int32_t)cur_world_id_ };
}
#endif

#if 0

class Context {
    AllocContext mem;

    template <typename ArchetypeT>
    inline ArchetypeRef<ArchetypeT> archetype();

    template <typename ArchetypeT>
    inline void clearArchetype();

    template <typename ArchetypeT>
    inline void clearTemporaries();

    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename... ComponentTs, typename Fn>
    inline void forEach(const Query<ComponentTs...> &query, Fn &&fn);

    template <typename... ComponentTs>
    inline uint32_t numMatches(const Query<ComponentTs...> &query);

    // Jobs
    template <typename Fn, typename... DepTs>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        DepTs && ... dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
                         bool is_child = true,
                         DepTs && ... dependencies);

    // FIXME: currently this function requires that the query reference
    // is valid at least until the returned job is completed.
    template <typename... ComponentTs, typename Fn, typename... DepTs>
    inline JobID parallelFor(const Query<ComponentTs...> &query, Fn &&fn,
                             bool is_child = true,
                             DepTs && ... dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID ioRead(const char *path, Fn &&fn, bool is_child = true,
                        DepTs && ... dependencies);

#ifdef MADRONA_USE_JOB_SYSTEM
    inline JobID currentJobID() const;
#endif

    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitImpl(Fn &&fn, bool is_child, DepTs && ... dependencies);

    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                             DepTs && ... dependencies);

    template <typename ContextT, typename... ComponentTs, typename Fn,
              typename... DepTs>
    inline JobID parallelForImpl(const Query<ComponentTs...> &query, Fn &&fn,
                                 bool is_child, DepTs && ... dependencies);

private:
    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitNImpl(Fn &&fn, uint32_t num_invocations, JobID parent_id,
                             DepTs && ... dependencies);

#ifdef MADRONA_USE_JOB_SYSTEM
    JobManager * const job_mgr_;
    StateManager * const state_mgr_;
    StateCache * const state_cache_;
    IOManager * const io_mgr_;
    const int worker_idx_;
    JobID cur_job_id_;
#endif

friend class JobManager;
};

template <typename ArchetypeT>
ArchetypeRef<ArchetypeT> Context::archetype()
{
    return state_mgr_->archetype<ArchetypeT>(
        MADRONA_MW_COND(cur_world_id_));
}

template <typename ArchetypeT>
void Context::clearArchetype()
{
    state_mgr_->clear<ArchetypeT>(MADRONA_MW_COND(cur_world_id_,)
                                  *state_cache_, false);
}

template <typename ArchetypeT>
void Context::clearTemporaries()
{
}

template <typename... ComponentTs>
Query<ComponentTs...> Context::query()
{
    return state_mgr_->query<ComponentTs...>();
}

template <typename... ComponentTs, typename Fn>
void Context::forEach(const Query<ComponentTs...> &query, Fn &&fn)
{
    state_mgr_->iterateQuery(MADRONA_MW_COND(cur_world_id_,) query,
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
    assert(num_invocations > 0);
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

    // FIXME: add isRunnable check in addition to no dependencies
    if constexpr (sizeof...(dependencies) == 0) {
        JobID parent_id = is_child ? cur_job_id_ : JobID::none();

        // Additional optimization: skip this proxy ID when only 1 archetype
        // is present (in fact for > 1 archetype might make sense to just
        // use the else codepath).
        JobID proxy_id = job_mgr_->reserveProxyJobID(worker_idx_, parent_id);

        state_mgr_->iterateArchetypes(MADRONA_MW_COND(cur_world_id_,)
                query, [this, &fn, proxy_id](int num_rows, auto ...ptrs) {
            if (num_rows == 0) {
                return;
            }

            // Clang complains this is unused without this->
            this->submitNImpl<ContextT>(
                    [fn = Fn(fn), ptrs...](ContextT &ctx, uint32_t idx) {
                fn(ctx, ptrs[idx]...);
            }, num_rows, proxy_id);
        });

        // Note that even though we "relinquish" the id here, it is still safe
        // to return the ID, since the generation stored in the ID will simply
        // be invalid if the entire parallelFor job finishes, just like a normal
        // job id.
        job_mgr_->relinquishProxyJobID(worker_idx_, proxy_id);

        return proxy_id;
    } else {
        return submitImpl<ContextT>([fn = std::forward<Fn>(fn), &query] (
                ContextT &ctx) {
            ctx.state_mgr_->iterateArchetypes(
                    MADRONA_MW_COND(ctx.cur_world_id_,) query,
                    [&ctx, &fn](int num_rows, auto ...ptrs) {
                if (num_rows == 0) {
                    return;
                }

                // FIXME reconsider copying ptrs into the closure here
                // FIXME currently copies the user function's closure
                // Could allow making a fake jobs with data but not a function
                // by extending reserveProxyJobID - that job could be dependent
                // on the parallel for job and hold the user function closure.
                // If we allowed runtime determined # of dependencies, the
                // fast path (no dependencies above) could return the dependent
                // data-only job rather than using the fake ID as a parent
                ctx.template submitNImpl<ContextT>(
                        [fn = Fn(fn), ptrs...](ContextT &ctx, uint32_t idx) {
                    fn(ctx, ptrs[idx]...);
                }, num_rows, true);
            });
        }, is_child, dependencies...);
    }
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
#endif

}
