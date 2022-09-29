#pragma once

#include <madrona/job.hpp>
#include <madrona/ecs.hpp>
#include <madrona/state.hpp>
#include <madrona/io.hpp>

namespace madrona {

class Context {
public:
    Context(WorkerInit &&init);
    Context(const Context &) = delete;

    AllocContext mem;

    // Registration
    template <typename ComponentT>
    void registerComponent();

    template <typename ArchetypeT>
    void registerArchetype();

    // State
    template <typename ArchetypeT>
    inline ArchetypeRef<ArchetypeT> archetype();

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntity(Transaction &txn, Args && ...args);

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntityNow(Args && ...args);

    inline void destroyEntity(Transaction &txn, Entity e);

    inline void destroyEntityNow(Entity e);

    template <typename ArchetypeT>
    inline void clearArchetype();

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

    template <typename... ComponentTs, typename Fn, typename... DepTs>
    inline JobID parallelFor(const Query<ComponentTs...> &query, Fn &&fn,
                             bool is_child = true,
                             DepTs && ... dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID ioRead(const char *path, Fn &&fn, bool is_child = true,
                        DepTs && ... dependencies);

    inline JobID currentJobID() const;

#ifdef MADRONA_MW_MODE
    inline uint32_t worldID() const;
#endif

protected:
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

    JobManager * const job_mgr_;
    StateManager * const state_mgr_;
    StateCache * const state_cache_;
    IOManager * const io_mgr_;
    const int worker_idx_;
    JobID cur_job_id_;
#ifdef MADRONA_MW_MODE
    uint32_t cur_world_id_;
#endif

friend class JobManager;
};

}

#include "context.inl"
