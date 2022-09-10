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

    template <typename ArchetypeT>
    inline ArchetypeRef<ArchetypeT> archetype();

    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename Fn, typename... DepTs>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        DepTs && ... dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
                         bool is_child = true,
                         DepTs && ... dependencies);

    template <typename... ComponentTs, typename Fn, typename... DepTs>
    inline JobID forAll(Query<ComponentTs...> query, Fn &&fn,
                        bool is_child = true,
                        DepTs && ... dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID ioRead(const char *path, Fn &&fn, bool is_child = true,
                        DepTs && ... dependencies);

    inline JobID currentJobID() const;

    inline StateManager & state();

protected:
    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitImpl(Fn &&fn, bool is_child, DepTs && ... dependencies);

    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                            DepTs && ... dependencies);

    template <typename ContextT, typename... ComponentTs, typename Fn,
              typename... DepTs>
    inline JobID forAllImpl(Query<ComponentTs...> query, Fn &&fn,
                            bool is_child, DepTs && ... dependencies);

private:
    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitNImpl(Fn &&fn, uint32_t num_invocations, JobID parent_id,
                            DepTs && ... dependencies);

    JobManager * const job_mgr_;
    StateManager * const state_mgr_;
    IOManager * const io_mgr_;
    const int worker_idx_;
    JobID cur_job_id_;

friend class JobManager;
};

}

#include "context.inl"
