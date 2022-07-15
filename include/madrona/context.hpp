#pragma once

#include <madrona/job.hpp>
#include <madrona/ecs.hpp>
#include <madrona/state.hpp>

namespace madrona {

class Context {
public:
    Context(JobManager &job_mgr, StateManager &state_mgr, void *world_data,
            int worker_idx);

    AllocContext mem;
    inline StateManager & state();

    template <typename T>
    inline T & world();

    template <typename Fn, typename... Deps>
    inline JobID queueJob(Fn &&fn, bool is_child = true,
                          Deps && ... dependencies);

    template <typename Fn, typename... Deps>
    inline JobID queueMultiJob(Fn &&fn, uint32_t num_invocations,
                               bool is_child = true,
                               Deps && ... dependencies);

#if 0
    template <typename Fn, typename... QueryArgs, typename... Deps>
    inline JobID queueForAll(Fn &&fn, const Query<QueryArgs...> &q,
        bool is_child = true, Deps && ...dependencies);

    template <typename... QueryArgs>
    inline Query<QueryArgs...> query();

    template <typename T>
    TableRef table();
#endif

private:
    template <typename Fn>
    Job makeJob(Fn &&fn);

    JobManager * const job_mgr_;
    StateManager * const state_mgr_;
    void * const world_data_;
    const int worker_idx_;
};

}

#include "context.inl"
