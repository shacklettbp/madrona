#pragma once

#include <madrona/job.hpp>
#include <madrona/ecs.hpp>
#include <madrona/state.hpp>
#include <madrona/io.hpp>

namespace madrona {

class Context {
public:
    Context(WorkerInit &&init);

    AllocContext mem;
    inline StateManager & state();

    template <typename Fn, typename... Deps>
    inline JobID submit(Fn &&fn, bool is_child = true,
                          Deps && ... dependencies);

    template <typename Fn, typename... Deps>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
                         bool is_child = true,
                         Deps && ... dependencies);

    template <typename... ColTypes, typename Fn, typename... Deps>
    inline JobID forAll(const Query<ColTypes...> &query, Fn &&fn,
                        bool is_child = true,
                        Deps && ... dependencies);

    template <typename Fn, typename... Deps>
    inline JobID ioRead(const char *path, Fn &&fn, bool is_child = true,
                        Deps && .. dependencies);

#if 0
    template <typename... QueryArgs>
    inline Query<QueryArgs...> query();

    template <typename T>
    TableRef table();
#endif

protected:
    template <typename ContextT, typename Fn, typename... Deps>
    inline JobID submitImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                            Deps && ... dependencies);

    template <typename ContextT, typename... ColTypes, typename Fn,
              typename... Deps>
    inline JobID forAllImpl(const Query<ColTypes...> &query, Fn &&fn,
                            bool is_child, Deps && ... dependencies);

private:
    template <typename ContextT, typename Fn>
    Job makeJob(Fn &&fn);

    JobManager * const job_mgr_;
    StateManager * const state_mgr_;
    IOManager * const io_mgr_;
    const int worker_idx_;
};

}

#include "context.inl"
