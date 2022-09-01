#pragma once

#include <madrona/context.hpp>

namespace madrona {

template <typename ContextT>
class CustomContext : public Context {
public:
    inline CustomContext(WorkerInit &&worker_init);

    template <typename Fn, typename... Deps>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        Deps && ... dependencies);

    template <typename Fn, typename... Deps>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
                         bool is_child = true,
                         Deps && ... dependencies);

    template <typename... ComponentTs, typename Fn, typename... Deps>
    inline JobID forAll(Query<ComponentTs...> query, Fn &&fn,
                        bool is_child = true,
                        Deps && ... dependencies);
};

}

#include "custom_context.inl"
