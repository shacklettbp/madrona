/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace madrona {

template <typename ContextT, typename DataT>
CustomContext<ContextT, DataT>::CustomContext(DataT *world_data,
                                              const WorkerInit &worker_init)
    : Context(world_data, worker_init)
{}

#if 0

template <typename ContextT, typename DataT>
class CustomContext : public Context {
public:
    template <typename Fn, typename... Deps>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        Deps && ... dependencies);

    template <typename Fn, typename... Deps>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
                         bool is_child = true,
                         Deps && ... dependencies);

    template <typename... ComponentTs, typename Fn, typename... Deps>
    inline JobID parallelFor(const Query<ComponentTs...> &query, Fn &&fn,
                             bool is_child = true,
                             Deps && ... dependencies);
}

template <typename ContextT, typename DataT>
template <typename Fn, typename... Deps>
JobID CustomContext<ContextT, DataT>::submit(Fn &&fn, bool is_child,
                                             Deps && ... dependencies)
{
    return submitImpl<ContextT>(std::forward<Fn>(fn), is_child,
                                std::forward<Deps>(dependencies)...);
}

template <typename ContextT, typename DataT>
template <typename Fn, typename... Deps>
JobID CustomContext<ContextT, DataT>::submitN(Fn &&fn,
    uint32_t num_invocations, bool is_child, Deps && ... dependencies)
{
    return submitNImpl<ContextT>(
        std::forward<Fn>(fn), num_invocations, is_child,
        std::forward<Deps>(dependencies)...);
}

template <typename ContextT, typename DataT>
template <typename... ComponentTs, typename Fn, typename... Deps>
JobID CustomContext<ContextT, DataT>::parallelFor(
    const Query<ComponentTs...> &query, Fn &&fn, bool is_child,
    Deps && ... dependencies)
{
    return parallelForImpl<ContextT>(query, std::forward<Fn>(fn), is_child,
                                      std::forward<Deps>(dependencies)...);
}

#endif

}
