#pragma once

namespace madrona {

template <typename ContextT>
CustomContext<ContextT>::CustomContext(WorkerInit &&worker_init)
    : Context(std::forward<WorkerInit>(worker_init))
{}

template <typename ContextT>
template <typename Fn, typename... Deps>
JobID CustomContext<ContextT>::submit(Fn &&fn, bool is_child,
                                      Deps && ... dependencies)
{
    return submitImpl<ContextT>(std::forward<Fn>(fn), is_child,
                                std::forward<Deps>(dependencies)...);
}

template <typename ContextT>
template <typename Fn, typename... Deps>
JobID CustomContext<ContextT>::submitN(Fn &&fn, uint32_t num_invocations,
                                       bool is_child, Deps && ... dependencies)
{
    return submitNImpl<ContextT>(
        std::forward<Fn>(fn), num_invocations, is_child,
        std::forward<Deps>(dependencies)...);
}

template <typename ContextT>
template <typename... ComponentTs, typename Fn, typename... Deps>
JobID CustomContext<ContextT>::forAll(const Query<ComponentTs...> &query,
                                      Fn &&fn, bool is_child,
                                      Deps && ... dependencies)
{
    return forAllImpl<ContextT>(query, std::forward<Fn>(fn), is_child,
                                std::forward<Deps>(dependencies)...);
}


}
