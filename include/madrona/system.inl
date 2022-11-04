#pragma once

namespace madrona {

template <typename SystemT>
CustomSystem<SystemT>::CustomSystem()
    : SystemBase(&CustomSystem<SystemT>::entry)
{}

template <typename SystemT>
void CustomSystem<SystemT>::entry(SystemBase *sys_base, void *data,
                                  uint32_t invocation_offset)
{
    SystemT *sys = static_cast<SystemT *>(sys_base);
    sys->run(data, invocation_offset);
}

#if 0
template <typename SystemT, typename... ComponentTs>
ParallelForSystem<SystemT, ComponentTs...>::ParallelForSystem()
    : SystemBase((SystemBase::EntryFn)
        &ParallelForSystem<SystemT, ComponentTs...>::entry)
{}

template <typename Fn, typename... ComponentTs>
LambdaParallelFor<Fn, ComponentTs...>::LambdaParallelForSystem(Fn &&fn)
    : ParallelForSystem<LambdaParallelForSystem<Fn, ComponentTs...>, ComponentTs...>(),
      fn(std::forward<Fn>(fn))
{}

template <typename Fn, typename... ComponentTs>
void LambdaParallelForSystem<Fn, ComponentTs...>::run(ContextT &ctx,
                                                      ComponentTs... &&components)
{
    fn(ctx, std::forward<ComponentTs>(components)...);
}
#endif

}
