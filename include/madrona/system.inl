#pragma once

namespace madrona {

#if 0
template <typename SystemT>
CustomSystem<SystemT>::CustomSystem()
    : SystemBase(&CustomSystem<SystemT>::entry)
{}

template <typename SystemT>
void CustomSystem<SystemT>::entry(SystemBase *sys_base,
                                  void *data,
                                  uint32_t invocation_offset)
{
    SystemT *sys = static_cast<SystemT *>(sys_base);
    sys->run(data, invocation_offset);
}

template <typename SystemT, typename... ComponentTs>
ParallelForSystem<SystemT, ComponentTs...>::ParallelForSystem(Context &ctx)
    : SystemBase((SystemBase::EntryFn)
        &ParallelForSystem<SystemT, ComponentTs...>::entry),
      query_(ctx.query<ComponentTs...>())
{}

template <typename SystemT, typename... ComponentTs>
void ParallelForSystem<SystemT, ComponentTs...>::entry(SystemBase *sys_base,
        void *data, uint32_t invocation_offset)
{
    (void)sys_base;
    (void)data;
    (void)invocation_offset;
}

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
