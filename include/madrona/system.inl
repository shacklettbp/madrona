#pragma once

namespace madrona {

template <typename SystemT, typename... ComponentTs>
ParallelForSystem<SystemT, ComponentTs...>::ParallelForSystem(const SystemInit &sys_init)
    : SystemBase(&ParallelForSystem<SystemT, ComponentTs...>::entry)
{}

template <typename... ComponentTs>
void ParallelForSystem<ComponentTs...>::entry(Context &ctx,
                                              uint32_t invocation_idx)
{

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


}
