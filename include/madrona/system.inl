#pragma once

namespace madrona {

template <typename SystemT, typename... ComponentTs>
ParallelForSystem<SystemT, ComponentTs...>::ParallelForSystem()
    : SystemBase(&ParallelForSystem<SystemT, ComponentTs...>::entry)
{}

template <typename... ComponentTs>
void ParallelForSystem<ComponentTs...>::entry(Context &ctx,
                                              uint32_t invocation_idx)
{
}

template <typename Fn, typename... ComponentTs>
LambdaParallelFor<Fn, ComponentTs...>::LambdaParallelFor(Fn &&fn)
    : ParallelForSystem<LambdaParallelFor<Fn, ComponentTs...>, ComponentTs...>(),
      fn(std::forward<Fn>(fn))
{}

template <typename Fn, typename... ComponentTs>
void LambdaParallelFor<Fn, ComponentTs...>::run(ContextT &ctx,
                                                ComponentTs... &&components)
{
    fn(ctx, std::forward<ComponentTs>(components)...);
}


}
