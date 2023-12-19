#pragma once

#include <madrona/fwd.hpp>
#include <madrona/utils.hpp>
#include <madrona/query.hpp>
#include <madrona/context.hpp>

#include <cstdint>

namespace madrona {

class SystemBase {
public:
    using EntryFn = void (*)(SystemBase *, void *, uint32_t);

    SystemBase(EntryFn entry_fn);
    AtomicU32 numInvocations; 
private:
    EntryFn entry_fn_;
friend class TaskGraph;
};

template <typename SystemT>
class CustomSystem : public SystemBase {
public:
    CustomSystem();

private:
    static void entry(SystemBase *sys, void *data,
                      uint32_t invocation_offset);
};

template <typename SystemT, typename... ComponentTs>
class ParallelForSystem : public SystemBase {
public:
    ParallelForSystem(Context &ctx);

private:
    static void entry(SystemBase *sys, void *data, uint32_t invocation_offset);

    Query<ComponentTs...> query_;
};

#if 0
template <typename Fn, typename... ComponentTs>
class LambdaParallelForSystem : public ParallelForSystem<
        LambdaParallelForSystem<Fn, ComponentTs...>, ComponentTs...> {
    using ContextT = utils::FirstArgTypeExtractor<decltype(Fn::operator())>;
public:
    static LambdaParallelForSystem<Fn, ComponentTs...> * allocate(Context &ctx);
    static void deallocate(Context &ctx,
                           LambdaParallelForSystem<Fn, ComponentTs...> *lambda);

    void run(ContextT &ctx, uint32_t invocation_idx);

private:
    LambdaParallelForSystem(Fn &&fn);

    Fn fn;
};
#endif

}

#include "system.inl"
