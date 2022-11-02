#pragma once

#include <madrona/fwd.hpp>
#include <madrona/utils.hpp>
#include <madrona/query.hpp>

#include <cstdint>

namespace madrona {

class SystemBase;

class SystemInit {
public:
    SystemInit(JobManager &job_mgr, StateManager &state_mgr);

private:
    JobManager *job_mgr_;
    StateManager *state_mgr_;

friend class SystemBase;
};

class SystemBase {
public:
    using GenericFn = void (*)();

    SystemBase(GenericFn fn_ptr);
    void enqueue(Context &ctx);

protected:

private:
    GenericFn fn_ptr_;
};

template <typename SystemT>
class CustomSystem : public SystemBase {
public:
    CustomSystem(const SystemInit &init);

};

template <typename SystemT, typename... ComponentTs>
class ParallelForSystem : public SystemBase {
public:
    ParallelForSystem(const SystemInit &init);

private:
    using ContextT = utils::FirstArgTypeExtractor<decltype(SystemT::run)>;

    static void entry(Context &ctx, uint32_t invocation_offset, uint32_t num_invocations);

    Query<ComponentTs...> query_;
};

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

}

#include "system.inl"
