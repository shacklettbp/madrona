#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/memory.hpp>

namespace madrona {
namespace mwGPU {
namespace entryKernels {

template <typename ContextT, typename WorldDataT, typename InitT>
__global__ void initECS(HostAllocInit alloc_init, void **exported_columns)
{
    HostAllocator *host_alloc = mwGPU::getHostAllocator();
    new (host_alloc) HostAllocator(alloc_init);

    TmpAllocator &tmp_alloc = TmpAllocator::get();
    new (&tmp_alloc) TmpAllocator();

    StateManager *state_mgr = mwGPU::getStateManager();
    new (state_mgr) StateManager(0);

    ECSRegistry ecs_registry(*state_mgr, exported_columns);
    WorldDataT::registerTypes(ecs_registry);
}

template <typename ContextT, typename WorldDataT, typename InitT>
__global__ void initWorlds(int32_t num_worlds,
                           void *inits_raw)
{
    InitT *inits = (InitT *)inits_raw;
    int32_t world_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (world_idx >= num_worlds) {
        return;
    }

    const InitT &init = inits[world_idx];
    WorldBase *world = TaskGraph::getWorld(world_idx);

    ContextT ctx =
        mwGPU::EntryBase<ContextT, WorldDataT>::makeContext(WorldID {
            world_idx,
        });

    new (world) WorldDataT(ctx, init);
}

template <typename ContextT, typename WorldDataT, typename InitT>
__global__ void initTasks()
{
    TaskGraph::Builder builder(1024, 1024);
    WorldDataT::setupTasks(builder);

    builder.build((TaskGraph *)mwGPU::GPUImplConsts::get().taskGraph);
}

}

template <typename ContextT, typename WorldDataT, typename InitT,
          decltype(entryKernels::initECS<ContextT, WorldDataT, InitT>) =
              entryKernels::initECS<ContextT, WorldDataT, InitT>,
          decltype(entryKernels::initWorlds<ContextT, WorldDataT, InitT>) =
              entryKernels::initWorlds<ContextT, WorldDataT, InitT>,
          decltype(entryKernels::initTasks<ContextT, WorldDataT, InitT>) =
              entryKernels::initTasks<ContextT, WorldDataT, InitT>
         >
struct alignas(16) MWGPUEntry {};

}
}

// This macro forces MWGPUEntry to be instantiated, which in turn instantiates
// the entryKernels::* __global__ entry points. static_assert with a trivially
// true check leaves no side effects in the scope where this macro is called.
#define MADRONA_BUILD_MWGPU_ENTRY(ContextT, WorldT, InitT) \
    static_assert(\
        alignof(::madrona::mwGPU::MWGPUEntry<ContextT, WorldT, InitT>) == 16);
