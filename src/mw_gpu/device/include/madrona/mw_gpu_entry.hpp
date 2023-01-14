#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>

namespace madrona {
namespace mwGPU {
namespace entryKernels {

template <typename ContextT, typename WorldDataT, typename... InitTs>
__global__ void initECS(HostAllocInit alloc_init, void *print_channel,
                        void *device_tracing, void **exported_columns)
{
    HostAllocator *host_alloc = mwGPU::getHostAllocator();
    new (host_alloc) HostAllocator(alloc_init);

    auto host_print = (HostPrint *)GPUImplConsts::get().hostPrintAddr;
    new (host_print) HostPrint(print_channel);

    auto device_tracing_addr = (DeviceTracing **)GPUImplConsts::get().deviceTracingAddr;
    *device_tracing_addr = reinterpret_cast<DeviceTracing *>(device_tracing);

    TmpAllocator &tmp_alloc = TmpAllocator::get();
    new (&tmp_alloc) TmpAllocator();

    StateManager *state_mgr = mwGPU::getStateManager();
    new (state_mgr) StateManager(0);

    ECSRegistry ecs_registry(*state_mgr, exported_columns);
    WorldDataT::registerTypes(ecs_registry);
}

template <typename ContextT, typename WorldDataT, typename... InitTs>
__global__ void initWorlds(int32_t num_worlds, InitTs * ...inits)
{
    int32_t world_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (world_idx >= num_worlds) {
        return;
    }

    WorldBase *world = TaskGraph::getWorld(world_idx);

    ContextT ctx = TaskGraph::makeContext<ContextT>(WorldID {
        world_idx,
    });

    new (world) WorldDataT(ctx, inits[world_idx] ...);
}

template <typename ContextT, typename WorldDataT, typename... InitTs>
__global__ void initTasks()
{
    TaskGraph::Builder builder(1024, 1024 * 2, 1024 * 5);
    WorldDataT::setupTasks(builder);

    builder.build((TaskGraph *)mwGPU::GPUImplConsts::get().taskGraph);
}

}

template <auto init_ecs, auto init_worlds, auto init_tasks>
struct MWGPUEntryInstantiate {};

template <typename ContextT, typename WorldDataT, typename... InitTs>
struct alignas(16) MWGPUEntry : MWGPUEntryInstantiate<
    entryKernels::initECS<ContextT, WorldDataT, InitTs...>,
    entryKernels::initWorlds<ContextT, WorldDataT, InitTs...>,
    entryKernels::initTasks<ContextT, WorldDataT, InitTs...>>
{};

}
}

// This macro forces MWGPUEntry to be instantiated, which in turn instantiates
// the entryKernels::* __global__ entry points. static_assert with a trivially
// true check leaves no side effects in the scope where this macro is called.
#define MADRONA_BUILD_MWGPU_ENTRY(ContextT, WorldT, ...) \
    static_assert(alignof(::madrona::mwGPU::MWGPUEntry<ContextT, WorldT, \
        __VA_ARGS__>) == 16);
