#pragma once
#include <madrona/taskgraph.hpp>
#include <madrona/macros.hpp>

#include "megakernel_consts.hpp"

namespace madrona {
namespace mwGPU {

#ifdef MADRONA_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-internal"
#endif
static inline __attribute__((always_inline)) void dispatch(
        uint32_t func_id,
        mwGPU::EntryData &entry_data,
        uint32_t invocation_offset);
#ifdef MADRONA_CLANG
#pragma clang diagnostic pop
#endif

static inline __attribute__((always_inline)) void megakernelImpl()
{
    {
        TaskGraph *taskgraph = (TaskGraph *)GPUImplConsts::get().taskGraph;
        taskgraph->init();
    }

    while (true) {
        TaskGraph *taskgraph = (TaskGraph *)GPUImplConsts::get().taskGraph;

        mwGPU::EntryData *entry_data;
        uint32_t func_id;
        int32_t invocation_offset;
        TaskGraph::WorkerState worker_state = taskgraph->getWork(
            &entry_data, &func_id, &invocation_offset);

        if (worker_state == TaskGraph::WorkerState::Exit) {
            break;
        }

        if (worker_state == TaskGraph::WorkerState::Loop) {
            __nanosleep(0);
            continue;
        }

        if (worker_state == TaskGraph::WorkerState::Run) {
            dispatch(func_id, *entry_data, invocation_offset);
        }

        taskgraph->finishWork();
    }
}

}
}

extern "C" __global__ void
__launch_bounds__(madrona::consts::numMegakernelThreads,
                  madrona::consts::numMegakernelBlocksPerSM)
madronaMWGPUMegakernel()
{
    madrona::mwGPU::megakernelImpl();
}
