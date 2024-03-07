#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>
#include <madrona/mw_gpu/cu_utils.hpp>

namespace madrona::mwGPU {

#ifdef MADRONA_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-internal"
#endif
static inline __attribute__((always_inline)) void dispatch(
        uint32_t func_id,
        NodeBase *node_data,
        uint32_t invocation_offset);
#ifdef MADRONA_CLANG
#pragma clang diagnostic pop
#endif

static inline __attribute__((always_inline)) void megakernelImpl(
    uint32_t taskgraph_id, int32_t start_node_idx,
    int32_t end_node_idx, int32_t num_blocks_per_sm)
{
    {
        mwGPU::getTaskGraph(taskgraph_id).init(
            start_node_idx, end_node_idx, num_blocks_per_sm);
    }

    __syncthreads();

    while (true) {
        TaskGraph &taskgraph = mwGPU::getTaskGraph(taskgraph_id);

        NodeBase *node_data;
        uint32_t func_id;
        uint32_t node_id;
        int32_t invocation_offset;
        TaskGraph::WorkerState worker_state = taskgraph.getWork(
            &node_data, &func_id, &node_id, &invocation_offset);

        if (worker_state == TaskGraph::WorkerState::Exit) {
            DeviceTracing::Log(
                mwGPU::DeviceEvent::blockExit,
                func_id, invocation_offset, node_id);
            break;
        }

        if (worker_state == TaskGraph::WorkerState::Loop) {
            __nanosleep(0);
            continue;
        }

        bool lane_executed;
        if (worker_state == TaskGraph::WorkerState::Run) {
            mwGPU::DeviceTracing::Log(
                mwGPU::DeviceEvent::blockStart,
                func_id, invocation_offset, node_id, threadIdx.x % 32 == 0);

            dispatch(func_id, node_data, invocation_offset);

            mwGPU::DeviceTracing::Log(
                mwGPU::DeviceEvent::blockWait,
                func_id, invocation_offset, node_id, threadIdx.x % 32 == 0);
            lane_executed = true;
        } else {
            lane_executed = false;
        }

        taskgraph.finishWork(lane_executed);
    }
}

}
