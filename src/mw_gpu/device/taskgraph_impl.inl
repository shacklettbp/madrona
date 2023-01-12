#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>

#include "megakernel_consts.hpp"

namespace madrona {

namespace mwGPU {

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

}

TaskGraph::TaskGraph(Node *nodes, uint32_t num_nodes, NodeData *node_datas)
    : sorted_nodes_(nodes),
      num_nodes_(num_nodes),
      node_datas_(node_datas),
      cur_node_idx_(num_nodes),
      init_barrier_(MADRONA_MWGPU_NUM_MEGAKERNEL_BLOCKS)
{}

TaskGraph::~TaskGraph()
{
    rawDealloc(sorted_nodes_);
}

struct TaskGraph::BlockState {
    WorkerState state;
    uint32_t nodeIdx;
    uint32_t numInvocations;
    uint32_t funcID;
    uint32_t runOffset;
};

static __shared__ TaskGraph::BlockState sharedBlockState;

void TaskGraph::init()
{
    int thread_idx = threadIdx.x;
    if (thread_idx != 0) {
        return;
    }

    int block_idx = blockIdx.x;

    if (block_idx == 0) {
        Node &first_node = sorted_nodes_[0];

        uint32_t new_num_invocations = computeNumInvocations(first_node);
        assert(new_num_invocations != 0);
        first_node.curOffset.store(0, std::memory_order_relaxed);
        first_node.numRemaining.store(new_num_invocations,
                                    std::memory_order_relaxed);
        first_node.totalNumInvocations.store(new_num_invocations,
            std::memory_order_relaxed);

        cur_node_idx_.store(0, std::memory_order_release);
    }

    init_barrier_.arrive_and_wait();

    device_tracing->DeviceEventLogging(mwGPU::DeviceEvent::calibration, 0, 0, 0);
}

void TaskGraph::setBlockState()
{
    uint32_t node_idx = cur_node_idx_.load(std::memory_order_acquire);
    if (node_idx == num_nodes_) {
        sharedBlockState.state = WorkerState::Exit;
        return;
    }

    Node &cur_node = sorted_nodes_[node_idx];

    uint32_t cur_offset = 
        cur_node.curOffset.load(std::memory_order_relaxed);

    uint32_t total_invocations =
        cur_node.totalNumInvocations.load(std::memory_order_relaxed);

    if (cur_offset >= total_invocations) {
        sharedBlockState.state = WorkerState::Loop;
        return;
    }

    cur_offset = cur_node.curOffset.fetch_add(consts::numMegakernelThreads,
        std::memory_order_relaxed);

    if (cur_offset >= total_invocations) {
        sharedBlockState.state = WorkerState::Loop;
        return;
    }

    sharedBlockState.state = WorkerState::Run;
    sharedBlockState.nodeIdx = node_idx;
    sharedBlockState.numInvocations = total_invocations;
    sharedBlockState.funcID = cur_node.funcID;
    sharedBlockState.runOffset = cur_offset;
}

uint32_t TaskGraph::computeNumInvocations(Node &node)
{
    if (node.fixedCount == 0) {
        auto data_ptr = (NodeBase *)node_datas_[node.dataIDX].userData;
        return data_ptr->numDynamicInvocations;
    } else {
        return node.fixedCount;
    }
}

TaskGraph::WorkerState TaskGraph::getWork(NodeBase **node_data,
                                          uint32_t *run_func_id,
                                          int32_t *run_offset)
{
    int thread_idx = threadIdx.x;

    if (thread_idx == 0) {
        setBlockState();
    }

    __syncthreads();

    WorkerState worker_state = sharedBlockState.state;

    if (worker_state != WorkerState::Run) {
        return worker_state;
    }

    uint32_t num_invocations = sharedBlockState.numInvocations;
    uint32_t base_offset = sharedBlockState.runOffset;

    int32_t thread_offset = base_offset + thread_idx;
    if (thread_offset >= num_invocations) {
        return WorkerState::PartialRun;
    }

    *node_data = (NodeBase *)
        node_datas_[sorted_nodes_[sharedBlockState.nodeIdx].dataIDX].userData;
    *run_func_id = sharedBlockState.funcID;
    *run_offset = thread_offset;

    return WorkerState::Run;
}

void TaskGraph::finishWork()
{
    int thread_idx = threadIdx.x;
    __syncthreads();

    if (thread_idx != 0) return;

    uint32_t num_finished = std::min(
        sharedBlockState.numInvocations - sharedBlockState.runOffset,
        consts::numMegakernelThreads);

    uint32_t node_idx = sharedBlockState.nodeIdx;
    Node &cur_node = sorted_nodes_[node_idx];

    uint32_t prev_remaining = cur_node.numRemaining.fetch_sub(num_finished,
        std::memory_order_acq_rel);

    if (prev_remaining == num_finished) {

        device_tracing->DeviceEventLogging(mwGPU::DeviceEvent::nodeFinish,
                                           cur_node.funcID, sharedBlockState.numInvocations, node_idx);

        uint32_t next_node_idx = node_idx + 1;

        while (true) {
            if (next_node_idx < num_nodes_) {
                uint32_t new_num_invocations =
                    computeNumInvocations(sorted_nodes_[next_node_idx]);

                if (new_num_invocations == 0) {
                    next_node_idx++;
                    continue;
                }

                Node &next_node = sorted_nodes_[next_node_idx];
                next_node.curOffset.store(0, std::memory_order_relaxed);
                next_node.numRemaining.store(new_num_invocations,
                                            std::memory_order_relaxed);
                next_node.totalNumInvocations.store(new_num_invocations,
                    std::memory_order_relaxed);

                device_tracing->DeviceEventLogging(
                    mwGPU::DeviceEvent::nodeStart,
                    next_node.funcID, new_num_invocations, next_node_idx);
            }

            cur_node_idx_.store(next_node_idx, std::memory_order_release);
            break;
        }
    }
}

namespace mwGPU {

static inline __attribute__((always_inline)) void megakernelImpl()
{
    {
        TaskGraph *taskgraph = (TaskGraph *)GPUImplConsts::get().taskGraph;
        taskgraph->init();
    }

    TaskGraph *taskgraph = (TaskGraph *)GPUImplConsts::get().taskGraph;

    while (true) {
        NodeBase *node_data;
        uint32_t func_id;
        int32_t invocation_offset;
        TaskGraph::WorkerState worker_state = taskgraph->getWork(
            &node_data, &func_id, &invocation_offset);

        if (worker_state == TaskGraph::WorkerState::Exit) {
            break;
        }

        if (worker_state == TaskGraph::WorkerState::Loop) {
            __nanosleep(0);
            continue;
        }

        if (worker_state == TaskGraph::WorkerState::Run) {
            taskgraph->device_tracing->DeviceEventLogging(
                mwGPU::DeviceEvent::blockStart,
                func_id, invocation_offset, sharedBlockState.nodeIdx);
            dispatch(func_id, node_data, invocation_offset);
            taskgraph->device_tracing->DeviceEventLogging(
                mwGPU::DeviceEvent::blockWait,
                func_id, invocation_offset, sharedBlockState.nodeIdx);
        }

        taskgraph->finishWork();
    }
}

}
}

extern "C" __global__ void madronaMWGPUComputeConstants(
    uint32_t num_worlds,
    uint32_t num_world_data_bytes,
    uint32_t world_data_alignment,
    madrona::mwGPU::GPUImplConsts *out_constants,
    size_t *job_system_buffer_size)
{
    using namespace madrona;
    using namespace madrona::mwGPU;

    uint64_t total_bytes = sizeof(TaskGraph);

    uint64_t state_mgr_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(StateManager));

    total_bytes = state_mgr_offset + sizeof(StateManager);

    uint64_t world_data_offset =
        utils::roundUp(total_bytes, (uint64_t)world_data_alignment);

    uint64_t total_world_bytes =
        (uint64_t)num_world_data_bytes * (uint64_t)num_worlds;

    total_bytes = world_data_offset + total_world_bytes;

    uint64_t host_allocator_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(mwGPU::HostAllocator));

    total_bytes = host_allocator_offset + sizeof(mwGPU::HostAllocator);

    uint64_t host_print_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(mwGPU::HostPrint));

    total_bytes = host_print_offset + sizeof(mwGPU::HostPrint);

    uint64_t tmp_allocator_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(TmpAllocator));

    total_bytes = tmp_allocator_offset + sizeof(TmpAllocator);

    uint64_t device_tracing_offset = utils::roundUp(
        total_bytes, (uint64_t)alignof(mwGPU::DeviceTracing *));

    total_bytes = device_tracing_offset + sizeof(mwGPU::DeviceTracing *);

    *out_constants = GPUImplConsts {
        /*.jobSystemAddr = */                  (void *)0ul,
        /* .taskGraph = */                     (void *)0ul,
        /* .stateManagerAddr = */              (void *)state_mgr_offset,
        /* .worldDataAddr =  */                (void *)world_data_offset,
        /* .hostAllocatorAddr = */             (void *)host_allocator_offset,
        /* .hostPrintAddr = */                 (void *)host_print_offset,
        /* .tmpAllocatorAddr */                (void *)tmp_allocator_offset,
        /* .DeviceTracing = */                 (void **)device_tracing_offset,
        /* .rendererASInstancesAddrs = */      (void **)0ul,
        /* .rendererInstanceCountsAddr = */    (void *)0ul,
        /* .rendererBLASesAddr = */            (void *)0ul,
        /* .rendererViewDatasAddr = */         (void *)0ul,
        /* .numWorldDataBytes = */             num_world_data_bytes,
        /* .numWorlds = */                     num_worlds,
        /* .jobGridsOffset = */                (uint32_t)0,
        /* .jobListOffset = */                 (uint32_t)0,
        /* .maxJobsPerGrid = */                0,
        /* .sharedJobTrackerOffset = */        (uint32_t)0,
        /* .userJobTrackerOffset = */          (uint32_t)0,
    };

    *job_system_buffer_size = total_bytes;
}

extern "C" __global__ void
__launch_bounds__(madrona::consts::numMegakernelThreads,
                  madrona::consts::numMegakernelBlocksPerSM)
madronaMWGPUMegakernel()
{
    madrona::mwGPU::megakernelImpl();
}
