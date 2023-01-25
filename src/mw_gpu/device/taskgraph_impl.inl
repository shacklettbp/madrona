#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>
#include <madrona/mw_gpu/cu_utils.hpp>

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
    int32_t nodeIdx;
    uint32_t totalNumInvocations;
    uint32_t funcID;
    uint32_t numThreadsPerInvocation;
    int32_t initOffset;
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
        // reset the pointer for each run
        mwGPU::DeviceTracing::resetIndex();
        // special calibration indicating the beginning of the kernel
        mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::calibration, 0, 0, 0);

        Node &first_node = sorted_nodes_[0];

        uint32_t new_num_invocations = computeNumInvocations(first_node);
        assert(new_num_invocations != 0);
        first_node.curOffset.store(0, std::memory_order_relaxed);
        first_node.numRemaining.store(new_num_invocations,
                                    std::memory_order_relaxed);
        first_node.totalNumInvocations.store(new_num_invocations,
            std::memory_order_relaxed);

        cur_node_idx_.store(0, std::memory_order_release);

#ifdef LIMIT_ACTIVE_BLOCKS
        for (size_t i = 0; i < num_SMs_; i++) {
            block_sm_offsets_[i].store(0, std::memory_order_relaxed);
        }
#endif
    }

    init_barrier_.arrive_and_wait();

    if (thread_idx == 0) {
        sharedBlockState.nodeIdx = -1;
        sharedBlockState.initOffset = -1;
    }

#ifdef LIMIT_ACTIVE_BLOCKS
    uint32_t sm_id;
    asm("mov.u32 %0, %smid;"
        : "=r"(sm_id));
    sharedBlockState.blockSMOffset = block_sm_offsets_[sm_id].fetch_add(1, std::memory_order_relaxed);
#endif
}

void TaskGraph::updateBlockState()
{
    uint32_t node_idx = cur_node_idx_.load(std::memory_order_acquire);
    if (node_idx == num_nodes_) {
        sharedBlockState.nodeIdx = node_idx;
        return;
    }

    if (node_idx == sharedBlockState.nodeIdx) {
        sharedBlockState.initOffset = -1;
        return;
    }

    Node &cur_node = sorted_nodes_[node_idx];

    uint32_t total_invocations =
        cur_node.totalNumInvocations.load(std::memory_order_relaxed);

    uint32_t num_threads_per_invocation = cur_node.numThreadsPerInvocation;

    sharedBlockState.nodeIdx = node_idx;
    sharedBlockState.totalNumInvocations = total_invocations;
    sharedBlockState.funcID = cur_node.funcID;
    sharedBlockState.numThreadsPerInvocation = num_threads_per_invocation;
    sharedBlockState.initOffset = cur_node.curOffset.fetch_add(
        consts::numMegakernelThreads / num_threads_per_invocation,
            std::memory_order_relaxed);
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
    const int thread_idx = threadIdx.x;
    int32_t warp_idx = thread_idx / 32;
    int32_t lane_idx = thread_idx % 32;

    int32_t node_idx;
    Node *cur_node;
    int32_t total_num_invocations;
    int32_t num_threads_per_invocation;
    int32_t base_offset;
    bool run_new_node = false;

    auto blockGetNextNode = [&]() {
        __syncthreads();

        if (thread_idx == 0) {
            updateBlockState();
        }
        __syncthreads();

        node_idx = sharedBlockState.nodeIdx;

        if (node_idx == num_nodes_) {
            return WorkerState::Exit;
        }

        int32_t block_init_offset = sharedBlockState.initOffset;
        if (block_init_offset == -1) {
            return WorkerState::Loop;
        }

        cur_node = &sorted_nodes_[node_idx];
        num_threads_per_invocation =
            sharedBlockState.numThreadsPerInvocation;
        total_num_invocations = sharedBlockState.totalNumInvocations;
        base_offset = block_init_offset +
            (warp_idx * 32) / num_threads_per_invocation;

        run_new_node = true;
        return WorkerState::Run;
    };

    if (sharedBlockState.initOffset == -1) {
        WorkerState ctrl = blockGetNextNode();
        if (ctrl != WorkerState::Run) {
            return ctrl;
        }
    } else {
        node_idx = sharedBlockState.nodeIdx;
        cur_node = &sorted_nodes_[node_idx];

        total_num_invocations = sharedBlockState.totalNumInvocations;
        num_threads_per_invocation = sharedBlockState.numThreadsPerInvocation;
        if (num_threads_per_invocation > 32) {
            if (thread_idx == 0) {
                sharedBlockState.initOffset = cur_node->curOffset.fetch_add(
                    consts::numMegakernelThreads / num_threads_per_invocation,
                    std::memory_order_relaxed);
            }

            __syncthreads();
            base_offset = sharedBlockState.initOffset;

            if (base_offset >= total_num_invocations) {
                WorkerState ctrl = blockGetNextNode();
                if (ctrl != WorkerState::Run) {
                    return ctrl;
                }
            }
        } else {
            if (lane_idx == 0) {
                base_offset = cur_node->curOffset.fetch_add(
                    32 / num_threads_per_invocation, std::memory_order_relaxed);
            }
            base_offset = __shfl_sync(mwGPU::allActive, base_offset, 0);

            if (base_offset >= total_num_invocations) {
                WorkerState ctrl = blockGetNextNode();
                if (ctrl != WorkerState::Run) {
                    return ctrl;
                }
            }
        }
    }

    if (base_offset >= total_num_invocations) {
        return WorkerState::Loop;
    }

    int32_t thread_offset = base_offset +
        lane_idx / num_threads_per_invocation;
    if (thread_offset >= total_num_invocations) {
        return WorkerState::PartialRun;
    }

    *node_data = (NodeBase *)
        node_datas_[sorted_nodes_[sharedBlockState.nodeIdx].dataIDX].userData;
    *run_func_id = sharedBlockState.funcID;
    *run_offset = thread_offset;

    return WorkerState::Run;
}

void TaskGraph::finishWork(bool lane_executed)
{
    uint32_t num_finished_threads;
    bool is_leader;

    uint32_t num_threads_per_invocation =
        sharedBlockState.numThreadsPerInvocation;
    if (num_threads_per_invocation > 32) {
        __syncthreads();

        num_finished_threads = consts::numMegakernelThreads;

        is_leader = threadIdx.x == 0;
    } else {
        __syncwarp(mwGPU::allActive);
        num_finished_threads =
            __popc(__ballot_sync(mwGPU::allActive, lane_executed));

        is_leader = threadIdx.x % 32 == 0;
    }

    if (!is_leader) {
        return;
    }

    uint32_t num_finished = num_finished_threads /
        num_threads_per_invocation;

    uint32_t node_idx = sharedBlockState.nodeIdx;

    Node &cur_node = sorted_nodes_[node_idx];

    uint32_t prev_remaining = cur_node.numRemaining.fetch_sub(num_finished,
        std::memory_order_acq_rel);

    if (prev_remaining == num_finished) {

        mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::nodeFinish,
            sharedBlockState.funcID, sharedBlockState.totalNumInvocations,
            node_idx, is_leader);

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

                mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::nodeStart,
                    next_node.funcID, new_num_invocations, next_node_idx, is_leader);
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

    __syncthreads();

    while (true) {
        TaskGraph *taskgraph = (TaskGraph *)GPUImplConsts::get().taskGraph;

        NodeBase *node_data;
        uint32_t func_id;
        int32_t invocation_offset;
        TaskGraph::WorkerState worker_state = taskgraph->getWork(
            &node_data, &func_id, &invocation_offset);

        if (worker_state == TaskGraph::WorkerState::Exit) {
            DeviceTracing::Log(
                mwGPU::DeviceEvent::blockExit,
                func_id, invocation_offset, sharedBlockState.nodeIdx);
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
                sharedBlockState.funcID, invocation_offset, sharedBlockState.nodeIdx, threadIdx.x % 32 == 0);

            dispatch(func_id, node_data, invocation_offset);

            mwGPU::DeviceTracing::Log(
                mwGPU::DeviceEvent::blockWait,
                sharedBlockState.funcID, invocation_offset, sharedBlockState.nodeIdx, threadIdx.x % 32 == 0);
            lane_executed = true;
        } else {
            lane_executed = false;
        }

        taskgraph->finishWork(lane_executed);
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
        total_bytes, (uint64_t)alignof(mwGPU::DeviceTracing));

    total_bytes = device_tracing_offset + sizeof(mwGPU::DeviceTracing);

    *out_constants = GPUImplConsts {
        /*.jobSystemAddr = */                  (void *)0ul,
        /* .taskGraph = */                     (void *)0ul,
        /* .stateManagerAddr = */              (void *)state_mgr_offset,
        /* .worldDataAddr =  */                (void *)world_data_offset,
        /* .hostAllocatorAddr = */             (void *)host_allocator_offset,
        /* .hostPrintAddr = */                 (void *)host_print_offset,
        /* .tmpAllocatorAddr */                (void *)tmp_allocator_offset,
        /* .deviceTracingAddr = */             (void *)device_tracing_offset,
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
