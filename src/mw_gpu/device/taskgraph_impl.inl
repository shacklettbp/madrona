#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>

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
    uint32_t totalNumInvocations;
    uint32_t funcID;
    uint32_t runOffset;
    uint32_t numThreadsPerInvocation;
#ifdef LIMIT_ACTIVE_THREADS
    uint32_t activeThreads;
#endif
#ifdef LIMIT_ACTIVE_BLOCKS
    uint32_t blockSMOffset;
#endif
#ifdef FETCH_MULTI_INVOCATIONS
    uint32_t numInvocationsPerThread;
#endif
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
        // special calibration indicating the beginning of the kernel
        mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::calibration, 1, 0, 0);

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

    // todo: no longer needed for global timer
    mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::calibration, 0, 0, 0);

#ifdef LIMIT_ACTIVE_BLOCKS
    uint32_t sm_id;
    asm("mov.u32 %0, %smid;"
        : "=r"(sm_id));
    sharedBlockState.blockSMOffset = block_sm_offsets_[sm_id].fetch_add(1, std::memory_order_relaxed);
#endif
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

    uint32_t num_threads_per_invocation = cur_node.numThreadsPerInvocation;

    uint32_t num_active_threads = consts::numMegakernelThreads;
    float num_active_blocks = (float)(total_invocations - cur_offset) * num_threads_per_invocation / num_active_threads / num_SMs_;

#ifdef LIMIT_ACTIVE_THREADS
    while (num_active_blocks < 1 && num_active_threads > max(32, num_threads_per_invocation)) {
        num_active_threads /= 2;
        num_active_blocks *= 2;
    }
#endif

#ifdef LIMIT_ACTIVE_BLOCKS
    // didn't find it really helped
    // potentially because of better cache hit when placing blocks together
    // maybe only enforce it for certain funcs
    // useless for large amount of worlds
    if (sharedBlockState.blockSMOffset > num_active_blocks) {
        sharedBlockState.state = WorkerState::Loop;
        return;
    }
#endif

    uint32_t num_invocations_per_thread = 1;

#ifdef FETCH_MULTI_INVOCATIONS
    // only useful when excessive block invocation happened, e.g., warp parallelism narrow phase implementation
    while (num_active_blocks >= 16) {
        num_invocations_per_thread *= 2;
        num_active_blocks /= 2;
    }
#endif

    cur_offset = cur_node.curOffset.fetch_add(num_active_threads / num_threads_per_invocation * num_invocations_per_thread,
        std::memory_order_relaxed);

    if (cur_offset >= total_invocations) {
        sharedBlockState.state = WorkerState::Loop;
        return;
    }

    sharedBlockState.state = WorkerState::Run;
    sharedBlockState.nodeIdx = node_idx;
    sharedBlockState.totalNumInvocations = total_invocations;
    sharedBlockState.funcID = cur_node.funcID;
    sharedBlockState.runOffset = cur_offset;
    sharedBlockState.numThreadsPerInvocation = num_threads_per_invocation;
#ifdef LIMIT_ACTIVE_THREADS
    sharedBlockState.activeThreads = num_active_threads;
#endif
#ifdef FETCH_MULTI_INVOCATIONS
    sharedBlockState.numInvocationsPerThread = num_invocations_per_thread;
#endif
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

    uint32_t total_num_invocations = sharedBlockState.totalNumInvocations;
    uint32_t base_offset = sharedBlockState.runOffset;

#ifdef LIMIT_ACTIVE_THREADS
    if (thread_idx >= sharedBlockState.activeThreads) {
        return WorkerState::PartialRun;
    }
#endif
    int32_t thread_offset = base_offset +
        thread_idx / sharedBlockState.numThreadsPerInvocation;
    if (thread_offset >= total_num_invocations) {
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

    uint32_t node_idx = sharedBlockState.nodeIdx;

    mwGPU::DeviceTracing::Log(
        mwGPU::DeviceEvent::blockWait,
        sharedBlockState.funcID, sharedBlockState.runOffset, node_idx);

    uint32_t num_finished = std::min(
        sharedBlockState.totalNumInvocations - sharedBlockState.runOffset,
#ifdef LIMIT_ACTIVE_THREADS
        sharedBlockState.activeThreads 
#else
        consts::numMegakernelThreads
#endif
           / sharedBlockState.numThreadsPerInvocation
#ifdef FETCH_MULTI_INVOCATIONS
             * sharedBlockState.numInvocationsPerThread);
#else
            );
#endif
    Node &cur_node = sorted_nodes_[node_idx];

    uint32_t prev_remaining = cur_node.numRemaining.fetch_sub(num_finished,
        std::memory_order_acq_rel);

    if (prev_remaining == num_finished) {

        mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::nodeFinish,
            sharedBlockState.funcID, sharedBlockState.totalNumInvocations,
            node_idx);

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

        if (worker_state == TaskGraph::WorkerState::Run) {
            DeviceTracing::Log(
                mwGPU::DeviceEvent::blockStart,
                func_id, invocation_offset, sharedBlockState.nodeIdx);
#ifdef FETCH_MULTI_INVOCATIONS
            for (size_t i = 0; i < sharedBlockState.numInvocationsPerThread; i++) {
#endif
                dispatch(func_id, node_data, invocation_offset);
#ifdef FETCH_MULTI_INVOCATIONS
                invocation_offset += 
#ifdef LIMIT_ACTIVE_THREADS
                    consts::numMegakernelThreads
#else
                    sharedBlockState.activeThreads 
#endif
                    / sharedBlockState.numThreadsPerInvocation;
            }
#endif
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
