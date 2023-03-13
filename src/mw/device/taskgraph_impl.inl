#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>
#include <madrona/mw_gpu/cu_utils.hpp>

#include "../render/interop.hpp"

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

struct TaskGraph::BlockState {
    int32_t nodeIdx;
    uint32_t totalNumInvocations;
    uint32_t funcID;
    uint32_t numThreadsPerInvocation;
    int32_t initOffset;
};

static __shared__ TaskGraph::BlockState sharedBlockState;

void inline TaskGraph::init()
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
        mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::calibration,
                                    blockDim.x * blockDim.y * blockDim.z / 32,
                                    gridDim.y,
                                    gridDim.x);

        Node &first_node = sorted_nodes_[0];

        uint32_t new_num_invocations = computeNumInvocations(first_node);
        assert(new_num_invocations != 0);
        first_node.curOffset.store_relaxed(0);
        first_node.numRemaining.store_relaxed(new_num_invocations);
        first_node.totalNumInvocations.store_relaxed(new_num_invocations);

        cur_node_idx_.store_release(0);

// #ifdef LIMIT_ACTIVE_BLOCKS
//         for (size_t i = 0; i < MADRONA_MWGPU_NUM_MEGAKERNEL_NUM_SMS; i++) {
//             block_sm_offsets_[i].store_relaxed(0);
//         }
// #endif
    }

    // init_barrier.arrive_and_wait();
    auto count = completed_blocks_.fetch_add_relaxed(1);
    if (count == gridDim.y * gridDim.x - 1) {
        synced_.store_relaxed(1);
    }
    while (synced_.load_relaxed() == 0) {
        __nanosleep(0);
    }
    count = completed_blocks_.fetch_sub_relaxed(1);
    if (count == 1) {
        synced_.store_relaxed(0);
    }

    if (thread_idx == 0) {
        sharedBlockState.nodeIdx = -1;
        sharedBlockState.initOffset = -1;
    }

#ifdef LIMIT_ACTIVE_BLOCKS
    uint32_t sm_id;
    asm("mov.u32 %0, %smid;"
        : "=r"(sm_id));
    sharedBlockState.blockSMOffset =
        block_sm_offsets_[sm_id].fetch_add_relaxed(1);
#endif
}


void inline TaskGraph::updateBlockState()
{
    uint32_t node_idx = cur_node_idx_.load_acquire();
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
        cur_node.totalNumInvocations.load_relaxed();

    uint32_t num_threads_per_invocation = cur_node.numThreadsPerInvocation;

    sharedBlockState.nodeIdx = node_idx;
    sharedBlockState.totalNumInvocations = total_invocations;
    sharedBlockState.funcID = cur_node.funcID;
    sharedBlockState.numThreadsPerInvocation = num_threads_per_invocation;
    sharedBlockState.initOffset = cur_node.curOffset.fetch_add_relaxed(
        blockDim.x * blockDim.y * blockDim.z / num_threads_per_invocation);
}

uint32_t inline TaskGraph::computeNumInvocations(Node &node)
{
    if (node.fixedCount == 0) {
        auto data_ptr = (NodeBase *)node_datas_[node.dataIDX].userData;
        return data_ptr->numDynamicInvocations;
    } else {
        return node.fixedCount;
    }
}

TaskGraph::WorkerState inline TaskGraph::getWork(NodeBase **node_data,
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
                sharedBlockState.initOffset =
                    cur_node->curOffset.fetch_add_relaxed(
                    blockDim.x * blockDim.y * blockDim.z / num_threads_per_invocation);
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
                base_offset = cur_node->curOffset.fetch_add_relaxed(
                    32 / num_threads_per_invocation);
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

void inline TaskGraph::finishWork(bool lane_executed)
{
    uint32_t num_finished_threads;
    bool is_leader;

    uint32_t num_threads_per_invocation =
        sharedBlockState.numThreadsPerInvocation;
    if (num_threads_per_invocation > 32) {
        __syncthreads();

        num_finished_threads = blockDim.x * blockDim.y * blockDim.z;

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

    uint32_t prev_remaining =
        cur_node.numRemaining.fetch_sub_acq_rel(num_finished);

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
                next_node.curOffset.store_relaxed(0);
                next_node.numRemaining.store_relaxed(new_num_invocations);
                next_node.totalNumInvocations.store_relaxed(
                    new_num_invocations);

                mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::nodeStart,
                    next_node.funcID, new_num_invocations, next_node_idx, is_leader);
            }

            cur_node_idx_.store_release(next_node_idx);
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
