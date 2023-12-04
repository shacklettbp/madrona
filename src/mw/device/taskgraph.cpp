#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>
#include <madrona/mw_gpu/cu_utils.hpp>

namespace madrona {

struct TaskGraph::BlockState {
    uint32_t nodeIdx;
    uint32_t totalNumInvocations;
    uint32_t funcID;
    uint32_t numThreadsPerInvocation;
    int32_t initOffset;
};

static __shared__ TaskGraph::BlockState sharedBlockState;

TaskGraph::TaskGraph(Node *nodes, uint32_t num_nodes, NodeData *node_datas)
    : sorted_nodes_(nodes),
      num_nodes_(num_nodes),
      node_datas_(node_datas),
      cur_node_idx_(num_nodes)
{
    for (int32_t i = 1; i <= MADRONA_MWGPU_MAX_BLOCKS_PER_SM; i++) {
        init_barriers_.emplace(i - 1, i * MADRONA_MWGPU_NUM_SMS);
    }
}

TaskGraph::~TaskGraph()
{
    rawDealloc(sorted_nodes_);
}

void TaskGraph::init(int32_t start_node_idx, int32_t end_node_idx,
                     int32_t num_blocks_per_sm)
{
    int thread_idx = threadIdx.x;
    if (thread_idx != 0) {
        return;
    }

    if (blockIdx.x == 0) {
        // reset the pointer for each run
        if (start_node_idx == 0) {
            mwGPU::DeviceTracing::resetIndex(); 
        }
        // special calibration indicating the beginning of the kernel
        mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::calibration,
                                  blockDim.x / 32, // # warps
                                  num_blocks_per_sm, // # blocks
                                  MADRONA_MWGPU_NUM_SMS); // # SMs

        end_node_idx_ = end_node_idx == -1 ? num_nodes_ : end_node_idx;

        while (start_node_idx < end_node_idx_) {
            if (computeNumInvocations(sorted_nodes_[start_node_idx]) == 0) {
                start_node_idx++;
            } else {
                break;
            }
        }

        if (start_node_idx < end_node_idx_) {
            Node &first_node = sorted_nodes_[start_node_idx];
            uint32_t new_num_invocations = computeNumInvocations(first_node);
            assert(new_num_invocations != 0);
            first_node.curOffset.store_relaxed(0);
            first_node.numRemaining.store_relaxed(new_num_invocations);
            first_node.totalNumInvocations.store_relaxed(new_num_invocations);
            mwGPU::DeviceTracing::Log(mwGPU::DeviceEvent::nodeStart,
                                        first_node.funcID,
                                        new_num_invocations,
                                        start_node_idx);
        }
        cur_node_idx_.store_release(start_node_idx);

// #ifdef LIMIT_ACTIVE_BLOCKS
//         for (size_t i = 0; i < MADRONA_MWGPU_NUM_SMS; i++) {
//             block_sm_offsets_[i].store_relaxed(0);
//         }
// #endif
    }

    auto &init_barrier = init_barriers_[num_blocks_per_sm - 1];
    init_barrier.arrive_and_wait();

    if (thread_idx == 0) {
        sharedBlockState.nodeIdx = 0xffffffff;
        sharedBlockState.initOffset = -1;
    }

// #ifdef LIMIT_ACTIVE_BLOCKS
//     uint32_t sm_id;
//     asm("mov.u32 %0, %smid;"
//         : "=r"(sm_id));
//     sharedBlockState.blockSMOffset =
//         block_sm_offsets_[sm_id].fetch_add_relaxed(1);
// #endif
}

void TaskGraph::updateBlockState()
{
    uint32_t node_idx = cur_node_idx_.load_acquire();
    if (node_idx == end_node_idx_) {
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
        blockDim.x / num_threads_per_invocation);
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

TaskGraph::WorkerState TaskGraph::getWork(
    NodeBase **node_data,
    uint32_t *run_func_id,
    uint32_t *run_node_id,
    int32_t *run_offset)
{
    const int thread_idx = threadIdx.x;
    int32_t warp_idx = thread_idx / 32;
    int32_t lane_idx = thread_idx % 32;

    uint32_t node_idx;
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

        if (node_idx == end_node_idx_) {
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
                    blockDim.x / num_threads_per_invocation);
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
    *run_node_id = node_idx;
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

        num_finished_threads = blockDim.x;

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
            if (next_node_idx < end_node_idx_) {
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

}
