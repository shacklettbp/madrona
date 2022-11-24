#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>

namespace madrona {
namespace consts {
static constexpr uint32_t numMegakernelThreads = 256;
}

TaskGraph::Builder::Builder(uint32_t max_num_nodes,
                            uint32_t max_num_dependencies)
    : nodes_((StagedNode *)malloc(sizeof(StagedNode) * max_num_nodes)),
      num_nodes_(0),
      all_dependencies_((NodeID *)malloc(sizeof(NodeID) * max_num_dependencies)),
      num_dependencies_(0)
{}

TaskGraph::Builder::~Builder()
{
    free(nodes_);
    free(all_dependencies_);
}

TaskGraph::NodeID TaskGraph::Builder::registerNode(
    const NodeInfo &node_info,
    Span<const NodeID> dependencies)
{
    uint32_t offset = num_dependencies_;
    uint32_t num_deps = dependencies.size();

    num_dependencies_ += num_deps;

    for (int i = 0; i < (int)num_deps; i++) {
        all_dependencies_[offset + i] = dependencies[i];
    }

    nodes_[num_nodes_++] = StagedNode {
        node_info,
        offset,
        num_deps,
    };

    return NodeID {
        num_nodes_ - 1,
    };
}

void TaskGraph::Builder::build(TaskGraph *out)
{
    assert(nodes_[0].numDependencies == 0);
    NodeState *sorted_nodes = 
        (NodeState *)malloc(sizeof(NodeState) * num_nodes_);
    bool *queued = (bool *)malloc(num_nodes_ * sizeof(bool));
    new (&sorted_nodes[0]) NodeState {
        nodes_[0].node,
        0,
        0,
    };
    queued[0] = true;

    uint32_t num_remaining_nodes = num_nodes_ - 1;
    uint32_t *remaining_nodes =
        (uint32_t *)malloc(num_remaining_nodes * sizeof(uint32_t));

    for (int64_t i = 1; i < (int64_t)num_nodes_; i++) {
        queued[i]  = false;
        remaining_nodes[i - 1] = i;
    }

    uint32_t sorted_idx = 1;

    while (num_remaining_nodes > 0) {
        uint32_t cur_node_idx = remaining_nodes[0];
        StagedNode &cur_node = nodes_[cur_node_idx];

        bool dependencies_satisfied = true;
        for (uint32_t dep_offset = 0; dep_offset < cur_node.numDependencies;
             dep_offset++) {
            uint32_t dep_nodetem_idx =
                all_dependencies_[cur_node.dependencyOffset + dep_offset].id;
            if (!queued[dep_nodetem_idx]) {
                dependencies_satisfied = false;
                break;
            }
        }

        remaining_nodes[0] =
            remaining_nodes[num_remaining_nodes - 1];
        if (!dependencies_satisfied) {
            remaining_nodes[num_remaining_nodes - 1] =
                cur_node_idx;
        } else {
            queued[cur_node_idx] = true;
            new (&sorted_nodes[sorted_idx++]) NodeState {
                cur_node.node,
                0,
                0,
                0,
            };
            num_remaining_nodes--;
        }
    }

    free(remaining_nodes);
    free(queued);

    new (out) TaskGraph(sorted_nodes, num_nodes_);
}

TaskGraph::TaskGraph(NodeState *nodes, uint32_t num_nodes)
    : sorted_nodes_(nodes),
      num_nodes_(num_nodes),
      cur_node_idx_(num_nodes),
      init_barrier_(82 * 5)
{}

TaskGraph::~TaskGraph()
{
    free(sorted_nodes_);
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
        NodeState &first_node = sorted_nodes_[0];

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
}

void TaskGraph::setBlockState()
{
    uint32_t node_idx = cur_node_idx_.load(std::memory_order_acquire);
    if (node_idx == num_nodes_) {
        sharedBlockState.state = WorkerState::Exit;
        return;
    }

    NodeState &cur_node = sorted_nodes_[node_idx];

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
    sharedBlockState.funcID = cur_node.info.funcID;
    sharedBlockState.runOffset = cur_offset;
}

uint32_t TaskGraph::computeNumInvocations(NodeState &node)
{
    switch (node.info.type) {
        case NodeType::ParallelFor: {
            StateManager *state_mgr = mwGPU::getStateManager();
            QueryRef *query_ref = node.info.data.parallelFor.query;
            return state_mgr->numMatchingEntities(query_ref);
        }
        default: {
            assert(false);
        }
    };

    // For some reason, putting __builtin_unreachable here completely breaks
    // everything??

    return 0;
}

TaskGraph::WorkerState TaskGraph::getWork(mwGPU::EntryData **entry_data,
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

    *entry_data = &sorted_nodes_[sharedBlockState.nodeIdx].info.data;
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
    NodeState &cur_node = sorted_nodes_[node_idx];

    uint32_t prev_remaining = cur_node.numRemaining.fetch_sub(num_finished,
        std::memory_order_acq_rel);

    if (prev_remaining == num_finished) {
        uint32_t next_node_idx = node_idx + 1;

        while (true) {
            if (next_node_idx < num_nodes_) {
                uint32_t new_num_invocations =
                    computeNumInvocations(sorted_nodes_[next_node_idx]);

                if (new_num_invocations == 0) {
                    next_node_idx++;
                    continue;
                }

                NodeState &next_node = sorted_nodes_[next_node_idx];
                next_node.curOffset.store(0, std::memory_order_relaxed);
                next_node.numRemaining.store(new_num_invocations,
                                            std::memory_order_relaxed);
                next_node.totalNumInvocations.store(new_num_invocations,
                    std::memory_order_relaxed);
            }

            cur_node_idx_.store(next_node_idx, std::memory_order_release);
            break;
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

    uint64_t chunk_allocator_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(ChunkAllocator));

    total_bytes = chunk_allocator_offset + sizeof(ChunkAllocator);

    uint64_t world_data_offset =
        utils::roundUp(total_bytes, (uint64_t)world_data_alignment);

    uint64_t total_world_bytes =
        (uint64_t)num_world_data_bytes * (uint64_t)num_worlds;

    total_bytes = world_data_offset + total_world_bytes;

    *out_constants = GPUImplConsts {
        .jobSystemAddr = (void *)0ul,
        .stateManagerAddr = (void *)state_mgr_offset,
        .chunkAllocatorAddr = (void *)chunk_allocator_offset,
        .chunkBaseAddr = (void *)0ul,
        .worldDataAddr = (void *)world_data_offset,
        .numWorldDataBytes = num_world_data_bytes,
        .numWorlds = num_worlds,
        .jobGridsOffset = (uint32_t)0,
        .jobListOffset = (uint32_t)0,
        .maxJobsPerGrid = 0,
        .sharedJobTrackerOffset = (uint32_t)0,
        .userJobTrackerOffset = (uint32_t)0,
        .taskGraph = (void *)0ul,
    };

    *job_system_buffer_size = total_bytes;
}
