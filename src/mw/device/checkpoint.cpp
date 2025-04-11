#include <algorithm>
#include <madrona/context.hpp>
#include <madrona/checkpoint.hpp>

namespace madrona::CheckpointSystem {

void registerTypes(ECSRegistry &registry)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    uint32_t num_checkpoints = state_mgr->getNumCheckpoints();

    registry.registerSingleton<WorldCheckpoint>(
        sizeof(Checkpoint) * num_checkpoints);

    registry.registerSingleton<WorldMostRecentWrite>();
}

void init(Context &ctx)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    uint32_t num_checkpoints = state_mgr->getNumCheckpoints();

    Checkpoint *ckpts = ctx.singleton<WorldCheckpoint>().ptr();
    for (uint32_t i = 0; i < num_checkpoints; ++i) {
        ckpts[i].data = MemoryRange::none();
    }
}

struct ReadbackNode : NodeBase {
    ReadbackNode() = default;

    void updateReadback(int32_t checkpoint_idx)
    {
        StateManager *state_mgr = mwGPU::getStateManager();
        assert(checkpoint_idx < state_mgr->getNumCheckpoints());

        // This just updates the size readback
        state_mgr->updateCheckpointReadback(checkpoint_idx);
    }

    static TaskGraph::NodeID addToGraph(
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> deps)
    {
        StateManager *state_mgr = mwGPU::getStateManager();

        auto data_id = builder.constructNodeData<ReadbackNode>();
        auto &ref = builder.getDataRef(data_id);

        return builder.addNodeFn<
            &ReadbackNode::updateReadback>(data_id, deps,
                    Optional<TaskGraph::NodeID>::none(),
                    state_mgr->getNumCheckpoints(),
                    1);
    }
};

TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    uint32_t num_checkpoints = state_mgr->getNumCheckpoints();

    TaskGraphNodeID cur_node;
    if (num_checkpoints) {
        cur_node = SortNodeBase::addToGraphMemoryRange(
                builder, deps, 0, true);
    }

    for (uint32_t i = 1; i < num_checkpoints; ++i) {
        cur_node = SortNodeBase::addToGraphMemoryRange(
                builder, {cur_node}, i, true);
    }

    return cur_node;
}

Checkpoint requestCheckpoint(Context &ctx,
                             uint32_t checkpoint_idx,
                             uint32_t num_bytes)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    uint32_t num_checkpoints = state_mgr->getNumCheckpoints();

    assert(checkpoint_idx < num_checkpoints);

    Checkpoint *ckpts = ctx.singleton<WorldCheckpoint>().ptr();

    Checkpoint &ckpt = ckpts[checkpoint_idx];

    uint32_t capacity = ckpt.data.numElements *
                        StateManager::getCheckpointElemSize();

    if (capacity < num_bytes) {
        uint32_t num_elems = (
                num_bytes + StateManager::getCheckpointElemSize() - 1) / 
                StateManager::getCheckpointElemSize();
        ckpt.data = state_mgr->allocMemoryRange(checkpoint_idx, num_elems);
        ckpt.numBytes = num_bytes;
    }

    ctx.singleton<WorldMostRecentWrite>().mostRecentCkpt = checkpoint_idx;

    return ckpt;
}

Checkpoint getCheckpoint(Context &ctx,
                         uint32_t checkpoint_idx)
{
    Checkpoint *ckpts = ctx.singleton<WorldCheckpoint>().ptr();
    return ckpts[checkpoint_idx];
}

void * getCheckpointPtr(Context &ctx,
                        Checkpoint ckpt)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    return state_mgr->memoryRangePointer(ckpt.data);
}

}

extern "C" __global__ void queryCheckpointInfo(uint32_t num_queries, void *readback)
{
    using namespace madrona;
    using namespace madrona::CheckpointSystem;

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries)
        return;

    uint64_t *readback_u64 = (uint64_t *)readback;

    struct TrajInfo {
        uint64_t worldID;
        uint64_t numSteps;
        uint64_t offset;
        uint64_t stepOffset;
    };

    uint64_t total_num_ckpts = *readback_u64;
    uint64_t *traj_avails = readback_u64 + 1;
    TrajInfo *traj_infos = 
        (TrajInfo *)(traj_avails + num_queries);
    uint64_t *ckpt_sizes = (uint64_t *)(traj_infos + num_queries);
    void **ckpt_ptrs = (void **)(ckpt_sizes + total_num_ckpts);



    // All data for this specific thread
    uint64_t *curr_traj_avails = traj_avails + tid;
    TrajInfo *curr_traj_info = traj_infos + tid;
    uint64_t *curr_ckpt_sizes = ckpt_sizes + curr_traj_info->offset;
    void **curr_ckpt_ptrs = ckpt_ptrs + curr_traj_info->offset;

    // First check if there are enough checkpoints.
    StateManager *state_mgr = mwGPU::getStateManager();
    uint32_t num_checkpoints = state_mgr->getNumCheckpoints();

    // Checkpoint *ckpts = ctx.singleton<WorldCheckpoint>().ptr();
    Checkpoint *ckpts = state_mgr->getSingleton<WorldCheckpoint>(
        WorldID { curr_traj_info->worldID }).ptr();
    uint32_t most_recent_write = state_mgr->getSingleton<WorldMostRecentWrite>(
        WorldID { curr_traj_info->worldID }).mostRecentCkpt;

    *curr_traj_avails = 1;

    uint64_t num_acquired_steps = 0;

    uint32_t max_steps = std::min(state_mgr->getNumCheckpoints(),
            (uint32_t)curr_traj_info->numSteps);
    for (uint32_t i = 0; i < max_steps; ++i) {
        uint32_t ckpt_idx = (num_checkpoints + most_recent_write - i) %
            num_checkpoints;

        if (ckpts[ckpt_idx].data == MemoryRange::none()) {
            // This trajectory isn't yet available
            *curr_traj_avails = 0;
            break;
        } else {
            void *ptr = state_mgr->memoryRangePointer(
                    ckpts[ckpt_idx].data);
            uint32_t num_bytes = ckpts[ckpt_idx].numBytes;

#if 0
            printf("ckpt num bytes = %u -> %p\n", num_bytes, 
                    curr_ckpt_sizes + curr_traj_info->numSteps - 1 - i);
#endif

            // We are going from last to first
            curr_ckpt_sizes[max_steps - 1 - i] = num_bytes;
            curr_ckpt_ptrs[max_steps - 1 - i] = ptr;

            num_acquired_steps++;
        }
    }

    *curr_traj_avails = num_acquired_steps;
}
