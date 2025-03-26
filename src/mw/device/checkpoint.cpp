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

extern "C" void queryCheckpointInfo(uint32_t num_queries, void *readback)
{
#if 0
    using namespace madrona;
    using namespace madrona::CheckpointSystem;

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries)
        return;

    uint32_t *readback_u32 = (uint32_t *)readback;

    uint32_t total_num_ckpts = *readback_u32;
    uint32_t *traj_avails = readback_u32 + 1;
    std::pair<uint32_t, uint32_t> *req_world_ids = 
        (std::pair<uint32_t, uint32_t> *)(readback_u32 + num_queries);

    



    uint32_t *traj_avail = readback_u32 + tid;
    uint32_t *req_world_id = readback_u32 + num_queries + tid * 2;
    uint32_t *req_num_steps = req_world_id + 1;
    uint32_t *ckpt_size = readback_u32 + num_queries + num_queries * 2 + tid;
    void **ckpt_ptr = (void **)(
            readback_u32 + 
            num_queries + 
            num_queries * 2 + 
            num_queries) + tid;

    // First check if there are enough checkpoints.
    StateManager *state_mgr = mwGPU::getStateManager();
    uint32_t num_checkpoints = state_mgr->getNumCheckpoints();

    // Checkpoint *ckpts = ctx.singleton<WorldCheckpoint>().ptr();
    Checkpoint *ckpts = state_mgr->getSingleton<WorldCheckpoint>(
            *req_world_id).ptr();
    uint32_t most_recent_write = state_mgr->getSingleton<WorldMostRecentWrite>(
            *req_world_id);

    for (uint32_t i = 0; i < num_checkpoints; ++i) {
        
    }
#endif
}
