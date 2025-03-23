#include <madrona/context.hpp>
#include <madrona/checkpoint.hpp>

namespace madrona::CheckpointSystem {

void registerTypes(ECSRegistry &registry)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    uint32_t num_checkpoints = state_mgr->getNumCheckpoints();

    registry.registerSingleton<WorldCheckpoint>(
        sizeof(Checkpoint) * num_checkpoints);
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

    // cur_node = builder.addToGraph<ReadbackNode>({cur_node});

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
