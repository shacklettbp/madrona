#include <madrona/checkpoint.hpp>

namespace madrona::CheckpointSystem {

void registerTypes(ECSRegistry &registry, uint32_t control_export_id)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    uint32_t num_checkpoints = state_mgr->getNumCheckpoints();

    registry.registerSingleton<WorldCheckpoint>(
        sizeof(Checkpoint) * num_checkpoints);
    registry.exportSingleton<Checkpoint>((int32_t)control_export_id);
}

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

}
