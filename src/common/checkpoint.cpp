#include <stdio.h>
#include <madrona/checkpoint.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona::CheckpointSystem {

void registerTypes(ECSRegistry &registry)
{
    // This is not implemented properly yet for the CPU
    registry.registerSingleton<WorldCheckpoint>(
        sizeof(Checkpoint));

    printf("WARNING: CPU checkpointing isn't implemented yet\n");
}

void init(Context &ctx)
{
    (void)ctx;

    /* Doesn't do anything on the CPU yet */
    printf("WARNING: CPU checkpointing isn't implemented yet\n");
}

TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps)
{
    (void)builder;
    (void)deps;

    return TaskGraphNodeID {};
}

Checkpoint requestCheckpoint(Context &ctx,
                             uint32_t checkpoint_idx,
                             uint32_t num_bytes)
{
    (void)ctx;
    (void)checkpoint_idx;
    (void)num_bytes;

    printf("WARNING: CPU checkpointing isn't implemented yet\n");

    return {};
}

Checkpoint getCheckpoint(Context &ctx,
                         uint32_t checkpoint_idx)
{
    (void)ctx;
    (void)checkpoint_idx;

    return {};
}

void * getCheckpointPtr(Context &ctx,
                        Checkpoint ckpt)
{
    return nullptr;
}

}
