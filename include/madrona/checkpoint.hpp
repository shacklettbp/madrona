#pragma once

#include <madrona/ecs.hpp>
#include <madrona/registry.hpp>
#include <madrona/taskgraph.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona {

struct Checkpoint {
    MemoryRange data;
    uint32_t numBytes;
};

struct WorldMostRecentWrite {
    uint32_t mostRecentCkpt;
};

struct WorldCheckpoint : VarComponent<Checkpoint> {};

namespace CheckpointSystem {
void registerTypes(ECSRegistry &);

TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

// Request bytes for this checkpoint
Checkpoint requestCheckpoint(Context &ctx,
                             uint32_t checkpoint_idx,
                             uint32_t num_bytes);

Checkpoint getCheckpoint(Context &ctx,
                         uint32_t checkpoint_idx);

void * getCheckpointPtr(Context &ctx,
                        Checkpoint ckpt);
}
    
}
