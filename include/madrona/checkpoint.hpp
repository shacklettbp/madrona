#pragma once

#include <madrona/ecs.hpp>
#include <madrona/registry.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona {

struct Checkpoint {
    MemoryRange data;
    uint32_t numBytes;
};

struct WorldCheckpoint : VarComponent<Checkpoint> {};

namespace CheckpointSystem {
void registerTypes(ECSRegistry &, uint32_t control_export_id);

TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);
}
    
}
