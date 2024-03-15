#pragma once

#include "physics_impl.hpp"

namespace madrona::phys::xpbd {

void registerTypes(ECSRegistry &registry);

void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id);

void init(Context &ctx);

TaskGraphNodeID setupXPBDSolverTasks(
    TaskGraphBuilder &builder,
    TaskGraphNodeID broadphase,
    CountT num_substeps);

}
