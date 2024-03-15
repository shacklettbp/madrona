#pragma once

#include "physics_impl.hpp"

namespace madrona::phys::tgs {

void registerTypes(ECSRegistry &registry);

void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id);

void init(Context &ctx);

TaskGraphNodeID setupTGSSolverTasks(
    TaskGraphBuilder &builder,
    TaskGraphNodeID broadphase,
    CountT num_substeps);

}
