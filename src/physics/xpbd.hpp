#pragma once

#include "physics_impl.hpp"

namespace madrona::phys::xpbd {

void registerTypes(ECSRegistry &registry);

TaskGraphNodeID setupXPBDSolverTasks(
    TaskGraphBuilder &builder,
    TaskGraphNodeID broadphase,
    CountT num_substeps);

}
