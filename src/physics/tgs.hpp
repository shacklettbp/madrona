#pragma once

#include "physics_impl.hpp"

namespace madrona::phys::tgs {

void registerTypes(ECSRegistry &registry);

TaskGraphNodeID setupTGSSolverTasks(
    TaskGraphBuilder &builder,
    TaskGraphNodeID broadphase,
    CountT num_substeps);

}
