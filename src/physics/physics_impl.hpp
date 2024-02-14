#pragma once

#include <madrona/physics.hpp>

namespace madrona::phys {

struct SolverData {
    float deltaT;
    float h;
    math::Vector3 g;
    float gMagnitude;
    float restitutionThreshold;

    Query<JointConstraint> jointQuery;
    Query<ContactConstraint> contactQuery;
};

namespace broadphase {

TaskGraphNodeID setupBVHTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

TaskGraphNodeID setupPreIntegrationTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

TaskGraphNodeID setupPostIntegrationTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

}

namespace narrowphase {

TaskGraphNodeID setupTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

}

}
