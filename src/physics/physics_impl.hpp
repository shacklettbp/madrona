#pragma once

#include <madrona/physics.hpp>

namespace madrona::phys {

struct SolverData {
    Contact *contacts;
    AtomicCount numContacts;

    JointConstraint *jointConstraints;
    AtomicCount numJointConstraints;

    CountT maxContacts;
    float deltaT;
    float h;
    math::Vector3 g;
    float gMagnitude;
    float restitutionThreshold;

    inline SolverData(CountT max_contacts_per_step,
                      CountT max_joint_constraints,
                      float delta_t,
                      CountT num_substeps,
                      math::Vector3 gravity);
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
