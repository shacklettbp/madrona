#pragma once

#include <madrona/physics.hpp>

namespace madrona::phys {

struct SolverData {
    Contact *contacts;
    std::atomic<CountT> numContacts;

    JointConstraint *jointConstraints;
    std::atomic<CountT> numJointConstraints;

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

TaskGraph::NodeID setupPreIntegrationTasks(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps);

TaskGraph::NodeID setupPostIntegrationTasks(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps);

}

namespace narrowphase {

TaskGraph::NodeID setupTasks(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps);

}

}
