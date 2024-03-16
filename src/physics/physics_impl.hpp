#pragma once

#include <madrona/physics.hpp>

namespace madrona::phys {

struct PhysicsSystemState {
    float deltaT;
    float h;
    math::Vector3 g;
    float gMagnitude;
    float restitutionThreshold;
    uint32_t contactArchetypeID;
    uint32_t jointArchetypeID;
};

struct CandidateTemporary : Archetype<CandidateCollision> {};

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

namespace RGDCols {
    constexpr inline CountT Position = 2;
    constexpr inline CountT Rotation = 3;
    constexpr inline CountT Scale = 4;
    constexpr inline CountT ObjectID = 5;
    constexpr inline CountT ResponseType = 6;
    constexpr inline CountT LeafID = 7;
    constexpr inline CountT Velocity = 8;
    constexpr inline CountT ExternalForce = 9;
    constexpr inline CountT ExternalTorque = 10;
    constexpr inline CountT SolverBase = 11;

    constexpr inline CountT CandidateCollision = 2;
    constexpr inline CountT ContactConstraint = 2;
    constexpr inline CountT JointConstraint = 2;
};

}
