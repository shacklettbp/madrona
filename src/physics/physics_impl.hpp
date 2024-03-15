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

struct Contact : Archetype<ContactConstraint> {};
struct CandidateTemporary : Archetype<CandidateCollision> {};

struct Joint : Archetype<JointConstraint> {};

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

struct Cols {
    static constexpr inline CountT Position = 2;
    static constexpr inline CountT Rotation = 3;
    static constexpr inline CountT Scale = 4;
    static constexpr inline CountT Velocity = 5;
    static constexpr inline CountT ObjectID = 6;
    static constexpr inline CountT ResponseType = 7;
    static constexpr inline CountT SubstepPrevState = 8;
    static constexpr inline CountT PreSolvePositional = 9;
    static constexpr inline CountT PreSolveVelocity = 10;
    static constexpr inline CountT ExternalForce = 11;
    static constexpr inline CountT ExternalTorque = 12;
    static constexpr inline CountT LeafID = 13;

    static constexpr inline CountT CandidateCollision = 2;
};

}
