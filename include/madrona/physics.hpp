#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/span.hpp>
#include <madrona/taskgraph_builder.hpp>
#include <madrona/context.hpp>

#include <madrona/broadphase.hpp>
#include <madrona/geo.hpp>

namespace madrona::phys {

struct ExternalForce : math::Vector3 {
    ExternalForce(math::Vector3 v)
        : Vector3(v)
    {}
};

struct ExternalTorque : math::Vector3 {
    ExternalTorque(math::Vector3 v)
        : Vector3(v)
    {}
};

enum class ResponseType : uint32_t {
    Dynamic,
    Kinematic,
    Static,
};

struct Velocity {
    math::Vector3 linear;
    math::Vector3 angular;
};

struct SolverBundleAlias {};

struct RigidBody : Bundle<
    base::ObjectInstance,
    ResponseType,
    broadphase::LeafID,
    Velocity, 
    ExternalForce,
    ExternalTorque,
    SolverBundleAlias
> {};

struct CandidateCollision {
    Loc a;
    Loc b;
    uint32_t aPrim;
    uint32_t bPrim;
};

struct ContactConstraint {
    Loc ref;
    Loc alt;
    math::Vector4 points[4];
    int32_t numPoints;
    math::Vector3 normal;
};

struct JointConstraint {
    enum class Type {
        Fixed,
        Hinge
    };

    struct Fixed {
        math::Quat attachRot1;
        math::Quat attachRot2;
        float separation;
    };

    struct Hinge {
        math::Vector3 a1Local;
        math::Vector3 a2Local;
        math::Vector3 b1Local;
        math::Vector3 b2Local;
    };

    Entity e1;
    Entity e2;
    Type type;

    union {
        Fixed fixed;
        Hinge hinge;
    };

    math::Vector3 r1;
    math::Vector3 r2;
};

struct CollisionEvent {
    Entity a;
    Entity b;
};

struct CollisionEventTemporary : Archetype<CollisionEvent> {};

// Per object state
struct RigidBodyMassData {
    float invMass;
    math::Vector3 invInertiaTensor;
    math::Vector3 toCenterOfMass;
    math::Quat toInteriaFrame;
};

struct RigidBodyFrictionData {
    float muS;
    float muD;
};

struct RigidBodyMetadata {
    RigidBodyMassData mass;
    RigidBodyFrictionData friction;
};

struct CollisionPrimitive {
    enum class Type : uint32_t {
        Sphere = 1 << 0,
        Hull = 1 << 1,
        Plane = 1 << 2,
    };

    struct Sphere {
        float radius;
    };

    struct Hull {
        geo::HalfEdgeMesh halfEdgeMesh;
    };

    struct Plane {};

    Type type;
    union {
        Sphere sphere;
        Plane plane;
        Hull hull;
    };
};

struct ObjectManager {
    CollisionPrimitive *collisionPrimitives;
    math::AABB *primitiveAABBs;

    math::AABB *rigidBodyAABBs;
    uint32_t *rigidBodyPrimitiveOffsets;
    uint32_t *rigidBodyPrimitiveCounts;
    RigidBodyMetadata *metadata;
};

struct ObjectData {
    ObjectManager *mgr;
};

namespace PhysicsSystem {
    enum class Solver : uint32_t {
        XPBD,
        TGS,
    };

    void init(Context &ctx,
              ObjectManager *obj_mgr,
              float delta_t,
              CountT num_substeps,
              math::Vector3 gravity,
              CountT max_dynamic_objects,
              Solver solver = Solver::XPBD);

    void reset(Context &ctx);
    broadphase::LeafID registerEntity(Context &ctx,
                                      Entity e,
                                      base::ObjectID obj_id);

    template <typename Fn>
    void findEntitiesWithinAABB(Context &ctx,
                                       math::AABB aabb,
                                       Fn &&fn);

    bool checkEntityAABBOverlap(Context &ctx,
                                       math::AABB aabb,
                                       Entity e);

    Entity makeFixedJoint(Context &ctx,
                          Entity e1, Entity e2,
                          math::Quat attach_rot1, math::Quat attach_rot2,
                          math::Vector3 r1, math::Vector3 r2,
                          float separation);

    Entity makeHingeJoint(Context &ctx,
                          Entity e1, Entity e2,
                          math::Vector3 a1_local, math::Vector3 a2_local,
                          math::Vector3 b1_local, math::Vector3 b2_local,
                          math::Vector3 r1, math::Vector3 r2);


    void registerTypes(ECSRegistry &registry,
                       Solver solver = Solver::XPBD);

    TaskGraphNodeID setupBroadphaseTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    TaskGraphNodeID setupPhysicsStepTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps,
        CountT num_substeps,
        Solver solver = Solver::XPBD);

    TaskGraphNodeID setupCleanupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    // Use the below two functions if you just want to use the broadphase without
    // the rest of the physics system

    TaskGraphNodeID setupStandaloneBroadphaseOverlapTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    TaskGraphNodeID setupStandaloneBroadphaseCleanupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

};

}

#include "physics.inl"
