#pragma once

#include <madrona/render/ecs.hpp>
#include <madrona/components.hpp>
#include <madrona/memory_range.hpp>
#include <madrona/taskgraph_builder.hpp>
#include <madrona/physics.hpp>

namespace madrona::phys::cv {

#if 0
enum class DofType {
    // The number of unique degrees of freedom (SE3). Maximum number of DOFs is 6
    FreeBody = 6,
    Hinge = 1,
    Ball = 3,
    FixedBody = 0,

    // When we add other types of physics DOF objects, we will encode
    // the number of degrees of freedom they all have here.
};
#endif

enum class DofType {
    FreeBody,
    Hinge,
    Ball,
    FixedBody,
    Slider,
    None
};

inline uint32_t getNumDofs(DofType type);

static constexpr uint32_t kMaxPositionCoords = 7;
static constexpr uint32_t kMaxVelocityCoords = 6;

struct DofObjectPosition {
    // There are multiple ways of interpreting the values in this struct.
    // If this is a free body (i.e., 6 degrees of freedom),
    // - q[0:3] are the position
    // - q[3:7] are the quaternion
    //
    // If this is a hinge joint (i.e., 1 degree of freedom),
    // - q[0] is the angle
    //
    // If this is a ball joint (i.e. 3 degrees of freedom),
    // - q[0:4] are the quaternion
    //
    // TODO: Other types of DOF objects.
    float q[kMaxPositionCoords];
};

struct DofObjectExtForce {
    float force[kMaxPositionCoords];
};

struct DofObjectVelocity {
    float qv[kMaxVelocityCoords];
};

struct DofObjectAcceleration {
    float dqv[kMaxVelocityCoords];
};

struct DofObjectNumDofs {
    DofType type;
    uint32_t numDofs;
};

struct ContactTmpState {
    float mu;
    // Contact coordinate system (first col is normal, two tangents)
    math::Mat3x3 C;
};

// During solve, we will store the individual contact point info
struct ContactPointInfo {
    uint32_t parentIdx;
    uint32_t subIdx;
};

struct SpatialVector {
    math::Vector3 linear;
    math::Vector3 angular;

    static SpatialVector fromVec(const float* v) {
        return { {v[0], v[1], v[2]}, {v[3], v[4], v[5]} };
    }

    float operator[](const CountT i) const {
        return i < 3 ? linear[i] : angular[i - 3];
    }
    float& operator[](const CountT i) {
        return i < 3 ? linear[i] : angular[i - 3];
    }

    SpatialVector& operator+=(const SpatialVector& rhs) {
        linear += rhs.linear;
        angular += rhs.angular;
        return *this;
    }

    SpatialVector& operator-=(const SpatialVector& rhs) {
        linear -= rhs.linear;
        angular -= rhs.angular;
        return *this;
    }

    SpatialVector& operator=(const SpatialVector& rhs) {
        linear = rhs.linear;
        angular = rhs.angular;
        return *this;
    }

    SpatialVector cross(const SpatialVector& rhs) const {
        return {
            angular.cross(rhs.linear) + linear.cross(rhs.angular),
            angular.cross(rhs.angular)
        };
    }

    SpatialVector crossStar(const SpatialVector& rhs) const {
        return {
            angular.cross(rhs.linear),
            angular.cross(rhs.angular) + linear.cross(rhs.linear)
        };
    }
};


// This represents the spatial inertia in Plücker coordinates
// [m * 1_{3x3};  -m * r^x
//  m * r^x;      I_world - m * r^x r^x ]
struct InertiaTensor {
    // The spatial inertia tensor is parameterized by 10 values:
    float mass;
    math::Vector3 mCom; // mass times [vector from Plücker origin to COM]

    // The left block of the spatial inertia matrix is symmetric so
    // 6 values are required to parameterize the first block
    // (I_world - m * r^x * r^x).
    // The values are ordered as:
    // [ 0 3 4
    //     1 5
    //       2 ]
    float spatial_inertia[6];

    // Helper function to add two inertia tensors together
    InertiaTensor& operator+=(const InertiaTensor& rhs) {
        mass += rhs.mass;
        mCom += rhs.mCom;
        for (int i = 0; i < 6; i++) {
            spatial_inertia[i] += rhs.spatial_inertia[i];
        }
        return *this;
    }

    // Multiply with vector [v] of length 6, storing the result in [out]
    void multiply(const float* v, float* out) const {
        math::Vector3 v_trans = {v[0], v[1], v[2]};
        math::Vector3 v_rot = {v[3], v[4], v[5]};
        math::Vector3 out_trans = mass * v_trans - mCom.cross(v_rot);
        math::Vector3 out_rot = mCom.cross(v_trans);
        out_rot[0] += spatial_inertia[0] * v_rot[0] + spatial_inertia[3] * v_rot[1] + spatial_inertia[4] * v_rot[2];
        out_rot[1] += spatial_inertia[3] * v_rot[0] + spatial_inertia[1] * v_rot[1] + spatial_inertia[5] * v_rot[2];
        out_rot[2] += spatial_inertia[4] * v_rot[0] + spatial_inertia[5] * v_rot[1] + spatial_inertia[2] * v_rot[2];
        out[0] = out_trans.x;
        out[1] = out_trans.y;
        out[2] = out_trans.z;
        out[3] = out_rot.x;
        out[4] = out_rot.y;
        out[5] = out_rot.z;
    }

    SpatialVector multiply(const SpatialVector& v) const {
        SpatialVector out;
        out.linear = mass * v.linear - mCom.cross(v.angular);
        out.angular = mCom.cross(v.linear);
        out.angular[0] += spatial_inertia[0] * v.angular[0]
        + spatial_inertia[3] * v.angular[1] + spatial_inertia[4] * v.angular[2];
        out.angular[1] += spatial_inertia[3] * v.angular[0]
        + spatial_inertia[1] * v.angular[1] + spatial_inertia[5] * v.angular[2];
        out.angular[2] += spatial_inertia[4] * v.angular[0]
        + spatial_inertia[5] * v.angular[1] + spatial_inertia[2] * v.angular[2];
        return out;
    }
};

// This is the phi linear operator which appears in Featherstone's Composite-
// Rigid-Body Algorithm. phi maps from our generalized velocities to the
// velocities in Plücker coordinates.
struct Phi {
    // Phi is parameterized by at most 6 values (7 for quaternion).
    //
    // For the free body, it just depends on the center of mass of the body.
    //
    // For the hinge, it depends on the normalized angular velocity vector
    // (first 3 values) of the hinge and the position of the joint
    // (not the body - last 3 values).
    //
    // For the ball joint, the position of the joint and the rotation of the
    // parent (quaternion)
    float v[7];
};

// Just some space to store temporary per-entity data.
struct DofObjectTmpState {
    // World-space position of body COM
    math::Vector3 comPos;

    // World-space orientation of body
    math::Quat composedRot;

    // Position of the rotation point. For free body, it's just the COM.
    // For the hinge, it's the position of the joint.
    math::Vector3 anchorPos;

    // Compressed values needed to compute Phi
    Phi phi;

    // Contains storage for both complete form of Phi and Phi_dot
    // MemoryRange/*<PhiUnit>*/ phiFull;
    // Offset in the full memory range for the entire body group
    uint32_t phiFullOffset;

    // Memory range containing everything.
    MemoryRange dynData;

    // The spatial inertia tensor in Plücker coordinates
    // Hold the combined inertia of subtree after combineSpatialInertia
    InertiaTensor spatialInertia;

    // Velocity, Acceleration, Force in Plücker coordinates (used for RNE)
    SpatialVector sVel;
    SpatialVector sAcc;
    SpatialVector sForce;

    // DOF offset in the BodyGroup
    uint32_t dofOffset;

    uint32_t numCollisionObjs;
    uint32_t collisionObjOffset;

    uint32_t numVisualObjs;
    uint32_t visualObjOffset;

    ResponseType responseType;

    float scratch[4];

    float * getPhiFull(Context &ctx);
};

struct HingeLimit {
    float lower;
    float upper;

    // dC/dq
    inline float dConstraintViolation(float q);
    inline float constraintViolation(float q);
};

struct SliderLimit {
    float lower;
    float upper;

    // dC/dq
    inline float dConstraintViolation(float q);
    inline float constraintViolation(float q);
};

struct DofObjectLimit {
    enum class Type {
        Hinge,
        Slider,
        None
    };

    // This is going to depend on the dof type
    Type type;

    union {
        HingeLimit hinge;
        SliderLimit slider;
    };

    // Offset in the equality jacobian
    uint32_t rowOffset;
};

struct DofObjectHierarchyDesc {
    Entity parent;

    // Relative position of the joint to the parent's COM.
    math::Vector3 relPositionParent;

    // Relative position of the child's COM to the joint.
    math::Vector3 relPositionLocal;

    // Rotation applied to the child
    math::Quat parentToChildRot;

    // Extra data:
    // For hinge, this is the hinge rotation axis.
    // For ball, this is the vector perpendicular to the plane of allowed
    //           angular velocities
    math::Vector3 axis;

    bool leaf;

    // Index in the body group hierarchy
    int32_t index;
    // Index of the parent in the body group hierarchy. -1 if root.
    int32_t parentIndex;

    Entity bodyGroup;
};

struct DofObjectInertial {
    float mass;
    math::Diag3x3 inertia;

    // Estimated inverse weight for the body
    float approxInvMassTrans;
    float approxInvMassRot;
    // Estimated inverse weight for each DOF
    float approxInvMassDof[kMaxPositionCoords];
};

struct DofObjectFriction {
    float muS;
};

struct DofObjectArchetype : public Archetype<
    DofObjectPosition,
    DofObjectVelocity,
    DofObjectAcceleration, // Δv, used for integration
    DofObjectExtForce,

    DofObjectTmpState,

    DofObjectHierarchyDesc,

    DofObjectInertial,
    DofObjectFriction,

    DofObjectLimit,

    DofObjectNumDofs
> {};

struct LinkParentDofObject {
    Entity parentDofObject;

    // Offset in `BodyObjectData`
    uint32_t mrOffset;
};

// This is the archetype for a signle body's link for collision detection
struct LinkCollider : public Archetype<
    RigidBody,
    LinkParentDofObject
> {};

struct LinkVisual : public Archetype<
    base::ObjectInstance,
    render::Renderable,
    LinkParentDofObject
> {};

struct BodyObjectData {
    // Either a renderable, or rigid body entity
    Entity proxy;
    math::Vector3 offset;
    math::Quat rotation;
    math::Diag3x3 scale;
};

struct BodyGroupHierarchy {
    // This includes the free body too which will be at index 0.
    uint32_t numBodies;
    uint32_t bodyCounter;

    MemoryRange mrBodies;

    uint32_t collisionObjsCounter;
    uint32_t visualObjsCounter;

    MemoryRange mrCollisionVisual;

    // Total number of DOFs in the body group
    uint32_t numDofs;

    // Memory range contains data for all data that is dynamically sized.
    MemoryRange dynData;

    uint32_t numEqualityRows;

    // "Expanded" parent array for kinematic tree (chain of 1-DOF joints)
    //   used for factorization. See Featherstone pg. 114
    uint32_t expandedParentOffset;

    // Center of mass of the body group
    math::Vector3 comPos;

    // Mass matrix (num_dof x num_dof) of the body group
    uint32_t massMatrixOffset;

    // LTDL factorization of matrix
    uint32_t massMatrixLTDLOffset;

    // Bias forces (num_dof) of the body group, gets replaced by the
    //  unconstrained acceleration after computeFreeAcceleration
    uint32_t biasOffset;

    // Offset to buffer containing prefix sum of the DOFs of the bodies.
    uint32_t dofPrefixSumOffset;

    // Temporary index used during stacking
    uint32_t tmpIdx0;
    uint32_t tmpIdx1;

    // Sum of diagonal elements of mass matrix
    float inertiaSum;

    // Some scratch space for various calculations
    float scratch[36];



    // Some helpers for the memory range
    float * getMassMatrix(Context &ctx);
    float * getMassMatrixLTDL(Context &ctx);
    float * getBias(Context &ctx);
    int32_t * getExpandedParent(Context &ctx);
    Entity * bodies(Context &ctx);
    uint32_t * getDofPrefixSum(Context &ctx);
    BodyObjectData *getCollisionData(Context &ctx);
    BodyObjectData *getVisualData(Context &ctx);
};

struct BodyGroup : public Archetype<
    BodyGroupHierarchy
> {};

struct BodyDesc {
    DofType type;
    math::Vector3 initialPos;
    math::Quat initialRot;
    ResponseType responseType;
    uint32_t numCollisionObjs;
    uint32_t numVisualObjs;
    float mass;
    math::Diag3x3 inertia;
    float muS;
};

struct CollisionDesc {
    uint32_t objID;
    math::Vector3 offset;
    math::Quat rotation;
    math::Diag3x3 scale;

    uint32_t linkIdx;
    // Index of the collider within the body
    uint32_t subIndex;
};

using VisualDesc = CollisionDesc;

Entity makeBodyGroup(Context &ctx, uint32_t num_bodies);
Entity makeBody(Context &ctx, Entity body_grp, BodyDesc desc);

void attachCollision(
        Context &ctx,
        Entity body_grp,
        Entity body,
        uint32_t idx,
        CollisionDesc desc);

void attachVisual(
        Context &ctx,
        Entity body_grp,
        Entity body,
        uint32_t idx,
        VisualDesc desc);

void setRoot(
        Context &ctx,
        Entity body_grp,
        Entity body);

struct JointHinge {
    // In parent's basis
    math::Vector3 relPositionParent;

    // In child's basis
    math::Vector3 relPositionChild;

    // Rotation applied to child's vectors relative to
    // parent's coordinate system.
    math::Quat relParentRotation;

    // In child's basis
    math::Vector3 hingeAxis;
};

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent,
        Entity child,
        JointHinge hinge_info);

struct JointBall {
    // In parent's basis
    math::Vector3 relPositionParent;

    // In child's basis
    math::Vector3 relPositionChild;

    // Rotation applied to child's vectors relative to
    // parent's coordinate system.
    math::Quat relParentRotation;
};

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent,
        Entity child,
        JointBall ball_info);

struct JointSlider {
    // In the parent's coordinate basis
    math::Vector3 relPositionParent;

    // In the child's coordinate basis
    math::Vector3 relPositionChild;

    // This is in the child's coordinate basis
    math::Vector3 slideVector;

    // Rotation applied to child's vectors relative to
    // parent's coordinate system.
    math::Quat relParentRotation;
};

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent,
        Entity child,
        JointSlider slider_info);

void attachLimit(
        Context &ctx,
        Entity body_grp,
        Entity body,
        HingeLimit limit);

void attachLimit(
        Context &ctx,
        Entity body_grp,
        Entity body,
        SliderLimit limit);

void registerTypes(ECSRegistry &registry);
void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id);
void init(Context &ctx, CVXSolve *cvx_solve);

TaskGraphNodeID setupCVInitTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps);

void initializeHierarchies(Context &ctx);

struct JointConnection {
    // These are body indices
    uint32_t parentIdx;
    uint32_t childIdx;

    // This determines which Joint struct to use
    DofType type;

    union {
        JointHinge hinge;
        JointBall ball;
        // ...
    };
};

// For loading pre-configured models
struct ModelConfig {
    // Assume that the first one is the root
    uint32_t numBodies;
    uint32_t bodiesOffset;

    uint32_t numConnections;
    uint32_t connectionsOffset;

    uint32_t numColliders;
    uint32_t collidersOffset;

    uint32_t numVisuals;
    uint32_t visualsOffset;
};

// This is the data for all models in that could possibly be loaded.
struct ModelData {
    BodyDesc *bodies;
    JointConnection *connections;
    CollisionDesc *colliders;
    VisualDesc *visuals;
};

// This returns the body group entity
Entity loadModel(Context &ctx,
                 ModelConfig cfg,
                 ModelData model_data);

}

#include "cvphysics.inl"
