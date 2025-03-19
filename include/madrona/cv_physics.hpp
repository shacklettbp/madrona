#pragma once

#include <madrona/string.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/components.hpp>
#include <madrona/memory_range.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona::phys::cv {

// The type of body / joint
enum class DofType : uint8_t {
    FreeBody,
    Hinge,
    Ball,
    FixedBody,
    Slider,
    None
};

struct HingeLimit {
    float lower;
    float upper;

    // dC/dq
    inline bool isActive(float q);
    inline float dConstraintViolation(float q);
    inline float constraintViolation(float q);
};

struct SliderLimit {
    float lower;
    float upper;

    // dC/dq
    inline bool isActive(float q);
    inline float dConstraintViolation(float q);
    inline float constraintViolation(float q);
};

// Joint limit
struct BodyLimitConstraint {
    enum class Type {
        Hinge,
        Slider,
        None
    };

    // This is going to depend on the dof type
    Type type;
    uint32_t bodyIdx;

    union {
        HingeLimit hinge;
        SliderLimit slider;
    };
};

// Data offsets for each body
struct BodyOffsets {
    uint8_t posOffset;
    uint8_t velOffset;
    uint8_t parent;    // 0xFF is invalid
    uint8_t parentWithDof; // Index of first parent with at least one DOF
    DofType dofType;

    uint8_t numDofs; // Dimension of velocity
    uint8_t eqOffset;
    uint8_t numEqs;

    static inline uint32_t getDofTypeDim(DofType type, bool is_pos = false);
};

// Description of the body in the hierarchy
struct BodyHierarchy {
    math::Vector3 axis;
    math::Vector3 relPositionParent;
    math::Vector3 relPositionLocal;
    math::Quat parentToChildRot;
};

struct BodyInertial {
    float mass;
    math::Diag3x3 inertia;

    // Estimated inverse weight for the body
    float approxInvMassTrans;
    float approxInvMassRot;

    // Estimated inverse weight for each DOF
    float approxInvMassDof[7];
};

struct BodyObjectData {
    enum class Type {
        Render,
        Collider,
        RenderCollider,
        None
    };

    // Either a renderable, or rigid body entity
    Entity proxy;

    Type type;

    math::Vector3 offset;
    math::Quat rotation;
    math::Diag3x3 scale;

    // This is only relevant for the collision objects
    Entity optionalRender;
};

struct BodyTransform {
    math::Vector3 com;
    math::Quat composedRot;
};

struct BodyPhi {
    float phi[7];
    float phiDot[7];
};

struct BodyGroupProperties {
    float globalScale;

    // qDim is the total amount of floats in the position vector (includes fixed
    // body Q values). numFixedQ is just the number of fixed body q values
    uint32_t qDim;
    uint32_t numFixedQ;

    uint32_t qvDim;
    uint32_t numBodies;
    uint32_t numEq;
    uint32_t numObjData;

    math::Vector3 comPos;
    // Sum of diagonals of mass matrix
    float inertiaSum;

    uint32_t numHashes;

    struct {
        uint32_t bodyCounter;

        // Index of the group in the world
        uint32_t grpIndex;

        // Offset in the global velocity vector
        uint32_t qvOffset;

        // Offset (row) in the equality jacobian
        uint32_t eqOffset;
    } tmp;
};

struct SpatialVector {
    math::Vector3 linear;
    math::Vector3 angular;

    static inline SpatialVector fromVec(const float* v);
    inline float operator[](const CountT i) const;
    inline float & operator[](const CountT i);
    inline SpatialVector & operator+=(const SpatialVector &rhs);
    inline SpatialVector & operator-=(const SpatialVector &rhs);
    inline SpatialVector crossMotion(const SpatialVector &rhs) const;
    inline SpatialVector crossStar(const SpatialVector &rhs) const;
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
    inline InertiaTensor & operator+=(const InertiaTensor &rhs);

    // Multiply with vector [v] of length 6, storing the result in [out]
    inline void multiply(const float* v, float* out) const;
    inline SpatialVector multiply(const SpatialVector &v) const;
};


struct BodySpatialVectors {
    SpatialVector sVel;
    SpatialVector sAcc;
    SpatialVector sForce;
    // Spatial inertia before CRB, composite spatial inertia after CRB
    InertiaTensor spatialInertia;
};

struct BodyNameHash {
    uint32_t bodyIdx;
    uint32_t hash;
};

// This contains data for all the bodies within the body group.
struct BodyGroupMemory {
    // This memory range contains data which must persist from frame to frame.
    // This includes:
    // - position vector (q)
    // - velocity vector (qv)
    // - acceleration vector (dqv)
    // - force (f)
    // - friction coefficients (mus)
    // - joint limits
    // - inertias
    // - expanded parent
    // - object data
    // - offsets
    // - name hashes
    MemoryRange qVectors;

    // This memory range contains data which will be overwritten from frame to frame.
    // This includes:
    // - composed COM/rotation
    // - phi and phi dot
    // - spatial vector / acceleration / force
    // - bias vector
    // - mass matrix
    // - mass matrix LTDL
    MemoryRange tmp;

    // We store these to avoid access to memory range intervals at every access.
    void *qVectorsPtr;

    // During initialization time, we use this pointer to store all the
    // desc structs in order to avoid the user needing to pre-determine
    // how many limits / object data etc...
    void *tmpPtr;

    // These functions all assume that the qVectorsPtr / tmpPtr are valid.
    inline float * q(BodyGroupProperties);
    inline float * qv(BodyGroupProperties);
    inline float * dqv(BodyGroupProperties);
    inline float * f(BodyGroupProperties);
    inline float * mus(BodyGroupProperties);
    inline BodyLimitConstraint * limits(BodyGroupProperties);
    inline BodyInertial * inertials(BodyGroupProperties);
    inline int32_t * expandedParent(BodyGroupProperties);
    inline int32_t * dofToBody(BodyGroupProperties);
    inline uint32_t * isStatic(BodyGroupProperties);
    inline BodyObjectData * objectData(BodyGroupProperties);
    inline BodyHierarchy * hierarchies(BodyGroupProperties);
    inline Entity * entities(BodyGroupProperties);
    inline BodyOffsets * offsets(BodyGroupProperties);
    inline BodyNameHash * nameHashes(BodyGroupProperties);

    inline BodyTransform * bodyTransforms(BodyGroupProperties);
    inline BodyPhi * bodyPhi(BodyGroupProperties);
    inline BodySpatialVectors * spatialVectors(BodyGroupProperties);
    inline float * biasVector(BodyGroupProperties);
    inline float * massMatrix(BodyGroupProperties);
    inline float * massLTDLMatrix(BodyGroupProperties);
    inline float * phiFull(BodyGroupProperties);
    inline float * phiDot(BodyGroupProperties);
    inline float * scratch(BodyGroupProperties);

    static inline uint32_t qVectorsNumBytes(BodyGroupProperties p);
    static inline uint32_t tmpNumBytes(BodyGroupProperties p);
};
struct BodyGroupArchetype : Archetype<
    BodyGroupProperties,
    BodyGroupMemory
> {};

// Wrapper for body groups which require initialization
//  (use for systems which only need to run once)
struct InitBodyGroup {
    Entity bodyGroup;
};

struct InitBodyGroupArchetype : Archetype<
    InitBodyGroup
> {};

struct DestroyBodyGroup {
    Entity bodyGroup;
};

struct DestroyBodyGroupArchetype : Archetype<
    DestroyBodyGroup
> {};

struct LinkParentDofObject {
    enum class Type {
        Collider,
        Render,
        RenderCollider
    };

    Entity bodyGroup;
    uint32_t bodyIdx;
    uint32_t objDataIdx;

    Type type;
};

// This is the archetype for a single body's link for collision detection
struct LinkCollider : public Archetype<
    RigidBody,
    LinkParentDofObject
> {};

struct LinkVisual : public Archetype<
    base::ObjectInstance,
    render::Renderable,
    LinkParentDofObject
> {};

struct DofObjectGroup {
    Entity bodyGroup;
    uint32_t idx;

    // Cache the row of the body group
    uint32_t bodyGroupRow;
};

struct DofObjectProxies {
    ResponseType responseType;

    uint32_t visualOffset;
    uint32_t numVisuals;

    uint32_t colliderOffset;
    uint32_t numColliders;
};

struct DofObjectArchetype : public Archetype<
    DofObjectGroup,
    DofObjectProxies
> {};

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

struct BodyDesc {
    DofType type;
    math::Vector3 initialPos;
    math::Quat initialRot;
    ResponseType responseType;
    uint32_t numCollisionObjs;
    uint32_t numVisualObjs;
    uint32_t numLimits;
    float mass;
    math::Diag3x3 inertia;
    float muS;
    BodyNameHash hash;
};

// "Global scale" scales everything in the body group uniformly
Entity makeBodyGroup(
        Context &ctx,
        uint32_t num_bodies,
        float global_scale = 1.f);

void destroyBodyGroup(Context &ctx, Entity body_grp);

Entity makeBody(Context &ctx, Entity body_grp, BodyDesc desc);

struct CollisionDesc {
    uint32_t objID;
    math::Vector3 offset;
    math::Quat rotation;
    math::Diag3x3 scale;

    // Required for URDF loading
    uint32_t linkIdx;
    // Index of the collider within the body
    uint32_t subIndex;

    // Optional to visualize the collision entities (-1 means we didn't
    // pass the collision objects to the renderer for visualization)
    int32_t renderObjID;
};

struct VisualDesc {
    uint32_t objID;
    math::Vector3 offset;
    math::Quat rotation;
    math::Diag3x3 scale;

    // Required for URDF loading
    uint32_t linkIdx;
    // Index of the collider within the body
    uint32_t subIndex;
};

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

    // Rotation applied to child's vectors relative to
    // parent's coordinate system.
    math::Quat relParentRotation;

    // This is in the child's coordinate basis
    math::Vector3 slideVector;
};

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent,
        Entity child,
        JointSlider slider_info);

struct JointFixed {
    // In the parent's coordinate basis
    math::Vector3 relPositionParent;

    // In the child's coordinate basis
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
        JointFixed fixed_info);

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

// Disable the collision between all colliders in joint_a
// and colliders in joint_b
void disableJointCollisions(
        Context &ctx,
        Entity grp,
        Entity joint_a,
        Entity joint_b);

// Positions, velocities, accelerations in joint space
float * getBodyGroupDofPos(Context &ctx, Entity body_grp);
float * getBodyGroupDofVel(Context &ctx, Entity body_grp);
float * getBodyGroupDofAcc(Context &ctx, Entity body_grp);
float * getBodyGroupForces(Context &ctx, Entity body_grp);

uint8_t getBodyNumDofs(Context &ctx, Entity body_grp, uint32_t body_idx);

float * getBodyDofPos(Context &ctx, Entity body_grp, uint32_t body_idx);
float * getBodyDofVel(Context &ctx, Entity body_grp, uint32_t body_idx);
float * getBodyDofAcc(Context &ctx, Entity body_grp, uint32_t body_idx);
float * getBodyForces(Context &ctx, Entity body_grp, uint32_t body_idx);

uint8_t getBodyNumDofs(Context &ctx, Entity body_grp, StringID string_id);

// The string ID can either be the link or joint name
float * getBodyDofPos(Context &ctx, Entity body_grp, StringID string_id);
float * getBodyDofVel(Context &ctx, Entity body_grp, StringID string_id);
float * getBodyDofAcc(Context &ctx, Entity body_grp, StringID string_id);
float * getBodyForces(Context &ctx, Entity body_grp, StringID string_id);

// Cartesian
BodyTransform getBodyWorldPos(Context &ctx, Entity body_grp, uint32_t body_idx);
BodyTransform getBodyWorldPos(Context &ctx, Entity body_grp, StringID string_id);

BodyInertial & getBodyInertial(Context &ctx, Entity body_grp, uint32_t body_idx);
BodyInertial & getBodyInertial(Context &ctx, Entity body_grp, StringID string_id);

// External forces:
void addHingeExternalForce(Context &ctx, Entity hinge_joint, float newtons);

// Mark the body group as requiring a reset
void markReset(Context &ctx, Entity body_grp);

void registerTypes(ECSRegistry &registry);
void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id);
void init(Context &ctx, CVXSolve *cvx_solve);

TaskGraphNodeID setupCVInitTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

TaskGraphNodeID setupCVResetTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps);

void initializeHierarchies(Context &ctx);

void setEnablePhysics(Context &ctx, bool value);

}

#include "cv_physics.inl"
