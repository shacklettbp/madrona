#pragma once

#include <madrona/range.hpp>
#include <madrona/components.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona::phys::cv {

// Attach this component to entities that you want to have obey physics.
struct CVPhysicalComponent {
    // This is going to be one of the DOF entities (i.e. DOFFreeBodyArchetype).
    Entity physicsEntity;
};

static constexpr uint32_t kMaxPositionCoords = 7;
static constexpr uint32_t kMaxVelocityCoords = 6;

enum class DofType {
    // The number of unique degrees of freedom (SE3)
    FreeBody = 6,
    Hinge = 1,
    Ball = 3,
    FixedBody = 0,

    // When we add other types of physics DOF objects, we will encode
    // the number of degrees of freedom they all have here.
};

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

struct DofObjectVelocity {
    float qv[kMaxVelocityCoords];
};

struct DofObjectAcceleration {
    float dqv[kMaxVelocityCoords];
};

struct DofObjectNumDofs {
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

struct PhiUnit {
    static constexpr CountT kNumValsPerUnit = 16;
    float values[kNumValsPerUnit];
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
    RangeMap/*<PhiUnit>*/ phiFull;

    // The spatial inertia tensor in Plücker coordinates
    // Hold the combined inertia of subtree after combineSpatialInertia
    InertiaTensor spatialInertia;

    // Velocity, Acceleration, Force in Plücker coordinates (used for RNE)
    SpatialVector sVel;
    SpatialVector sAcc;
    SpatialVector sForce;
};

struct DofObjectHierarchyDesc {
    // The child is going to query the sync value of the parent.
    // If the value is 0, the child must keep waiting. Otherwise,
    // if it's 1, that means that the child can look up whatever
    // value it's been waiting on.
    AtomicU32 sync;
    
    Entity parent;

    // Relative position of the joint to the parent's COM.
    math::Vector3 relPositionParent;

    // Relative position of the child's COM to the joint.
    math::Vector3 relPositionLocal;
    
    // Extra data:
    // For hinge, this is the hinge rotation axis.
    // For ball, this is the vector perpendicular to the plane of allowed
    //           angular velocities
    math::Vector3 hingeAxis;

    bool leaf;

    // Index in the body group hierarchy
    int32_t index;
    // Index of the parent in the body group hierarchy. -1 if root.
    int32_t parentIndex;

    Loc bodyGroup;
};

struct DofObjectArchetype : public Archetype<
    DofObjectPosition,
    DofObjectVelocity,
    DofObjectAcceleration, // Δv, used for integration

    DofObjectTmpState,

    DofObjectHierarchyDesc,

    // Currently, this is being duplicated but it's small. We can
    // maybe find a way around this later.
    base::ObjectID,

    DofObjectNumDofs
> {};



// Example of range map allocation usage.
struct MassMatrixUnit {
    static constexpr CountT kNumValsPerUnit = 16;
    float values[kNumValsPerUnit];
};

struct ParentArrayUnit {
    static constexpr CountT kNumValsPerUnit = 16;
    int32_t values[kNumValsPerUnit];
};

struct BodyFloatUnit {
    static constexpr CountT kNumValsPerUnit = 16;
    float values[kNumValsPerUnit];
};

struct BodyGroupHierarchy {
    static constexpr uint32_t kMaxJoints = 8;

    // This includes the free body too which will be at index 0.
    uint32_t numBodies;
    Entity bodies[kMaxJoints];

    // Total number of DOFs in the body group
    uint32_t numDofs;

    // "Expanded" parent array for kinematic tree (chain of 1-DOF joints)
    //   used for factorization. See Featherstone pg. 114
    RangeMap/*<ParentArrayUnit>*/ expandedParent;

    // Center of mass of the body group
    math::Vector3 comPos;

    // Mass matrix (num_dof x num_dof) of the body group
    RangeMap/*<MassMatrixUnit>*/ massMatrix;

    // LTDL factorization of matrix
    RangeMap/*<MassMatrixUnit>*/ massMatrixLTDL;

    // Bias forces (num_dof) of the body group, gets replaced by the
    //  unconstrained acceleration after computeFreeAcceleration
    RangeMap/*<BodyFloatUnit>*/ bias;

    // Temporary index used during stacking
    uint32_t tmpIdx;
};

struct BodyGroup : public Archetype<
    BodyGroupHierarchy
> {};

 

Entity makeCVBodyGroup(Context &ctx);
    
// For now, initial velocities are just going to be 0
// Also, when defining a body group, you need to make sure to define in
// order of parent to children.
void makeCVPhysicsEntity(Context &ctx, 
                         Entity body,
                         base::Position position,
                         base::Rotation rotation,
                         base::ObjectID obj_id,
                         DofType dof_type);

void cleanupPhysicalEntity(Context &ctx, Entity e);



void registerTypes(ECSRegistry &registry);
void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id);
void init(Context &ctx, CVXSolve *cvx_solve);
TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps);

void setCVGroupRoot(Context &ctx,
                    Entity body_group,
                    Entity body);

void setCVEntityParentHinge(Context &ctx,
                            Entity body_group,
                            Entity parent, Entity child,
                            math::Vector3 rel_pos_parent,
                            math::Vector3 rel_pos_child,
                            math::Vector3 hinge_axis);

void setCVEntityParentBall(Context &ctx,
                           Entity body_group,
                           Entity parent, Entity child,
                           math::Vector3 rel_pos_parent,
                           math::Vector3 rel_pos_child);

void initializeHierarchies(Context &ctx);

}
