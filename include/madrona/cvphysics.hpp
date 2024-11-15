#pragma once

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
    // TODO: Other types of DOF objects.
    float q[kMaxPositionCoords];
};

struct DofObjectVelocity {
    float qv[kMaxVelocityCoords];
};

struct DofObjectNumDofs {
    uint32_t numDofs;
};

struct ContactTmpState {
    float mu;
    math::Vector3 n;
    math::Vector3 t;
    math::Vector3 s;
    math::Vector3 rRefComToPt[4];
    math::Vector3 rAltComToPt[4];
    float maxPenetration;
    uint32_t num_contacts;
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
    // Phi is parameterized by at most 6 values.
    //
    // For the free body, it just depends on the center of mass of the body.
    //
    // For the hinge, it depends on the normalized angular velocity vector 
    // (first 3 values) of the hinge and the position of the joint 
    // (not the body - last 3 values).
    float v[6];
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

// Just some space to store temporary per-entity data.
struct DofObjectTmpState {
    // World-space position of body COM
    math::Vector3 comPos;

    // World-space orientation of body
    math::Quat composedRot;

    // Position of the rotation point. For free body, it's just the COM.
    // For the hinge, it's the position of the joint.
    math::Vector3 anchorPos;

    // Map from generalized velocities to Plücker coordinates
    Phi phi;

    // The spatial inertia tensor in Plücker coordinates
    // Hold the combined inertia of subtree after combineSpatialInertia
    InertiaTensor spatialInertia;

    // Velocity, Acceleration, Force in Plücker coordinates
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
    
#ifdef MADRONA_GPU_MODE
    static_assert(false, "Need to implement GPU DOF object hierarchy");
#else
    Entity parent;
#endif

    // Relative position of the joint to the parent's COM.
    math::Vector3 relPositionParent;

    // Relative position of the child's COM to the joint.
    math::Vector3 relPositionLocal;
    
    // Extra data:
    math::Vector3 hingeAxis;

    bool leaf;

    // Index in the body group hierarchy
    int32_t index;
    // Index of the parent in the body group hierarchy.
    int32_t parentIndex;
};

struct DofObjectArchetype : public Archetype<
    DofObjectPosition,
    DofObjectVelocity,

    DofObjectTmpState,

    DofObjectHierarchyDesc,

    // Currently, this is being duplicated but it's small. We can
    // maybe find a way around this later.
    base::ObjectID,

    DofObjectNumDofs
> {};

struct BodyGroupHierarchy {
    static constexpr uint32_t kMaxJoints = 8;

    // This includes the free body too which will be at index 0.
    uint32_t numBodies;
    Entity bodies[kMaxJoints];

    // Center of mass of the body group
    math::Vector3 comPos;

    // Total number of DOFs in the body group
    uint32_t numDofs;

    // Mass matrix (num_dof x num_dof) of the body group
    float *massMatrix;

    // Bias forces (num_dof) of the body group
    float *biasForces;
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

}
