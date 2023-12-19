#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/span.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona::phys {

namespace geometry {

struct HalfEdge {
    uint32_t next;
    uint32_t rootVertex;
    uint32_t face;
};

struct Plane {
    math::Vector3 normal; // Potentially unnormalized
    float d;
};

struct Segment {
    math::Vector3 p1;
    math::Vector3 p2;
};

struct HalfEdgeMesh {
    template <typename Fn>
    inline void iterateFaceIndices(uint32_t face, Fn &&fn) const;
    inline uint32_t twinIDX(uint32_t half_edge_id) const;
    inline uint32_t numEdges() const;
    inline uint32_t edgeToHalfEdge(uint32_t edge_id) const;

    HalfEdge *halfEdges;
    uint32_t *faceBaseHalfEdges;
    Plane *facePlanes;
    math::Vector3 *vertices;

    uint32_t numHalfEdges;
    uint32_t numFaces;
    uint32_t numVertices;
};

}

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

struct CollisionEvent {
    Entity a;
    Entity b;
};

struct CandidateCollision {
    Loc a;
    Loc b;
    uint32_t aPrim;
    uint32_t bPrim;
};

struct CandidateTemporary : Archetype<CandidateCollision> {};

struct Contact {
    Loc ref;
    Loc alt;
    math::Vector4 points[4];
    int32_t numPoints;
    math::Vector3 normal;
    float lambdaN[4];
};

struct CollisionEventTemporary : Archetype<CollisionEvent> {};

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

    static inline JointConstraint setupFixed(
        Entity e1, Entity e2,
        math::Quat attach_rot1, math::Quat attach_rot2,
        math::Vector3 r1, math::Vector3 r2,
        float separation);

    static inline JointConstraint setupHinge(
        Entity e1, Entity e2,
        math::Vector3 a1_local, math::Vector3 a2_local,
        math::Vector3 b1_local, math::Vector3 b2_local,
        math::Vector3 r1, math::Vector3 r2);
};

struct ConstraintData : Archetype<JointConstraint> {};

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
        geometry::HalfEdgeMesh halfEdgeMesh;
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

namespace broadphase {

struct LeafID {
    int32_t id;
};

class BVH {
public:
    BVH(const ObjectManager *obj_mgr,
        CountT max_leaves,
        float leaf_velocity_expansion,
        float leaf_accel_expansion);

    inline LeafID reserveLeaf(Entity e, base::ObjectID obj_id);
    inline math::AABB getLeafAABB(LeafID leaf_id) const;

    template <typename Fn>
    inline void findOverlaps(const math::AABB &aabb, Fn &&fn) const;

    template <typename Fn>
    inline void findOverlapsForLeaf(LeafID leaf_id, Fn &&fn) const;

    Entity traceRay(math::Vector3 o,
                    math::Vector3 d,
                    float *out_hit_t,
                    math::Vector3 *out_hit_normal,
                    float t_max = float(INFINITY));

    void updateLeafPosition(LeafID leaf_id,
                            const math::Vector3 &pos,
                            const math::Quat &rot,
                            const math::Diag3x3 &scale,
                            const math::Vector3 &linear_vel,
                            const math::AABB &obj_aabb);

    math::AABB expandLeaf(LeafID leaf_id,
                          const math::Vector3 &linear_vel);

    void refitLeaf(LeafID leaf_id, const math::AABB &leaf_aabb);

    inline void rebuildOnUpdate();
    void updateTree();

    inline void clearLeaves();

private:
    static constexpr int32_t sentinel_ = 0xFFFF'FFFF_i32;

    struct Node {
        float minX[4];
        float minY[4];
        float minZ[4];
        float maxX[4];
        float maxY[4];
        float maxZ[4];
        int32_t children[4];
        int32_t parentID;

        inline bool isLeaf(CountT child) const;
        inline int32_t leafIDX(CountT child) const;

        inline void setLeaf(CountT child, int32_t idx);
        inline void setInternal(CountT child, int32_t internal_idx);
        inline bool hasChild(CountT child) const;
        inline void clearChild(CountT child);
    };

    // FIXME: evaluate whether storing this in-line in the tree
    // makes sense or if we should force a lookup through the entity ID
    struct LeafTransform {
        math::Vector3 pos;
        math::Quat rot;
        math::Diag3x3 scale;
    };

    inline CountT numInternalNodes(CountT num_leaves) const;

    void rebuild();
    void refit(LeafID *leaf_ids, CountT num_moved);

    bool traceRayIntoLeaf(int32_t leaf_idx,
                          math::Vector3 world_ray_o,
                          math::Vector3 world_ray_d,
                          float t_min,
                          float t_max,
                          float *hit_t,
                          math::Vector3 *hit_normal);

    Node *nodes_;
    CountT num_nodes_;
    const CountT num_allocated_nodes_;
    Entity *leaf_entities_;
    const ObjectManager *obj_mgr_;
    base::ObjectID *leaf_obj_ids_;
    math::AABB *leaf_aabbs_; // FIXME: remove this, it's duplicated data
    LeafTransform  *leaf_transforms_;
    uint32_t *leaf_parents_;
    int32_t *sorted_leaves_;
    AtomicI32 num_leaves_;
    int32_t num_allocated_leaves_;
    float leaf_velocity_expansion_;
    float leaf_accel_expansion_;
    bool force_rebuild_;
};

}

namespace solver {

struct SubstepPrevState {
    math::Vector3 prevPosition;
    math::Quat prevRotation;
};

struct PreSolvePositional {
    math::Vector3 x;
    math::Quat q;
};

struct PreSolveVelocity {
    math::Vector3 v;
    math::Vector3 omega;
};

}

struct RigidBodyPhysicsSystem {
    static void init(Context &ctx,
                     ObjectManager *obj_mgr,
                     float delta_t,
                     CountT num_substeps,
                     math::Vector3 gravity,
                     CountT max_dynamic_objects,
                     CountT max_contacts_per_world,
                     CountT max_joint_constraints_per_world);

    static void reset(Context &ctx);
    static broadphase::LeafID registerEntity(Context &ctx,
                                             Entity e,
                                             base::ObjectID obj_id);

    template <typename Fn>
    static void findEntitiesWithinAABB(Context &ctx,
                                       math::AABB aabb,
                                       Fn &&fn);

    static bool checkEntityAABBOverlap(Context &ctx,
                                       math::AABB aabb,
                                       Entity e);

    static void registerTypes(ECSRegistry &registry);

    static TaskGraphNodeID setupBroadphaseTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    static TaskGraphNodeID setupSubstepTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps,
        CountT num_substeps);

    static TaskGraphNodeID setupCleanupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);
};

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

#include "physics.inl"
