#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/span.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona::phys {

namespace geometry {

// TODO: Should probably wrap this with proper RAII (eh?) - not supposed to be an API
struct FastPolygonList {
    uint32_t maxIndices;
    uint32_t *buffer;
    uint32_t size;
    uint32_t edgeCount;
    uint32_t polygonCount;

    FastPolygonList &operator=(const FastPolygonList &other);

    void allocate(uint32_t maxIdx);

    void free();

    // Creation
    void addPolygon(Span<const uint32_t> vertex_indices);

    // Iteration helper functions
    inline uint32_t *begin() { return &buffer[1]; }
    inline uint32_t *next(uint32_t *iterator) { return iterator + iterator[-1] + 1; }
    inline uint32_t *end() { return buffer + size + 1; }
    inline uint32_t getPolygonVertexCount(uint32_t *iterator) { return iterator[-1]; }
    inline uint32_t getIteratorIndex(uint32_t *iterator) { return iterator - buffer; }
    inline uint32_t *getIteratorFromIteratorIndex(uint32_t index) { return &buffer[index]; }
};

// Ok this is quite confusing right now but polygon data is data for each polygon
// This could maybe in the future be a struct or something but for now is just an int
// indexing to a half edge which is part of the polygon
using PolygonData = uint32_t;

// PolygonID is an int indexing into the list of PolygonDatas
// In order to get the half edge of a polygon given PolygonID,
// you would index into the list of Polygons with the PolygonID,
// then use the half edge index stored in PolygonData to access
// the corresponding half edge
using PolygonID = uint32_t;
using EdgeData = uint32_t;
using HalfEdgeID = uint32_t;

// This is an index into the mVertices array
using VertexID = uint32_t;

struct HalfEdge {
    // Don't really need anything else for our purposes
    HalfEdgeID next;
    HalfEdgeID twin;
    VertexID rootVertex;
    // Face (iterator in the polygon structure - for now into the fast polygon list)
    PolygonID polygon;
};

struct Plane {
    math::Vector3 normal; // Potentially unnormalized
    float d;
};

struct Segment {
    math::Vector3 p1;
    math::Vector3 p2;
};


// For our purposes, we just need to be able to easily iterate
// over all the faces and edges of the mesh. That's it
class HalfEdgeMesh {
public:
    // Accept different formats
    void construct(
            FastPolygonList &polygons,
            uint32_t vertexCount, const math::Vector3 *vertices);

    // Normalized
    math::Vector3 getFaceNormal(PolygonID polygon,
                                const math::Vector3 *vertices) const;

    // Normalized normal
    Plane getPlane(PolygonID polygon, const math::Vector3 *vertices) const;

    template <typename Fn>
    void iteratePolygonIndices(PolygonID poly, Fn &&fn);

    // Get ordered vertices of a polygon (face)
    uint32_t getPolygonVertices(
            const PolygonID &polygon,
            math::Vector3 *outVertices,
            const math::Vector3 *vertices) const;

    void getPolygonVertices(math::Vector3 *outVertices, const PolygonID &polygon, const math::Vector3 *vertices) const;


    // Can be used one after the other
    uint32_t getPolygonVertexCount(const PolygonID &polygon) const;
    void getPolygonSidePlanes(Plane *planes, const PolygonID &polygon, const math::Vector3 *vertices) const;

    // Normalized normals
    std::pair<math::Vector3, math::Vector3> getEdgeNormals(const HalfEdge &hEdge, math::Vector3 *vertices) const;

    // Normalized direction
    math::Vector3 getEdgeDirection(const EdgeData &edge, const math::Vector3 *vertices) const;

    Segment getEdgeSegment(const EdgeData &edge, math::Vector3 *vertices) const;

    // Normalized direction
    math::Vector3 getEdgeDirection(const HalfEdge &edge, math::Vector3 *vertices) const;

    math::Vector3 getEdgeOrigin(const EdgeData &edge, math::Vector3 *vertices) const;
    math::Vector3 getEdgeOrigin(const HalfEdge &edge, math::Vector3 *vertices) const;

public:
    inline uint32_t getVertexCount() const;
    inline const math::Vector3 & vertex(uint32_t id) const;
    inline const math::Vector3 * vertices() const;

    uint32_t getPolygonCount() const;
    const PolygonData &polygon(uint32_t id) const;

    uint32_t getEdgeCount() const;
    const EdgeData &edge(uint32_t id) const;

    const HalfEdge &halfEdge(HalfEdgeID id) const;

public:
    // For now, just array of indices which point into the half edge array
    PolygonData *mPolygons;
    Plane *mFacePlanes;
    EdgeData    *mEdges;
    // Where all the half edges are stored
    HalfEdge    *mHalfEdges;
    math::Vector3   *mVertices;

    uint32_t mHalfEdgeCount;
    uint32_t mPolygonCount;
    uint32_t mEdgeCount;
    uint32_t mVertexCount;
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
    Loc e1;
    Loc e2;
    math::Vector3 r1;
    math::Vector3 r2;
    math::Quat axes1;
    math::Quat axes2;
    float separation;
};

struct ConstraintData : Archetype<JointConstraint> {};

// Per object state
struct RigidBodyMetadata {
    math::Vector3 invInertiaTensor;
    float invMass;
    float muS;
    float muD;
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
    RigidBodyMetadata *metadata;
    math::AABB *aabbs;
    CollisionPrimitive *primitives;

    // Half Edge Mesh Buffers
    geometry::PolygonData *polygonDatas;

    // everywhere needs all 4 components besides finding
    // the minimizing face
    geometry::Plane *facePlanes;

    geometry::EdgeData *edgeDatas;
    geometry::HalfEdge *halfEdges;
    math::Vector3 *vertices;
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
    BVH(CountT max_leaves, float leaf_velocity_expansion,
        float leaf_accel_expansion);

    inline LeafID reserveLeaf(Entity e, CollisionPrimitive *prim);
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
    CollisionPrimitive **leaf_primitives_;
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

    static void registerTypes(ECSRegistry &registry);
    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps,
                                        CountT num_substeps);

    static TaskGraph::NodeID setupCleanupTasks(TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> deps);
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
