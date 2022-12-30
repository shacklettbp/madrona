#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/span.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona {
namespace phys {

namespace geometry {
// TODO: Should probably wrap this with proper RAII (eh?) - not supposed to be an API
struct FastPolygonList {
    uint32_t maxIndices;
    uint32_t *buffer;
    uint32_t size;
    uint32_t edgeCount;
    uint32_t polygonCount;

    FastPolygonList &operator=(const FastPolygonList &other);

    void constructCube();

    void allocate(uint32_t maxIdx);

    void free();

    // Creation
    template <typename ...T> void addPolygon(uint32_t count, T &&...vertexIndices);

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
    math::Vector3 point;
    // Has to be normalized
    math::Vector3 normal;
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
            uint32_t vertexCount, math::Vector3 *vertices);

    void constructCube();

    // Normalized
    math::Vector3 getFaceNormal(const PolygonID &polygon, const math::Vector3 *vertices) const;

    // Normalized normal
    Plane getPlane(const PolygonID &polygon, const math::Vector3 *vertices) const;

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
    math::Vector3 getEdgeDirection(const EdgeData &edge, math::Vector3 *vertices) const;

    Segment getEdgeSegment(const EdgeData &edge, math::Vector3 *vertices) const;

    // Normalized direction
    math::Vector3 getEdgeDirection(const HalfEdge &edge, math::Vector3 *vertices) const;

    math::Vector3 getEdgeOrigin(const EdgeData &edge, math::Vector3 *vertices) const;
    math::Vector3 getEdgeOrigin(const HalfEdge &edge, math::Vector3 *vertices) const;

public:
    uint32_t getVertexCount() const;
    const math::Vector3 &vertex(uint32_t id) const;

    uint32_t getPolygonCount() const;
    const PolygonData &polygon(uint32_t id) const;

    uint32_t getEdgeCount() const;
    const EdgeData &edge(uint32_t id) const;

    const HalfEdge &halfEdge(HalfEdgeID id) const;

public:
    // For now, just array of indices which point into the half edge array
    PolygonData *mPolygons;
    EdgeData    *mEdges;
    // Where all the half edges are stored
    HalfEdge    *mHalfEdges;
    math::Vector3   *mVertices;

    uint32_t mHalfEdgeCount;
    uint32_t mPolygonCount;
    uint32_t mEdgeCount;
    uint32_t mVertexCount;
};

// Contains transformed vertices for given geometry
struct CollisionMesh {
    uint32_t vertexCount;
    math::Vector3 *vertices;
    math::Vector3 center;

    // This is also going to contain FastPolygonList for information about each face
    const geometry::HalfEdgeMesh *halfEdgeMesh;

    math::Vector3 position;
    math::Quat rotation;
};
}

struct Velocity {
    math::Vector3 linear;
    math::Vector3 angular;
};

struct CollisionAABB : math::AABB {
    inline CollisionAABB(math::AABB aabb)
        : AABB(aabb)
    {}
};

struct CollisionEvent {
    Entity a;
    Entity b;
};

struct CandidateCollision {
    Entity a;
    Entity b;
};

struct CandidateTemporary : Archetype<CandidateCollision> {};

struct Contact {
    Entity ref;
    Entity alt;
    math::Vector4 points[4];
    int32_t numPoints;
    math::Vector3 normal;
};

struct CollisionEventTemporary : Archetype<CollisionEvent> {};


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
    geometry::EdgeData *edgeDatas;
    geometry::HalfEdge *halfEdges;
    math::Vector3 *vertices;
};

namespace broadphase {

struct LeafID {
    int32_t id;
};

class BVH {
public:
    BVH(CountT max_leaves);

    inline LeafID reserveLeaf(Entity e);

    template <typename Fn>
    inline void findOverlaps(const math::AABB &aabb, Fn &&fn) const;

    void updateLeaf(LeafID leaf_id,
                    const CollisionAABB &obj_aabb);

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

    inline CountT numInternalNodes(CountT num_leaves) const;

    void rebuild();
    void refit(LeafID *leaf_ids, CountT num_moved);

    Node *nodes_;
    CountT num_nodes_;
    const CountT num_allocated_nodes_;
    math::AABB *leaf_aabbs_;
    math::Vector3 *leaf_centers_;
    uint32_t *leaf_parents_;
    Entity *leaf_entities_;
    int32_t *sorted_leaves_;
    std::atomic<int32_t> num_leaves_;
    int32_t num_allocated_leaves_;
    bool force_rebuild_;
};

void updateLeavesEntry(
    Context &ctx,
    const LeafID &leaf_id,
    const CollisionAABB &aabb);

void updateBVHEntry(
    Context &, BVH &bvh);

void findOverlappingEntry(
    Context &ctx,
    const Entity &e,
    const CollisionAABB &obj_aabb,
    const Velocity &);

}

namespace solver {

struct SubstepPrevState {
    math::Vector3 prevPosition;
    math::Quat prevRotation;
};

struct SubstepStartState {
    math::Vector3 startPosition;
    math::Quat startRotation;
};

struct SubstepVelocityState {
    math::Vector3 prevLinear;
    math::Vector3 prevAngular;
};

}

struct RigidBodyPhysicsSystem {
    static void init(Context &ctx,
                     ObjectManager *obj_mgr,
                     float delta_t,
                     CountT num_substeps,
                     CountT max_dynamic_objects,
                     CountT max_contacts_per_step);

    static void reset(Context &ctx);
    static broadphase::LeafID registerEntity(Context &ctx, Entity e);

    static void registerTypes(ECSRegistry &registry);
    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps,
                                        CountT num_substeps);

    static TaskGraph::NodeID setupCleanupTasks(TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> deps);

};


namespace narrowphase {

struct FaceQuery {
    float separation;
    int32_t faceIdx;
};

struct EdgeQuery {
    float separation;
    math::Vector3 normal;
    int32_t edgeIdxA;
    int32_t edgeIdxB;
};

struct Manifold {
    math::Vector4 contactPoints[4];
    int32_t numContactPoints;
    math::Vector3 normal;
    bool aIsReference;
};

// Returned vertices will be stored in linear bump allocator
Manifold doSAT(const geometry::CollisionMesh &a, const geometry::CollisionMesh &b);
Manifold doSATPlane(const geometry::Plane &plane, const geometry::CollisionMesh &a);

inline float abs(float f) {
    return (f < 0.0f) ? -f : f;
}

// Returns the signed distance
inline float getDistanceFromPlane(const geometry::Plane &plane, const math::Vector3 &a) {
    float adotn = a.dot(plane.normal);
    float pdotn = plane.point.dot(plane.normal);
    return (adotn - pdotn);
}

// Need to be normalized
inline bool areParallel(const math::Vector3 &a, const math::Vector3 &b) {
    float d = abs(a.dot(b));

    return abs(d - 1.0f) < 0.0001f;
}

// Get intersection on plane of the line passing through 2 points
inline math::Vector3 planeIntersection(const geometry::Plane &plane, const math::Vector3 &p1, const math::Vector3 &p2) {
    float distance = getDistanceFromPlane(plane, p1);

    return p1 + (p2 - p1) * (-distance / plane.normal.dot(p2 - p1));
}


}


}
}

#include "physics.inl"

