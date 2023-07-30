#pragma once

#include <madrona/physics.hpp>

namespace madrona::phys {

struct SourceCollisionPrimitive {
    struct HullInput {
        uint32_t hullIDX;
    };

    CollisionPrimitive::Type type;
    union {
        CollisionPrimitive::Sphere sphere;
        HullInput hullInput;
    };
};

struct SourceCollisionObject {
    Span<const SourceCollisionPrimitive> prims;
    float invMass;
    RigidBodyFrictionData friction;
};

struct RigidBodyAssets {
    struct HullData {
        const geometry::HalfEdge *halfEdges;
        const uint32_t *faceBaseHalfEdges;
        const geometry::Plane *facePlanes;
        const math::Vector3 *vertices;

        uint32_t numHalfEdges;
        uint32_t numFaces;
        uint32_t numVerts;
    } hullData;

    // Per Primitive Data
    const CollisionPrimitive *primitives;
    const math::AABB *primitiveAABBs;

    // Per Object Data
    const RigidBodyMetadata *metadatas;
    const math::AABB *objAABBs;
    const uint32_t *primOffsets;
    const uint32_t *primCounts;

    uint32_t numConvexHulls;
    uint32_t totalNumPrimitives;
    uint32_t numObjs;

    static void * processRigidBodies(
        Span<const imp::SourceMesh> convex_hull_meshes,
        Span<const SourceCollisionPrimitive> collision_objs,
        bool rebuild_hulls,
        StackAlloc &tmp_alloc,
        RigidBodyAssets *out_assets);
};

}
