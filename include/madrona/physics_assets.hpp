#pragma once

#include <madrona/physics.hpp>
#include <madrona/importer.hpp>
#include <madrona/stack_alloc.hpp>
#include <madrona/mesh_bvh.hpp>

#include <madrona/geo.hpp>

namespace madrona::phys {

struct SourceCollisionPrimitive {
    struct HullInput {
        uint32_t hullIDX;
    };

    CollisionPrimitive::Type type;
    union {
        CollisionPrimitive::Sphere sphere;
        CollisionPrimitive::Plane plane;
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
        geo::HalfEdge *halfEdges;
        uint32_t *faceBaseHalfEdges;
        geo::Plane *facePlanes;
        math::Vector3 *vertices;

        uint32_t numHalfEdges;
        uint32_t numFaces;
        uint32_t numVerts;
    } hullData;

    // Per Primitive Data
    CollisionPrimitive *primitives;
    math::AABB *primitiveAABBs;

    // Per Object Data
    RigidBodyMetadata *metadatas;
    math::AABB *objAABBs;
    uint32_t *primOffsets;
    uint32_t *primCounts;

    uint32_t numConvexHulls;
    uint32_t totalNumPrimitives;
    uint32_t numObjs;

    static void * processRigidBodyAssets(
        Span<const imp::SourceMesh> convex_hull_meshes,
        Span<const SourceCollisionObject> collision_objs,
        bool build_convex_hulls,
        StackAlloc &tmp_alloc,
        RigidBodyAssets *out_assets,
        CountT *out_num_bytes);
};


}
