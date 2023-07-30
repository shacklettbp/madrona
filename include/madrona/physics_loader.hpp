#pragma once

#include <madrona/physics.hpp>
#include <madrona/importer.hpp>

#include <madrona/exec_mode.hpp>

namespace madrona::phys {

struct RigidBodyAssets {
    struct Hulls {
        const geometry::HalfEdge *halfEdges;
        const uint32_t *faceBaseHalfEdges;
        const geometry::Plane *facePlanes;
        const math::Vector3 *vertices;

        CountT totalNumHullHalfedges;
        CountT totalNumHullFaces;
        CountT totalNumHullVerts;
    } hulls;

    // Per Primitive Data
    const CollisionPrimitive *primitives;
    const math::AABB *primitiveAABBs;

    // Per Object Data
    const RigidBodyMetadata *metadatas;
    const math::AABB *objAABBs;
    const uint32_t *primOffsets;
    const uint32_t *primCounts;

    CountT totalNumPrimitives;
    CountT numObjs;
};

class PhysicsLoader {
public:
    PhysicsLoader(ExecMode exec_mode, CountT max_objects);
    ~PhysicsLoader();
    PhysicsLoader(PhysicsLoader &&o);

    Optional<ImportedRigidBodies> importRigidBodyData(
        const SourceCollisionObject *collision_objs,
        CountT num_objects,
        bool build_hulls = true);

    CountT loadObjects(const RigidBodyMetadata *metadatas,
                       const math::AABB *obj_aabbs,
                       const uint32_t *prim_offsets,
                       const uint32_t *prim_counts,
                       CountT num_objs,
                       const CollisionPrimitive *primitives,
                       const math::AABB *primitive_aabbs,
                       CountT total_num_primitives,
                       const geometry::HalfEdge *hull_halfedges,
                       CountT total_num_hull_halfedges,
                       const uint32_t *hull_face_base_halfedges,
                       const geometry::Plane *hull_face_planes,
                       CountT total_num_hull_faces,
                       const math::Vector3 *hull_verts,
                       CountT total_num_hull_verts);



    ObjectManager & getObjectManager();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
