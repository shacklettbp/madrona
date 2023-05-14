#pragma once

#include <madrona/physics.hpp>
#include <madrona/importer.hpp>

namespace madrona {
namespace phys {

class PhysicsLoader {
public:
    enum class StorageType {
        CPU,
        CUDA,
    };

    struct ImportedRigidBodies {
        ~ImportedRigidBodies();

        // Per Primitive Data
        HeapArray<CollisionPrimitive> collisionPrimitives;
        HeapArray<math::AABB> primitiveAABBs;

        // Per Object Data
        HeapArray<uint32_t> primOffsets;
        HeapArray<uint32_t> primCounts;
        HeapArray<RigidBodyMetadata> metadatas;
        HeapArray<math::AABB> objectAABBs;
    };

    struct SourceCollisionPrimitive {
        struct Hull {
            const imp::SourceMesh *mesh;
        };

        CollisionPrimitive::Type type;
        union {
            CollisionPrimitive::Sphere sphere;
            Hull hull;
        };
        float invMass;
        RigidBodyFrictionData friction;
    };

    struct SourceCollisionObject {
        Span<SourceCollisionPrimitive> prims;
    };

    PhysicsLoader(StorageType storage_type, CountT max_objects);
    ~PhysicsLoader();
    PhysicsLoader(PhysicsLoader &&o);

    ImportedRigidBodies importRigidBodyData(
        const SourceCollisionObject *collision_objs,
        CountT num_objects,
        bool merge_coplanar_faces = false);

    CountT loadObjects(const CollisionPrimitive *primitives,
                       const math::AABB *primitive_aabbs,
                       const uint32_t *prim_offsets,
                       const uint32_t *prim_counts,
                       const RigidBodyMetadata *metadatas,
                       const math::AABB *obj_aabbs,
                       CountT total_num_primitives,
                       CountT num_objs);


    ObjectManager & getObjectManager();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
