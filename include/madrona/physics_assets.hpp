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

    PhysicsLoader(StorageType storage_type, CountT max_objects);
    ~PhysicsLoader();
    PhysicsLoader(PhysicsLoader &&o);

    struct ImportedCollisionMeshes {
        HeapArray<geometry::HalfEdgeMesh> halfEdgeMeshes;
        HeapArray<math::AABB> meshAABBs;

        HeapArray<uint32_t> primOffsets;
        HeapArray<uint32_t> primCounts;
        HeapArray<RigidBodyMassData> massDatas;
        HeapArray<math::AABB> objectAABBs;
    };

    ImportedCollisionMeshes importCollisionMeshes(
        const imp::SourceObject *src_objects,
        const float *inv_masses, 
        CountT num_objects,
        bool merge_coplanar_faces);

    CountT loadObjects(const RigidBodyMetadata *metadatas,
                       const math::AABB *aabbs,
                       const CollisionPrimitive *primitives,
                       CountT num_objs);


    ObjectManager & getObjectManager();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
