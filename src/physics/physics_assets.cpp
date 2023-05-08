#include <madrona/physics_assets.hpp>
#include <madrona/importer.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace madrona {
namespace phys {

#ifndef MADRONA_CUDA_SUPPORT
[[noreturn]] static void noCUDA()
{
    FATAL("PhysicsLoader: Not built with CUDA support");
}
#endif

struct PhysicsLoader::Impl {
    RigidBodyMetadata *metadatas;
    math::AABB *aabbs;
    CollisionPrimitive *primitives;

    // For half edge meshes
    geometry::PolygonData *polygonDatas;
    geometry::Plane *facePlanes;
    geometry::EdgeData *edgeDatas;
    geometry::HalfEdge *halfEdges;
    math::Vector3 *vertices;

    CountT polygonCount;
    CountT edgeCount;
    CountT halfEdgeCount;
    CountT vertexCount;

    ObjectManager *mgr;
    CountT curLoadedObjs;
    CountT maxObjs;
    StorageType storageType;

    static Impl * init(StorageType storage_type, CountT max_objects)
    {
        CountT max_vertices_per_object = 100;
        CountT max_polygons_per_object = 100;
        CountT max_edges_per_object = 100;
        CountT max_half_edges_per_object = 100;

        size_t num_metadata_bytes =
            sizeof(RigidBodyMetadata) * max_objects;

        size_t num_aabb_bytes =
            sizeof(math::AABB) * max_objects;

        size_t num_primitive_bytes =
            sizeof(CollisionPrimitive) * max_objects;

        size_t num_vertices_bytes =
            sizeof(math::Vector3) * max_objects * max_vertices_per_object; 

        size_t num_polygon_bytes =
            sizeof(geometry::PolygonData) * max_objects * max_polygons_per_object; 

        size_t num_face_plane_bytes =
            sizeof(geometry::Plane) * max_objects * max_polygons_per_object; 

        size_t num_edges_bytes =
            sizeof(geometry::EdgeData) * max_objects * max_edges_per_object; 

        size_t num_half_edges_bytes =
            sizeof(geometry::HalfEdge) * max_objects * max_half_edges_per_object; 

        RigidBodyMetadata *metadata_ptr;
        math::AABB *aabb_ptr;
        CollisionPrimitive *primitives;
        geometry::PolygonData *polygonDatas_ptr;
        geometry::Plane *facePlanes_ptr;
        geometry::EdgeData *edgeDatas_ptr;
        geometry::HalfEdge *halfEdges_ptr;
        math::Vector3 *vertices_ptr;

        ObjectManager *mgr;

        switch (storage_type) {
        case StorageType::CPU: {
            metadata_ptr =
                (RigidBodyMetadata *)malloc(num_metadata_bytes);
            aabb_ptr = (math::AABB *)malloc(num_aabb_bytes);
            primitives = (CollisionPrimitive *)malloc(num_primitive_bytes);

            polygonDatas_ptr = (geometry::PolygonData *)malloc(num_polygon_bytes);
            facePlanes_ptr = (geometry::Plane *)malloc(num_face_plane_bytes);
            edgeDatas_ptr = (geometry::EdgeData *)malloc(num_edges_bytes);
            halfEdges_ptr = (geometry::HalfEdge *)malloc(num_half_edges_bytes);
            vertices_ptr = (math::Vector3 *)malloc(num_vertices_bytes);

            mgr = new ObjectManager {
                metadata_ptr,
                aabb_ptr,
                primitives,
                polygonDatas_ptr,
                facePlanes_ptr,
                edgeDatas_ptr,
                halfEdges_ptr,
                vertices_ptr
            };
        } break;
        case StorageType::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
            noCUDA();
#else
            metadata_ptr =
                (RigidBodyMetadata *)cu::allocGPU(num_metadata_bytes);
            aabb_ptr = (math::AABB *)cu::allocGPU(num_aabb_bytes);
            primitives =
                (CollisionPrimitive *)cu::allocGPU(num_primitive_bytes);

            polygonDatas_ptr = (geometry::PolygonData *)cu::allocGPU(num_polygon_bytes);
            facePlanes_ptr = (geometry::Plane *)cu::allocGPU(num_face_plane_bytes);
            edgeDatas_ptr = (geometry::EdgeData *)cu::allocGPU(num_edges_bytes);
            halfEdges_ptr = (geometry::HalfEdge *)cu::allocGPU(num_half_edges_bytes);
            vertices_ptr = (math::Vector3 *)cu::allocGPU(num_vertices_bytes);

            mgr = (ObjectManager *)cu::allocGPU(sizeof(ObjectManager));

            ObjectManager local {
                metadata_ptr,
                aabb_ptr,
                primitives,
                polygonDatas_ptr,
                facePlanes_ptr,
                edgeDatas_ptr,
                halfEdges_ptr,
                vertices_ptr
            };

            REQ_CUDA(cudaMemcpy(mgr, &local, sizeof(ObjectManager),
                                cudaMemcpyHostToDevice));
#endif
        } break;
        default: __builtin_unreachable();
        }

        return new Impl {
            .metadatas = metadata_ptr,
            .aabbs = aabb_ptr,
            .primitives = primitives,
            .polygonDatas = polygonDatas_ptr,
            .facePlanes = facePlanes_ptr,
            .edgeDatas = edgeDatas_ptr,
            .halfEdges = halfEdges_ptr,
            .vertices = vertices_ptr,
            .polygonCount = 0,
            .edgeCount = 0,
            .halfEdgeCount = 0,
            .vertexCount = 0,
            .mgr = mgr,
            .curLoadedObjs = 0,
            .maxObjs = max_objects,
            .storageType = storage_type,
        };
    }
};

PhysicsLoader::PhysicsLoader(StorageType storage_type, CountT max_objects)
    : impl_(Impl::init(storage_type, max_objects))
{}

PhysicsLoader::~PhysicsLoader()
{
    if (impl_ == nullptr) {
        return;
    }

    switch (impl_->storageType) {
    case StorageType::CPU: {
        delete impl_->mgr;
        free(impl_->primitives);
        free(impl_->aabbs);
        free(impl_->metadatas);
        free(impl_->polygonDatas);
        free(impl_->facePlanes);
        free(impl_->edgeDatas);
        free(impl_->halfEdges);
        free(impl_->vertices);
    } break;
    case StorageType::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        noCUDA();
#else
        cu::deallocGPU(impl_->mgr);
        cu::deallocGPU(impl_->primitives);
        cu::deallocGPU(impl_->aabbs);
        cu::deallocGPU(impl_->metadatas);
        cu::deallocGPU(impl_->polygonDatas);
        cu::deallocGPU(impl_->facePlanes);
        cu::deallocGPU(impl_->edgeDatas);
        cu::deallocGPU(impl_->halfEdges);
        cu::deallocGPU(impl_->vertices);
#endif
    } break;
    }
}

PhysicsLoader::PhysicsLoader(PhysicsLoader &&o) = default;

HeapArray<PhysicsLoader::LoadedHull> PhysicsLoader::importConvexDecompFromDisk(
    const char *obj_path)
{
    auto imp_assets = imp::ImportedAssets::importFromDisk(obj_path);
    if (!imp_assets.has_value()) {
        FATAL("Failed to load collision mesh from %s", obj_path);
    }

    if (imp_assets->objects.size() != 1) {
        FATAL("Collision mesh source file should only have 1 object");
    }

    const auto &imp_obj = imp_assets->objects[0];

    CountT num_meshes = imp_obj.meshes.size();
    HeapArray<LoadedHull> loaded_hulls(num_meshes);

    for (CountT mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
        const imp::SourceMesh &imp_mesh = imp_obj.meshes[mesh_idx];

        math::AABB aabb = math::AABB::point(imp_mesh.positions[0]);
        for (CountT vert_idx = 1; vert_idx < (CountT)imp_mesh.numVertices;
             vert_idx++) {
            aabb.expand(imp_mesh.positions[vert_idx]);
        }

        geometry::HalfEdgeMesh he_mesh;
        he_mesh.construct(imp_mesh.positions, imp_mesh.numVertices,
                          imp_mesh.indices, imp_mesh.faceCounts,
                          imp_mesh.numFaces);

        loaded_hulls.insert(mesh_idx, {
            aabb,
            he_mesh,
        });
    }

    return loaded_hulls;
}

CountT PhysicsLoader::loadObjects(
    const RigidBodyMetadata *metadatas,
    const math::AABB *aabbs,
    const CollisionPrimitive *primitives_original,
    CountT num_objs)
{
    CountT cur_offset = impl_->curLoadedObjs;
    impl_->curLoadedObjs += num_objs;
    assert(impl_->curLoadedObjs <= impl_->maxObjs);

    size_t num_metadata_bytes = sizeof(RigidBodyMetadata) * num_objs;
    size_t num_aabb_bytes = sizeof(math::AABB) * num_objs;
    size_t num_prim_bytes = sizeof(CollisionPrimitive) * num_objs;

    RigidBodyMetadata *metadatas_dst = &impl_->metadatas[cur_offset];
    math::AABB *aabbs_dst = &impl_->aabbs[cur_offset];
    CollisionPrimitive *prims_dst = &impl_->primitives[cur_offset];

    CollisionPrimitive *primitives = (CollisionPrimitive *)malloc(sizeof(CollisionPrimitive) * num_objs);
    memcpy(primitives, primitives_original, sizeof(CollisionPrimitive) * num_objs);

    // FIXME: This function seems to leak all the pre-compaction mesh memory
    // compaction

    switch (impl_->storageType) {
    case StorageType::CPU: {
        for (int i = 0; i < num_objs; ++i) {
            if (primitives[i].type == CollisionPrimitive::Type::Hull) {
                auto &hEdgeMesh = primitives[i].hull.halfEdgeMesh;
                memcpy(
                    impl_->polygonDatas + impl_->polygonCount,
                    hEdgeMesh.mPolygons,
                    sizeof(geometry::PolygonData) * hEdgeMesh.mPolygonCount);
                hEdgeMesh.mPolygons = impl_->polygonDatas + impl_->polygonCount;

                memcpy(
                    impl_->facePlanes + impl_->polygonCount,
                    hEdgeMesh.mFacePlanes,
                    sizeof(geometry::Plane) * hEdgeMesh.mPolygonCount);
                hEdgeMesh.mFacePlanes =
                    impl_->facePlanes + impl_->polygonCount;

                impl_->polygonCount += hEdgeMesh.mPolygonCount;

                memcpy(
                    impl_->edgeDatas + impl_->edgeCount,
                    hEdgeMesh.mEdges,
                    sizeof(geometry::EdgeData) * hEdgeMesh.mEdgeCount);
                hEdgeMesh.mEdges = impl_->edgeDatas + impl_->edgeCount;
                impl_->edgeCount += hEdgeMesh.mEdgeCount;

                memcpy(
                    impl_->halfEdges + impl_->halfEdgeCount,
                    hEdgeMesh.mHalfEdges,
                    sizeof(geometry::HalfEdge) * hEdgeMesh.mHalfEdgeCount);
                hEdgeMesh.mHalfEdges = impl_->halfEdges + impl_->halfEdgeCount;
                impl_->halfEdgeCount += hEdgeMesh.mHalfEdgeCount;

                memcpy(
                    impl_->vertices + impl_->vertexCount,
                    hEdgeMesh.mVertices,
                    sizeof(math::Vector3) * hEdgeMesh.mVertexCount);
                hEdgeMesh.mVertices = impl_->vertices + impl_->vertexCount;
                impl_->vertexCount += hEdgeMesh.mVertexCount;
            }
        }

        memcpy(metadatas_dst, metadatas, num_metadata_bytes);
        memcpy(aabbs_dst, aabbs, num_aabb_bytes);
        memcpy(prims_dst, primitives, num_prim_bytes);
    } break;
    case StorageType::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        noCUDA();
#else
        for (int i = 0; i < num_objs; ++i) {
            if (primitives[i].type == CollisionPrimitive::Type::Hull) {
                auto &hEdgeMesh = primitives[i].hull.halfEdgeMesh;

                cudaMemcpy(
                    impl_->polygonDatas + impl_->polygonCount,
                    hEdgeMesh.mPolygons,
                    sizeof(geometry::PolygonData) * hEdgeMesh.mPolygonCount,
                    cudaMemcpyHostToDevice);
                hEdgeMesh.mPolygons = impl_->polygonDatas + impl_->polygonCount;

                cudaMemcpy(
                    impl_->facePlanes + impl_->polygonCount,
                    hEdgeMesh.mFacePlanes,
                    sizeof(geometry::Plane) * hEdgeMesh.mPolygonCount,
                    cudaMemcpyHostToDevice);
                hEdgeMesh.mFacePlanes =
                    impl_->facePlanes + impl_->polygonCount;
                impl_->polygonCount += hEdgeMesh.mPolygonCount;

                cudaMemcpy(
                    impl_->edgeDatas + impl_->edgeCount,
                    hEdgeMesh.mEdges,
                    sizeof(geometry::EdgeData) * hEdgeMesh.mEdgeCount,
                    cudaMemcpyHostToDevice);
                hEdgeMesh.mEdges = impl_->edgeDatas + impl_->edgeCount;
                impl_->edgeCount += hEdgeMesh.mEdgeCount;

                cudaMemcpy(
                    impl_->halfEdges + impl_->halfEdgeCount,
                    hEdgeMesh.mHalfEdges,
                    sizeof(geometry::HalfEdge) * hEdgeMesh.mHalfEdgeCount,
                    cudaMemcpyHostToDevice);
                hEdgeMesh.mHalfEdges = impl_->halfEdges + impl_->halfEdgeCount;
                impl_->halfEdgeCount += hEdgeMesh.mHalfEdgeCount;

                cudaMemcpy(
                    impl_->vertices + impl_->vertexCount,
                    hEdgeMesh.mVertices,
                    sizeof(math::Vector3) * hEdgeMesh.mVertexCount,
                    cudaMemcpyHostToDevice);
                hEdgeMesh.mVertices = impl_->vertices + impl_->vertexCount;
                impl_->vertexCount += hEdgeMesh.mVertexCount;
            }
        }

        cudaMemcpy(metadatas_dst, metadatas, num_metadata_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(aabbs_dst, aabbs, num_aabb_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(prims_dst, primitives, num_prim_bytes,
                   cudaMemcpyHostToDevice);
#endif
    } break;
    default: __builtin_unreachable();
    }

    free(primitives);

    return cur_offset;
}

ObjectManager & PhysicsLoader::getObjectManager()
{
    return *impl_->mgr;
}

}
}

