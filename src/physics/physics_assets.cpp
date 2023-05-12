#include <madrona/physics_assets.hpp>
#include <madrona/importer.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <unordered_map>

namespace madrona::phys {
using namespace geometry;

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
    PolygonData *polygonDatas;
    Plane *facePlanes;
    EdgeData *edgeDatas;
    HalfEdge *halfEdges;
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
            sizeof(PolygonData) * max_objects * max_polygons_per_object; 

        size_t num_face_plane_bytes =
            sizeof(Plane) * max_objects * max_polygons_per_object; 

        size_t num_edges_bytes =
            sizeof(EdgeData) * max_objects * max_edges_per_object; 

        size_t num_half_edges_bytes =
            sizeof(HalfEdge) * max_objects * max_half_edges_per_object; 

        RigidBodyMetadata *metadata_ptr;
        math::AABB *aabb_ptr;
        CollisionPrimitive *primitives;
        PolygonData *polygonDatas_ptr;
        Plane *facePlanes_ptr;
        EdgeData *edgeDatas_ptr;
        HalfEdge *halfEdges_ptr;
        math::Vector3 *vertices_ptr;

        ObjectManager *mgr;

        switch (storage_type) {
        case StorageType::CPU: {
            metadata_ptr =
                (RigidBodyMetadata *)malloc(num_metadata_bytes);
            aabb_ptr = (math::AABB *)malloc(num_aabb_bytes);
            primitives = (CollisionPrimitive *)malloc(num_primitive_bytes);

            polygonDatas_ptr = (PolygonData *)malloc(num_polygon_bytes);
            facePlanes_ptr = (Plane *)malloc(num_face_plane_bytes);
            edgeDatas_ptr = (EdgeData *)malloc(num_edges_bytes);
            halfEdges_ptr = (HalfEdge *)malloc(num_half_edges_bytes);
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

            polygonDatas_ptr = (PolygonData *)cu::allocGPU(num_polygon_bytes);
            facePlanes_ptr = (Plane *)cu::allocGPU(num_face_plane_bytes);
            edgeDatas_ptr = (EdgeData *)cu::allocGPU(num_edges_bytes);
            halfEdges_ptr = (HalfEdge *)cu::allocGPU(num_half_edges_bytes);
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

// FIXME: better allocation strategy
static void freeHalfEdgeMesh(HalfEdgeMesh &mesh)
{
    free(mesh.halfEdges);
    free(mesh.faceBaseHalfEdges);
    free(mesh.facePlanes);
    free(mesh.vertices);
}

static inline HalfEdgeMesh buildHalfEdgeMesh(
    const math::Vector3 *vert_positions,
    CountT num_vertices, 
    const uint32_t *indices,
    const uint32_t *face_counts,
    CountT num_faces)
{
    using namespace madrona::math;

    uint32_t num_hedges = 0;
    for (CountT face_idx = 0; face_idx < num_faces; face_idx++) {
        num_hedges += face_counts[face_idx];
    }

    assert(num_hedges % 2 == 0);

    // We already know how many polygons there are
    auto face_base_hedges =
        (uint32_t *)malloc(sizeof(uint32_t) * num_faces);
    auto hedges = (HalfEdge *)malloc(sizeof(HalfEdge) * num_hedges);
    auto face_planes = (Plane *)malloc(sizeof(Plane) * num_faces);
    auto positions =
        (math::Vector3 *)malloc(sizeof(math::Vector3) * num_vertices);
    memcpy(positions, vert_positions, sizeof(math::Vector3) * num_vertices);

    std::unordered_map<uint64_t, uint32_t> edge_to_hedge;

    auto makeEdgeID = [](uint32_t a_idx, uint32_t b_idx) {
        return ((uint64_t)a_idx << 32) | (uint64_t)b_idx;
    };

    CountT num_assigned_hedges = 0;
    const uint32_t *cur_face_indices = indices;
    for (CountT face_idx = 0; face_idx < num_faces; face_idx++) {
        CountT num_face_vertices = face_counts[face_idx];
        for (CountT vert_offset = 0; vert_offset < num_face_vertices;
             vert_offset++) {
            uint32_t a_idx = cur_face_indices[vert_offset];
            uint32_t b_idx = cur_face_indices[
                (vert_offset + 1) % num_face_vertices];

            uint64_t cur_edge_id = makeEdgeID(a_idx, b_idx);

            auto cur_edge_lookup = edge_to_hedge.find(cur_edge_id);
            if (cur_edge_lookup == edge_to_hedge.end()) {
                uint32_t cur_hedge_id = num_assigned_hedges;
                uint32_t twin_hedge_id = num_assigned_hedges + 1;

                num_assigned_hedges += 2;

                uint64_t twin_edge_id = makeEdgeID(b_idx, a_idx);

                auto [new_edge_iter, inserted] =
                    edge_to_hedge.emplace(cur_edge_id, cur_hedge_id);
                assert(inserted);

                auto [_, inserted] =
                    edge_to_hedge.emplace(twin_edge_id, twin_hedge_id)
                assert(inserted);

                cur_edge_lookup = new_edge_iter;
            }

            uint32_t hedge_idx = edge_lookup->second;
            if (vert_offset == 0) {
                face_base_hedges[face_idx] = hedge_idx;
            }

            uint32_t c_idx = cur_face_indices[
                (vert_offset + 2) % num_face_vertices];

            auto next_edge_id = makeEdgeID(b_idx, c_idx);
            auto next_edge_lookup = edge_to_hedge.find(next_edge_id);

            // If next doesn't exist yet, we can assume it will be the next
            // allocated half edge
            uint32_t next_hedge_idx == edge_to_hedge.end() ?
                num_assigned_hedges ? next_edge_lookup->second;

            hedges[hedge_idx] = HalfEdge {
                .next = next_hedge_idx,
                .rootVertex = a_idx,
                .face = face_idx,
            };
        }

        Vector3 base_pos = positions[cur_face_indices[0]];
        Vector3 e01 = positions[cur_face_indices[1]] - base_pos;
        Vector3 e02 = positions[cur_face_indices[2]] - base_pos;

        Vector3 n = math::cross(e01, e02).normalize();

        face_planes[face_idx] = Plane {
            n,
            dot(n, base_pos),
        };

        cur_face_indices += num_face_vertices;
    }

    assert(num_assigned_hedges == num_hedges);

    return HalfEdgeMesh {
        hedges,
        face_base_hedges,
        face_planes,
        positions,
        num_hedges,
        num_faces,
        num_vertices,
    };
}

static inline HalfEdgeMesh mergeCoplanarFaces(
    const HalfEdgeMesh &src_mesh)
{
    constexpr float tolerance = 1e-5;
    constexpr uint32_t sentinel = 0xFFFF'FFFF;

    using namespace geometry;
    using namespace math;

    auto new_hedges = (HalfEdge *)malloc(
        sizeof(HalfEdge) * (src_mesh.numHalfEdges));

    auto new_face_base_hedges = (uint32_t *)malloc(
        sizeof(uint32_t) * src_mesh.numHalfEdges);

    auto new_faceplanes = (Plane *)malloc(
        sizeof(Plane) * src_mesh.numHalfEdges);

    auto new_vertices = (Vector3 *)malloc(
        sizeof(Vector3) * src_mesh.numVertices);

    memcpy(new_vertices, src_mesh.vertices,
           src_mesh.numVertices * sizeof(Vector3));

    HeapArray<uint32_t> new_hedge_idxs(src_mesh.numHalfEdges);
    HeapArray<uint32_t> next_remap(src_mesh.numHalfEdges);
    for (CountT i = 0; i < new_hedge_idxs.size(); i++) {
        new_hedge_idxs[i] = 0xFFFF'FFFF;
        next_remap[i] = i;
    }

    HeapArray<bool> faces_merged(src_mesh.numFaces);
    for (CountT i = 0; i < faces_merged.size(); i++) {
        faces_merged[i] = false;
    }

    CountT num_new_hedges = 0;
    CountT num_new_faces = 0;

    for (uint32_t orig_face_idx = 0; orig_face_idx < src_mesh.numFaces;
         orig_face_idx++) {
        if (faces_merged[orig_face_idx]) {
            continue;
        }

        uint32_t new_face_idx = num_new_faces++;
        Plane face_plane = src_mesh.facePlanes[orig_face_idx];
        new_faceplanes[new_face_idx] = face_plane;

        uint32_t face_start_hedge = faceBaseHalfEdges[orig_face_idx];

        // To avoid special casing, initial prev is set to the ID
        // of the next half edge to be assigned. The correct next will be
        // written after the loop completes.
        uint32_t prev_new_hedge_idx = num_new_hedges;
        uint32_t cur_hedge_idx = face_start_hedge;
        uint32_t new_face_root = sentinel;
        do {
            // If we wind up back at the same half edge twice, there is a
            // problem. This ensures that following the same next pointer twice
            // will trigger an assert
            next_remap[cur_hedge_idx] = sentinel;
            assert(cur_hedge_idx != sentinel);

            uint32_t twin_hedge_idx = src_mesh.twinIDX(cur_hedge_idx);

            const HalfEdge &cur_hedge = src_mesh.halfEdges[cur_hedge_idx];
            const HalfEdge &twin_hedge = src_mesh.halfEdges[twin_hedge_idx];

            Vector3 cur_normal =
                src_mesh.facePlanes[cur_hedge.face].normal;
            Vector3 twin_normal =
                src_mesh.facePlanes[twin_hedge.face].normal;

            if (dot(cur_normal, twin_normal) >= 1.f - tolerance) {
                faces_merged[twin_hedge.face] = true;
                next_remap[twin_hedge_idx] = cur_hedge.next;
                
                cur_hedge_idx = twin_hedge.next;
                continue;
            }

            uint32_t new_hedge_idx = new_hedge_idxs[cur_hedge_idx];
            if (new_hedge_idx == sentinel) {
                new_hedge_idx = num_new_hedges;
                new_hedge_idxs[cur_hedge_idx] = new_hedge_idx;
                new_hedge_idxs[twin_hedge_idx] = new_hedge_idx + 1;
                num_new_hedges += 2;
            }

            if (new_face_root == sentinel) {
                new_face_root = new_hedge_idx;
            }

            new_hedges[new_hedge_idx] = HalfEdge {
                .next = 0,
                .rootVertex = cur_hedge.rootVertex,
                .face = new_face_idx,
            };

            new_hedges[prev_new_hedge_idx].next = new_hedge_idx;

            cur_hedge_idx = next_remap[cur_hedge.next];
            prev_new_hedge_idx = new_hedge_idx;
        } while (cur_hedge_idx != face_start_hedge);

        // Set final next link in loop
        new_hedges[prev_new_hedge_idx].next = new_face_root;
        new_face_base_hedges[new_face_idx] = new_face_root;
    }

    assert(num_new_faces > 0);

    // FIXME: the above code has two issues:
    // 1) It can orphan vertices. These shuold be filtered out in a final pass
    // 2) There is some tolerance in the normal, which means face vertices may
    // not actually form a perfect plane. Worth trying to correct errors?

    return HalfEdgeMesh {
        .halfEdges = new_hedges,
        .faceBaseHalfEdges = new_face_base_hedges,
        .facePlanes = new_faceplanes,
        .vertices = new_vertices,
        .numHalfEdges = num_new_hedges,
        .numFaces = num_new_faces,
        .numVertices = num_vertices,
    };
}

PhysicsLoader::ConvexDecompositions PhysicsLoader::processConvexDecompositions(
    const imp::SourceObject *src_objs,
    const float *inv_masses,
    CountT num_objects,
    bool merge_coplanar_faces)
{
    CountT total_num_vertices = 0;

    for (CountT obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        const imp::SourceObject &src_obj = src_objs[obj_idx];

        for (const auto &mesh : src_obj.meshes) {
            total_num_vertices += mesh.numVertices;
        }
    }

    HeapArray<math::Vector3> all_verts(total_num_vertices);

    CountT num_meshes = src_obj.meshes.size();
    HeapArray<LoadedHull> loaded_hulls(num_meshes);

    for (CountT mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
        const imp::SourceMesh &src_mesh = src_obj.meshes[mesh_idx];

        math::AABB aabb = math::AABB::point(src_mesh.positions[0]);
        for (CountT vert_idx = 1; vert_idx < (CountT)src_mesh.numVertices;
             vert_idx++) {
            aabb.expand(src_mesh.positions[vert_idx]);
        }

        HalfEdgeMesh he_mesh = buildHalfEdgeMesh(src_mesh.positions, 
            src_mesh.numVertices, src_mesh.indices, src_mesh.faceCounts,
            src_mesh.numFaces);

        if (merge_coplanar_faces) {
            HalfEdgeMesh merged_mesh = mergeCoplanarFaces(he_mesh);

        } else {
            loaded_hulls.insert(mesh_idx, {
                aabb,
                he_mesh,
            });
        }
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
                    sizeof(PolygonData) * hEdgeMesh.mPolygonCount);
                hEdgeMesh.mPolygons = impl_->polygonDatas + impl_->polygonCount;

                memcpy(
                    impl_->facePlanes + impl_->polygonCount,
                    hEdgeMesh.mFacePlanes,
                    sizeof(Plane) * hEdgeMesh.mPolygonCount);
                hEdgeMesh.mFacePlanes =
                    impl_->facePlanes + impl_->polygonCount;

                impl_->polygonCount += hEdgeMesh.mPolygonCount;

                memcpy(
                    impl_->edgeDatas + impl_->edgeCount,
                    hEdgeMesh.mEdges,
                    sizeof(EdgeData) * hEdgeMesh.mEdgeCount);
                hEdgeMesh.mEdges = impl_->edgeDatas + impl_->edgeCount;
                impl_->edgeCount += hEdgeMesh.mEdgeCount;

                memcpy(
                    impl_->halfEdges + impl_->halfEdgeCount,
                    hEdgeMesh.mHalfEdges,
                    sizeof(HalfEdge) * hEdgeMesh.mHalfEdgeCount);
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
                    sizeof(PolygonData) * hEdgeMesh.mPolygonCount,
                    cudaMemcpyHostToDevice);
                hEdgeMesh.mPolygons = impl_->polygonDatas + impl_->polygonCount;

                cudaMemcpy(
                    impl_->facePlanes + impl_->polygonCount,
                    hEdgeMesh.mFacePlanes,
                    sizeof(Plane) * hEdgeMesh.mPolygonCount,
                    cudaMemcpyHostToDevice);
                hEdgeMesh.mFacePlanes =
                    impl_->facePlanes + impl_->polygonCount;
                impl_->polygonCount += hEdgeMesh.mPolygonCount;

                cudaMemcpy(
                    impl_->edgeDatas + impl_->edgeCount,
                    hEdgeMesh.mEdges,
                    sizeof(EdgeData) * hEdgeMesh.mEdgeCount,
                    cudaMemcpyHostToDevice);
                hEdgeMesh.mEdges = impl_->edgeDatas + impl_->edgeCount;
                impl_->edgeCount += hEdgeMesh.mEdgeCount;

                cudaMemcpy(
                    impl_->halfEdges + impl_->halfEdgeCount,
                    hEdgeMesh.mHalfEdges,
                    sizeof(HalfEdge) * hEdgeMesh.mHalfEdgeCount,
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

