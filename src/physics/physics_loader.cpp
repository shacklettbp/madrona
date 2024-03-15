#include <madrona/physics_loader.hpp>
#include <madrona/importer.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <unordered_map>

namespace madrona::phys {
using namespace geo;
using namespace math;

#ifndef MADRONA_CUDA_SUPPORT
[[noreturn]] static void noCUDA()
{
    FATAL("PhysicsLoader: Not built with CUDA support");
}
#endif

struct PhysicsLoader::Impl {
    CollisionPrimitive *primitives;
    AABB *primAABBs;

    AABB *objAABBs;
    uint32_t *rigidBodyPrimitiveOffsets;
    uint32_t *rigidBodyPrimitiveCounts;
    RigidBodyMetadata *metadatas;

    CountT curPrimOffset;
    CountT curObjOffset;

    ObjectManager *mgr;
    CountT maxPrims;
    CountT maxObjs;
    ExecMode execMode;

    static Impl * init(ExecMode exec_mode, CountT max_objects)
    {
        constexpr CountT max_prims_per_object = 20;

        size_t num_collision_prim_bytes =
            sizeof(CollisionPrimitive) * max_objects * max_prims_per_object; 

        size_t num_collision_aabb_bytes =
            sizeof(AABB) * max_objects * max_prims_per_object; 

        size_t num_obj_aabb_bytes =
            sizeof(AABB) * max_objects;

        size_t num_offset_bytes =
            sizeof(uint32_t) * max_objects;

        size_t num_count_bytes =
            sizeof(uint32_t) * max_objects;

        size_t num_metadata_bytes =
            sizeof(RigidBodyMetadata) * max_objects;

        CollisionPrimitive *primitives_ptr;
        AABB *prim_aabb_ptr;

        AABB *obj_aabb_ptr;
        uint32_t *offsets_ptr;
        uint32_t *counts_ptr;
        RigidBodyMetadata *metadata_ptr;

        ObjectManager *mgr;

        switch (exec_mode) {
            case ExecMode::CPU: {
                primitives_ptr = (CollisionPrimitive *)malloc(
                    num_collision_prim_bytes);

                prim_aabb_ptr = (AABB *)malloc(
                    num_collision_aabb_bytes);

                obj_aabb_ptr = (AABB *)malloc(num_obj_aabb_bytes);

            offsets_ptr = (uint32_t *)malloc(num_offset_bytes);
            counts_ptr = (uint32_t *)malloc(num_count_bytes);

            metadata_ptr =
                (RigidBodyMetadata *)malloc(num_metadata_bytes);

            mgr = new ObjectManager {
                primitives_ptr,
                prim_aabb_ptr,
                obj_aabb_ptr,
                offsets_ptr,
                counts_ptr,
                metadata_ptr,
            };
        } break;
        case ExecMode::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
            noCUDA();
#else
            primitives_ptr = (CollisionPrimitive *)cu::allocGPU(
                num_collision_prim_bytes);

            prim_aabb_ptr = (AABB *)cu::allocGPU(
                num_collision_aabb_bytes);

            obj_aabb_ptr = (AABB *)cu::allocGPU(num_obj_aabb_bytes);

            offsets_ptr = (uint32_t *)cu::allocGPU(num_offset_bytes);
            counts_ptr = (uint32_t *)cu::allocGPU(num_count_bytes);

            metadata_ptr =
                (RigidBodyMetadata *)cu::allocGPU(num_metadata_bytes);

            mgr = (ObjectManager *)cu::allocGPU(sizeof(ObjectManager));

            ObjectManager local {
                primitives_ptr,
                prim_aabb_ptr,
                obj_aabb_ptr,
                offsets_ptr,
                counts_ptr,
                metadata_ptr,
            };

            REQ_CUDA(cudaMemcpy(mgr, &local, sizeof(ObjectManager),
                                cudaMemcpyHostToDevice));
#endif
        } break;
        default: MADRONA_UNREACHABLE();
        }

        return new Impl {
            .primitives = primitives_ptr,
            .primAABBs = prim_aabb_ptr,
            .objAABBs = obj_aabb_ptr,
            .rigidBodyPrimitiveOffsets = offsets_ptr,
            .rigidBodyPrimitiveCounts = counts_ptr,
            .metadatas = metadata_ptr,
            .curPrimOffset = 0,
            .curObjOffset = 0,
            .mgr = mgr,
            .maxPrims = max_objects * max_prims_per_object,
            .maxObjs = max_objects,
            .execMode = exec_mode,
        };
    }
};

PhysicsLoader::PhysicsLoader(ExecMode exec_mode, CountT max_objects)
    : impl_(Impl::init(exec_mode, max_objects))
{}

PhysicsLoader::~PhysicsLoader()
{
    if (impl_ == nullptr) {
        return;
    }

    switch (impl_->execMode) {
    case ExecMode::CPU: {
        delete impl_->mgr;
        free(impl_->primitives);
        free(impl_->primAABBs);
        free(impl_->objAABBs);
        free(impl_->rigidBodyPrimitiveOffsets);
        free(impl_->rigidBodyPrimitiveCounts);
        free(impl_->metadatas);
    } break;
    case ExecMode::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        noCUDA();
#else
        cu::deallocGPU(impl_->primitives);
        cu::deallocGPU(impl_->primAABBs);
        cu::deallocGPU(impl_->objAABBs);
        cu::deallocGPU(impl_->rigidBodyPrimitiveOffsets);
        cu::deallocGPU(impl_->rigidBodyPrimitiveCounts);
        cu::deallocGPU(impl_->metadatas);
#endif
    } break;
    }
}

PhysicsLoader::PhysicsLoader(PhysicsLoader &&o) = default;

CountT PhysicsLoader::loadRigidBodies(const RigidBodyAssets &assets)
{
    CountT cur_obj_offset = impl_->curObjOffset;
    impl_->curObjOffset += assets.numObjs;
    CountT cur_prim_offset = impl_->curPrimOffset;
    impl_->curPrimOffset += assets.totalNumPrimitives;
    assert(impl_->curObjOffset <= impl_->maxObjs);
    assert(impl_->curPrimOffset <= impl_->maxPrims);

    CollisionPrimitive *prims_dst = &impl_->primitives[cur_prim_offset];
    AABB *prim_aabbs_dst = &impl_->primAABBs[cur_prim_offset];

    AABB *obj_aabbs_dst = &impl_->objAABBs[cur_obj_offset];
    uint32_t *offsets_dst = &impl_->rigidBodyPrimitiveOffsets[cur_obj_offset];
    uint32_t *counts_dst = &impl_->rigidBodyPrimitiveCounts[cur_obj_offset];
    RigidBodyMetadata *metadatas_dst = &impl_->metadatas[cur_obj_offset];

    // FIXME: redo all this, leaks memory, slow, etc. Very non optimal on the
    // CPU.

    uint32_t *offsets_tmp = (uint32_t *)malloc(
        sizeof(uint32_t) * assets.numObjs);
    for (CountT i = 0; i < (CountT)assets.numObjs; i++) {
        offsets_tmp[i] = assets.primOffsets[i] + cur_prim_offset;
    }

    HalfEdge *hull_halfedges;
    uint32_t *hull_face_base_halfedges;
    Plane *hull_face_planes;
    Vector3 *hull_verts;
    switch (impl_->execMode) {
    case ExecMode::CPU: {
        memcpy(prim_aabbs_dst, assets.primitiveAABBs,
               sizeof(AABB) * assets.totalNumPrimitives);

        memcpy(obj_aabbs_dst, assets.objAABBs,
               sizeof(AABB) * assets.numObjs);
        memcpy(offsets_dst, offsets_tmp,
               sizeof(uint32_t) * assets.numObjs);
        memcpy(counts_dst, assets.primCounts,
               sizeof(uint32_t) * assets.numObjs);
        memcpy(metadatas_dst, assets.metadatas,
               sizeof(RigidBodyMetadata) * assets.numObjs);

        hull_halfedges = (HalfEdge *)malloc(
            sizeof(HalfEdge) * assets.hullData.numHalfEdges);
        hull_face_base_halfedges = (uint32_t *)malloc(
            sizeof(uint32_t) * assets.hullData.numFaces);
        hull_face_planes = (Plane *)malloc(
            sizeof(Plane) * assets.hullData.numFaces);
        hull_verts = (Vector3 *)malloc(
            sizeof(Vector3) * assets.hullData.numVerts);

        memcpy(hull_halfedges, assets.hullData.halfEdges,
               sizeof(HalfEdge) * assets.hullData.numHalfEdges);
        memcpy(hull_face_base_halfedges, assets.hullData.faceBaseHalfEdges,
               sizeof(uint32_t) * assets.hullData.numFaces);
        memcpy(hull_face_planes, assets.hullData.facePlanes,
               sizeof(Plane) * assets.hullData.numFaces);
        memcpy(hull_verts, assets.hullData.vertices,
               sizeof(Vector3) * assets.hullData.numVerts);
    } break;
    case ExecMode::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        noCUDA();
#else
        cudaMemcpy(prim_aabbs_dst, assets.primitiveAABBs,
                   sizeof(AABB) * assets.totalNumPrimitives,
                   cudaMemcpyHostToDevice);

        cudaMemcpy(obj_aabbs_dst, assets.objAABBs,
                   sizeof(AABB) * assets.numObjs,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(offsets_dst, offsets_tmp,
                   sizeof(uint32_t) * assets.numObjs,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(counts_dst, assets.primCounts,
                   sizeof(uint32_t) * assets.numObjs,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(metadatas_dst, assets.metadatas,
                   sizeof(RigidBodyMetadata) * assets.numObjs,
                   cudaMemcpyHostToDevice);

        hull_halfedges = (HalfEdge *)cu::allocGPU(
            sizeof(HalfEdge) * assets.hullData.numHalfEdges);
        hull_face_base_halfedges = (uint32_t *)cu::allocGPU(
            sizeof(uint32_t) * assets.hullData.numFaces);
        hull_face_planes = (Plane *)cu::allocGPU(
            sizeof(Plane) * assets.hullData.numFaces);
        hull_verts = (Vector3 *)cu::allocGPU(
            sizeof(Vector3) * assets.hullData.numVerts);

        cudaMemcpy(hull_halfedges, assets.hullData.halfEdges,
                   sizeof(HalfEdge) * assets.hullData.numHalfEdges,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(hull_face_base_halfedges, assets.hullData.faceBaseHalfEdges,
                   sizeof(uint32_t) * assets.hullData.numFaces,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(hull_face_planes, assets.hullData.facePlanes,
                   sizeof(Plane) * assets.hullData.numFaces,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(hull_verts, assets.hullData.vertices,
                   sizeof(Vector3) * assets.hullData.numVerts,
                   cudaMemcpyHostToDevice);
#endif
    }
    }

    auto primitives_tmp = (CollisionPrimitive *)malloc(
        sizeof(CollisionPrimitive) * assets.totalNumPrimitives);
    memcpy(primitives_tmp, assets.primitives,
           sizeof(CollisionPrimitive) * assets.totalNumPrimitives);

    for (CountT i = 0; i < (CountT)assets.totalNumPrimitives; i++) {
        CollisionPrimitive &cur_primitive = primitives_tmp[i];
        if (cur_primitive.type != CollisionPrimitive::Type::Hull) continue;

        HalfEdgeMesh &he_mesh = cur_primitive.hull.halfEdgeMesh;

        // FIXME: incoming HalfEdgeMeshes should have offsets or something
        CountT hedge_offset = he_mesh.halfEdges - assets.hullData.halfEdges;
        CountT face_offset =
            he_mesh.facePlanes - assets.hullData.facePlanes;
        CountT vert_offset = he_mesh.vertices - assets.hullData.vertices;

        he_mesh.halfEdges = hull_halfedges + hedge_offset;
        he_mesh.faceBaseHalfEdges = hull_face_base_halfedges + face_offset;
        he_mesh.facePlanes = hull_face_planes + face_offset;
        he_mesh.vertices = hull_verts + vert_offset;
    }

    switch (impl_->execMode) {
    case ExecMode::CPU: {
        memcpy(prims_dst, primitives_tmp,
               sizeof(CollisionPrimitive) * assets.totalNumPrimitives);
    } break;
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(prims_dst, primitives_tmp,
            sizeof(CollisionPrimitive) * assets.totalNumPrimitives,
            cudaMemcpyHostToDevice);
#endif
    } break;
    }

    free(primitives_tmp);
    free(offsets_tmp);

    return cur_obj_offset;
}

ObjectManager & PhysicsLoader::getObjectManager()
{
    return *impl_->mgr;
}

}
