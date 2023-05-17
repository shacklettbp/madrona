#include <madrona/physics_assets.hpp>
#include <madrona/importer.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <unordered_map>

namespace madrona::phys {
using namespace geometry;
using namespace math;
using SourceCollisionPrimitive = PhysicsLoader::SourceCollisionPrimitive;
using SourceCollisionObject = PhysicsLoader::SourceCollisionObject;
using ImportedRigidBodies = PhysicsLoader::ImportedRigidBodies;

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
    StorageType storageType;

    static Impl * init(StorageType storage_type, CountT max_objects)
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

        switch (storage_type) {
            case StorageType::CPU: {
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
        case StorageType::CUDA: {
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
        default: __builtin_unreachable();
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
        free(impl_->primAABBs);
        free(impl_->objAABBs);
        free(impl_->rigidBodyPrimitiveOffsets);
        free(impl_->rigidBodyPrimitiveCounts);
        free(impl_->metadatas);
    } break;
    case StorageType::CUDA: {
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

// FIXME: better allocation / ownership strategy
static void freeHalfEdgeMesh(HalfEdgeMesh &mesh)
{
    free(mesh.halfEdges);
    free(mesh.faceBaseHalfEdges);
    free(mesh.facePlanes);
    free(mesh.vertices);
}

PhysicsLoader::ImportedRigidBodies::~ImportedRigidBodies()
{
    // FIXME: change halfEdgeMesh data ownership
    for (CollisionPrimitive &prim : collisionPrimitives) {
        if (prim.type == CollisionPrimitive::Type::Hull) {
            freeHalfEdgeMesh(prim.hull.halfEdgeMesh);
        }
    }
}

static inline HalfEdgeMesh buildHalfEdgeMesh(
    const Vector3 *vert_positions,
    CountT num_vertices, 
    const uint32_t *indices,
    const uint32_t *face_counts,
    CountT num_faces)
{
    auto numFaceVerts = [face_counts](CountT face_idx) {
        if (face_counts == nullptr) {
            return 3_u32;
        } else {
            return face_counts[face_idx];
        }
    };

    using namespace madrona::math;

    uint32_t num_hedges = 0;
    for (CountT face_idx = 0; face_idx < num_faces; face_idx++) {
        num_hedges += numFaceVerts(face_idx);
    }

    assert(num_hedges % 2 == 0);

    // We already know how many polygons there are
    auto face_base_hedges =
        (uint32_t *)malloc(sizeof(uint32_t) * num_faces);
    auto hedges = (HalfEdge *)malloc(sizeof(HalfEdge) * num_hedges);
    auto face_planes = (Plane *)malloc(sizeof(Plane) * num_faces);
    auto positions =
        (Vector3 *)malloc(sizeof(Vector3) * num_vertices);
    memcpy(positions, vert_positions, sizeof(Vector3) * num_vertices);

    std::unordered_map<uint64_t, uint32_t> edge_to_hedge;

    auto makeEdgeID = [](uint32_t a_idx, uint32_t b_idx) {
        return ((uint64_t)a_idx << 32) | (uint64_t)b_idx;
    };

    CountT num_assigned_hedges = 0;
    const uint32_t *cur_face_indices = indices;
    for (CountT face_idx = 0; face_idx < num_faces; face_idx++) {
        CountT num_face_vertices = numFaceVerts(face_idx);
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

                auto [new_edge_iter, cur_inserted] =
                    edge_to_hedge.emplace(cur_edge_id, cur_hedge_id);
                assert(cur_inserted);

                auto [new_twin_iter, twin_inserted] =
                    edge_to_hedge.emplace(twin_edge_id, twin_hedge_id);
                assert(twin_inserted);

                cur_edge_lookup = new_edge_iter;
            }

            uint32_t hedge_idx = cur_edge_lookup->second;
            if (vert_offset == 0) {
                face_base_hedges[face_idx] = hedge_idx;
            }

            uint32_t c_idx = cur_face_indices[
                (vert_offset + 2) % num_face_vertices];

            auto next_edge_id = makeEdgeID(b_idx, c_idx);
            auto next_edge_lookup = edge_to_hedge.find(next_edge_id);

            // If next doesn't exist yet, we can assume it will be the next
            // allocated half edge
            uint32_t next_hedge_idx = next_edge_lookup == edge_to_hedge.end() ?
                num_assigned_hedges : next_edge_lookup->second;

            hedges[hedge_idx] = HalfEdge {
                .next = next_hedge_idx,
                .rootVertex = a_idx,
                .face = uint32_t(face_idx),
            };
        }

        Vector3 base_pos = positions[cur_face_indices[0]];
        Vector3 e01 = positions[cur_face_indices[1]] - base_pos;
        Vector3 e02 = positions[cur_face_indices[2]] - base_pos;

        Vector3 n = cross(e01, e02).normalize();

        face_planes[face_idx] = Plane {
            n,
            dot(n, base_pos),
        };

        cur_face_indices += num_face_vertices;
    }

    assert(num_assigned_hedges == num_hedges);

    return HalfEdgeMesh {
        .halfEdges = hedges,
        .faceBaseHalfEdges = face_base_hedges,
        .facePlanes = face_planes,
        .vertices = positions,
        .numHalfEdges = uint32_t(num_hedges),
        .numFaces = uint32_t(num_faces),
        .numVertices = uint32_t(num_vertices),
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

        uint32_t face_start_hedge = src_mesh.faceBaseHalfEdges[orig_face_idx];

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
        .numHalfEdges = uint32_t(num_new_hedges),
        .numFaces = uint32_t(num_new_faces),
        .numVertices = src_mesh.numVertices,
    };
}

namespace {
struct MassProperties {
    Diag3x3 inertiaTensor;
    Vector3 centerOfMass;
    Quat toDiagonal;
};
}

// Below functions diagonalize the inertia tensor and compute the necessary
// rotation for diagonalization as a quaternion.
// Source: Computing the Singular Value Decomposition of 3x3 matrices with
// minimal branching and  elementary floating point operations.
// McAdams et al 2011

// McAdams Algorithm 2:
static Quat approxGivensQuaternion(const Mat3x3 &m)
{

    constexpr float gamma = 5.82842712474619f;
    constexpr float c_star = 0.9238795325112867f;
    constexpr float s_star = 0.3826834323650898f;

    float a11 = m[0][0], a12 = m[1][0], a22 = m[1][1];

    float ch = 2.f * (a11 - a22);
    float sh = a12;

    float sh2 = sh * sh;
    float ch2 = ch * ch;

    bool b = (gamma * sh2) < ch2;

    float omega = rsqrtApprox(ch2 + sh2);
    ch = b ? omega * ch : c_star;
    sh = b ? omega * sh : s_star;

    return Quat { ch, 0, 0, sh };
}

// Equation 12: approxGivensQuaternion returns an unscaled quaternion,
// need to rescale
static Mat3x3 jacobiIterConjugation(const Mat3x3 &m, float ch, float sh)
{
    float ch2 = ch * ch;
    float sh2 = sh * sh;
    float q_scale = ch2 + sh2;

    float q11 = (ch2 - sh2) / q_scale;
    float q12 = (-2.f * sh * ch) / q_scale;
    float q21 = (2.f * sh * ch) / q_scale;
    float q22 = (ch2 - sh2) / q_scale;

    // Output = Q^T * m * Q. Given above values for Q, direct solution to
    // compute output (given 0s for other terms) computed using SymPy

    auto [m11, m21, m31] = m[0];
    auto [m12, m22, m32] = m[1];
    auto [m13, m23, m33] = m[2];

    float m11q11_m21q21 = m11 * q11 + m21 * q21;
    float m11q12_m21q22 = m11 * q12 + m21 * q22;

    float m12q11_m22q21 = m12 * q11 + m22 * q21;
    float m12q12_m22q22 = m12 * q12 + m22 * q22;
    
    return Mat3x3 {{
        Vector3 {
            q11 * m11q11_m21q21 + q21 * m12q11_m22q21,
            q11 * m11q12_m21q22 + q21 * m12q12_m22q22,
            m31 * q11 + m32 * q21,
        },
        Vector3 {
            q12 * m11q11_m21q21 + q22 * m12q11_m22q21,
            q12 * m11q12_m21q22 + q22 * m12q12_m22q22,
            m31 * q12 + m32 * q22,
        },
        Vector3 {
            m13 * q11 + m23 * q21,
            m13 * q12 + m23 * q22,
            m33,
        },
    }};
}

// Inertia tensor is symmetric positive semi definite, so we only need to
// perform the symmetric eigenanalysis part of the algorithm.
static void diagonalizeInertiaTensor(const Mat3x3 &m,
                                     Diag3x3 *out_diag,
                                     Quat *out_rot)
{
    using namespace math;

    constexpr CountT num_jacobi_iters = 8; // Double the number in the paper

    Mat3x3 cur_mat = m;
    Quat accumulated_rot { 1, 0, 0, 0 };
    for (CountT i = 0; i < num_jacobi_iters; i++) {
        Quat cur_rot = approxGivensQuaternion(m);

        cur_mat = jacobiIterConjugation(cur_mat, cur_rot.w, cur_rot.z);

        accumulated_rot = cur_rot * accumulated_rot;
    }

    Quat final_rot = accumulated_rot.normalize();

    // Compute the diagonal (all other terms should be ~0)
    {
        Mat3x3 q = Mat3x3::fromQuat(final_rot);

        auto [m11, m21, m31] = m[0];
        auto [m12, m22, m32] = m[1];
        auto [m13, m23, m33] = m[2];

        auto [q11, q21, q31] = q[0];
        auto [q12, q22, q32] = q[1];
        auto [q13, q23, q33] = q[2];

        out_diag->d0 = q11 * (m11 * q11 + m12 * q21 + m13 * q31) +
                       q21 * (m21 * q11 + m22 * q21 + m23 * q31) +
                       q31 * (m31 * q11 + m32 * q21 + m33 * q31);

        out_diag->d1 = q12 * (m11 * q12 + m12 * q22 + m13 * q32) +
                       q22 * (m21 * q12 + m22 * q22 + m23 * q32) +
                       q32 * (m31 * q12 + m32 * q22 + m33 * q32);
        
        out_diag->d2 = q13 * (m11 * q13 + m12 * q23 + m13 * q33) +
                       q23 * (m21 * q13 + m22 * q23 + m23 * q33) +
                       q33 * (m31 * q13 + m32 * q23 + m33 * q33);
    }

    *out_rot = accumulated_rot;
}

// http://number-none.com/blow/inertia/
static inline MassProperties computeMassProperties(
    const SourceCollisionObject &src_obj)
{
    using namespace math;
    const Mat3x3 C_canonical {{
        { 1.f / 60.f, 1.f / 120.f, 1.f / 120.f },
        { 1.f / 120.f, 1.f / 60.f, 1.f / 120.f },
        { 1.f / 120.f, 1.f / 120.f, 1.f / 60.f },
    }};
    constexpr float density = 1.f;

    Mat3x3 C_total {{
        Vector3::zero(),
        Vector3::zero(),
        Vector3::zero(),
    }};

    float m_total = 0;
    Vector3 x_total = Vector3::zero();

    auto processTet = [&](Vector3 v1, Vector3 v2, Vector3 v3) {
        // Reference point is (0, 0, 0) so tet edges are just the vertex
        // positions
        Vector3 e1 = v1;
        Vector3 e2 = v2;
        Vector3 e3 = v3;

        // Covariance matrix
        Mat3x3 A {{ e1, e2, e3 }};
        float det_A = A.determinant();
        Mat3x3 C = det_A * A * C_canonical * A.transpose();

        // Mass
        float volume = 1.f / 6.f * det_A;
        float m = volume * density;

        Vector3 x = 0.25f * e1 + 0.25f * e2 + 0.25f * e3;

        // Accumulate tetrahedron properties
        float old_m_total = m_total;
        m_total += m;
        x_total = (x * m + x_total * old_m_total) / m_total;

        C_total += C;
    };

    for (const SourceCollisionPrimitive &prim : src_obj.prims) {
        if (prim.type == CollisionPrimitive::Type::Sphere) {
            // FIXME: need to allow offset for primitives
            m_total += 1.f;

            float r = prim.sphere.radius;

            // Note that we need the sphere's covariance matrix,
            // not the inertia tensor (hence 1/2 standard formulas)
            float v = 1.f / 5.f * r * r;
            C_total += Mat3x3 {{
                Vector3 { v, 0.f, 0.f },
                Vector3 { 0.f, v, 0.f },
                Vector3 { 0.f, 0.f, v },
            }};
            continue;
        } else if (prim.type == CollisionPrimitive::Type::Plane) {
            // Plane has infinite mass / inertia. The rest of the
            // object must as well

            return MassProperties {
                Diag3x3::uniform(INFINITY),
                Vector3::zero(),
                Quat { 1, 0, 0, 0 },
            };
        }

        // Hull primitive
 
        const imp::SourceMesh &src_mesh = *prim.hull.mesh;

        const uint32_t *cur_indices = src_mesh.indices;
        for (CountT face_idx = 0; face_idx < (CountT)src_mesh.numFaces;
             face_idx++) {
            CountT num_face_vertices = src_mesh.faceCounts ?
                src_mesh.faceCounts[face_idx] : 3;

            uint32_t idx1 = cur_indices[0];
            Vector3 v1 = src_mesh.positions[idx1];
            for (CountT i = 1; i < num_face_vertices - 1; i++) {
                uint32_t idx2 = cur_indices[i];
                Vector3 v2 = src_mesh.positions[idx2];

                uint32_t idx3 = cur_indices[i + 1];
                Vector3 v3 = src_mesh.positions[idx3];

                processTet(v1, v2, v3);
            }

            cur_indices += num_face_vertices;
        }
    }

    auto translateCovariance = [](const Mat3x3 &C,
                                  Vector3 x, // COM
                                  float m,
                                  Vector3 delta_x) {
        Mat3x3 term1 = outerProduct(delta_x, x);
        Mat3x3 term2 = outerProduct(x, delta_x); 
        Mat3x3 term3 = outerProduct(delta_x, delta_x);
        return C + m * (term1 + term2 + term3);
    };

    // Move accumulated covariance matrix to center of mass
    C_total = translateCovariance(C_total, x_total, m_total, -x_total);

    float tr_C = C_total[0][0] + C_total[1][1] + C_total[2][2];
    const Mat3x3 tr_C_diag {{
        Vector3 { tr_C, 0, 0 },
        Vector3 { 0, tr_C, 0 },
        Vector3 { 0, 0, tr_C },
    }};

    // Compute inertia tensor 
    Mat3x3 inertia_tensor = tr_C_diag - C_total;

    // Rescale total mass of inertia tensor (unless infinity)
    float inv_mass = 1.f / m_total;
    inertia_tensor *= inv_mass;

    printf("Inertia Tensor:\n"
           "%f %f %f\n"
           "%f %f %f\n"
           "%f %f %f\n"
           "COM: (%f %f %f) mass: %f\n", 
           inertia_tensor[0].x,
           inertia_tensor[1].x,
           inertia_tensor[2].x,
           inertia_tensor[0].y,
           inertia_tensor[1].y,
           inertia_tensor[2].y,
           inertia_tensor[0].z,
           inertia_tensor[1].z,
           inertia_tensor[2].z,
           x_total.x,
           x_total.y,
           x_total.z,
           m_total
           );

    Diag3x3 diag_inertia;
    Quat rot_to_diag;
    diagonalizeInertiaTensor(inertia_tensor, &diag_inertia, &rot_to_diag);

    printf("Diag Inertia tensor: (%f %f %f) rot: (%f %f %f %f)\n\n",
           diag_inertia.d0, diag_inertia.d1, diag_inertia.d2,
           rot_to_diag.w,
           rot_to_diag.x,
           rot_to_diag.y,
           rot_to_diag.z);

    return MassProperties {
        diag_inertia,
        x_total,
        rot_to_diag,
    };
}

static inline RigidBodyMassData toMassData(const MassProperties &mass_props,
                                           float inv_m)
{
    Diag3x3 inv_inertia = inv_m / mass_props.inertiaTensor;

    return {
        .invMass = inv_m,
        .invInertiaTensor = Vector3 { // FIXME
            inv_inertia.d0,
            inv_inertia.d1,
            inv_inertia.d2,
        },
        .toCenterOfMass = mass_props.centerOfMass,
        .toInteriaFrame = mass_props.toDiagonal,
    };
}

static void setupSpherePrimitive(const SourceCollisionPrimitive &src_prim,
                                 CollisionPrimitive *out_prim,
                                 AABB *out_aabb)
{
    out_prim->sphere = src_prim.sphere;

    const float r = src_prim.sphere.radius;

    *out_aabb = AABB {
        .pMin = { -r, -r, -r },
        .pMax = { r, r, r },
    };
}

static void setupPlanePrimitive(const SourceCollisionPrimitive &,
                                CollisionPrimitive *out_prim,
                                AABB *out_aabb)
{
    out_prim->plane = CollisionPrimitive::Plane {};
    
    *out_aabb = AABB {
        .pMin = { -FLT_MAX, -FLT_MAX, -FLT_MAX },
        .pMax = { FLT_MAX, FLT_MAX, 0 },
    };
}

static void setupHullPrimitive(const SourceCollisionPrimitive &src_prim,
                               CollisionPrimitive *out_prim,
                               AABB *out_aabb,
                               CountT *total_num_halfedges,
                               CountT *total_num_faces,
                               CountT *total_num_vertices,
                               bool merge_coplanar_faces)
{
    const imp::SourceMesh *src_mesh = src_prim.hull.mesh;

    HalfEdgeMesh he_mesh = buildHalfEdgeMesh(src_mesh->positions, 
        src_mesh->numVertices, src_mesh->indices, src_mesh->faceCounts,
        src_mesh->numFaces);

    if (merge_coplanar_faces) {
        HalfEdgeMesh merged_mesh = mergeCoplanarFaces(he_mesh);
        // FIXME: better allocation strategy
        freeHalfEdgeMesh(he_mesh);
        he_mesh = merged_mesh;
    }

    AABB mesh_aabb = AABB::point(src_mesh->positions[0]);
    for (CountT vert_idx = 1; vert_idx < (CountT)src_mesh->numVertices;
         vert_idx++) {
        mesh_aabb.expand(src_mesh->positions[vert_idx]);
    }

    out_prim->hull.halfEdgeMesh = he_mesh;
    *out_aabb = mesh_aabb;

    *total_num_halfedges += he_mesh.numHalfEdges;
    *total_num_faces += he_mesh.numFaces;
    *total_num_vertices += he_mesh.numVertices;
}

PhysicsLoader::ImportedRigidBodies PhysicsLoader::importRigidBodyData(
    const SourceCollisionObject *collision_objs,
    CountT num_objects,
    bool merge_coplanar_faces)
{
    using namespace math;
    using Type = CollisionPrimitive::Type;

    HeapArray<uint32_t> prim_offsets(num_objects);
    HeapArray<uint32_t> prim_counts(num_objects);
    HeapArray<AABB> obj_aabbs(num_objects);
    HeapArray<RigidBodyMetadata> metadatas(num_objects);

    CountT total_num_prims = 0;
    for (CountT obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        const SourceCollisionObject &collision_obj = collision_objs[obj_idx];
        CountT cur_num_prims = collision_obj.prims.size();

        prim_offsets[obj_idx] = total_num_prims;
        prim_counts[obj_idx] = cur_num_prims;
        total_num_prims += cur_num_prims;

        metadatas[obj_idx].friction = collision_objs[obj_idx].friction;
    }

    HeapArray<CollisionPrimitive> collision_prims(total_num_prims);
    HeapArray<AABB> prim_aabbs(total_num_prims);

    CountT cur_prim_offset = 0;
    CountT total_num_halfedges = 0;
    CountT total_num_faces = 0;
    CountT total_num_vertices = 0;
    for (CountT obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        const SourceCollisionObject &collision_obj = collision_objs[obj_idx];

        auto obj_aabb = AABB::invalid();
        for (const SourceCollisionPrimitive &src_prim : collision_obj.prims) {
            CountT out_prim_idx = cur_prim_offset++;
            CollisionPrimitive *out_prim = &collision_prims[out_prim_idx];
            out_prim->type = src_prim.type;
            AABB prim_aabb;

            switch (src_prim.type) {
            case Type::Sphere: {
                setupSpherePrimitive(src_prim, out_prim, &prim_aabb);
            } break;
            case Type::Plane: {
                setupPlanePrimitive(src_prim, out_prim, &prim_aabb);
            } break;
            case Type::Hull: {
                setupHullPrimitive(src_prim, out_prim, &prim_aabb,
                    &total_num_halfedges, &total_num_faces,
                    &total_num_vertices, merge_coplanar_faces);
            } break;
            }

            prim_aabbs[out_prim_idx] = prim_aabb;
            obj_aabb = AABB::merge(obj_aabb, prim_aabb);
        }

        obj_aabbs[obj_idx] = obj_aabb;

        MassProperties mass_props = computeMassProperties(collision_obj);
        metadatas[obj_idx].mass =
            toMassData(mass_props, collision_obj.invMass);
    }

    // Combine half edge data into linear arrays
    ImportedRigidBodies::MergedHullData hull_data {
        .halfEdges = HeapArray<HalfEdge>(total_num_halfedges),
        .faceBaseHEs = HeapArray<uint32_t>(total_num_faces),
        .facePlanes = HeapArray<Plane>(total_num_faces),
        .positions = HeapArray<Vector3>(total_num_vertices),
    };

    CountT cur_halfedge_offset = 0;
    CountT cur_face_offset = 0;
    CountT cur_vert_offset = 0;
    for (CountT prim_idx = 0; prim_idx < total_num_prims; prim_idx++) {
        CollisionPrimitive &cur_prim = collision_prims[prim_idx];
        if (cur_prim.type != Type::Hull) continue;

        HalfEdgeMesh &he_mesh = cur_prim.hull.halfEdgeMesh;
        
        HalfEdge *he_out = &hull_data.halfEdges[cur_halfedge_offset];
        uint32_t *face_bases_out = &hull_data.faceBaseHEs[cur_face_offset];
        Plane *face_planes_out = &hull_data.facePlanes[cur_face_offset];
        Vector3 *pos_out = &hull_data.positions[cur_vert_offset];

        memcpy(he_out, he_mesh.halfEdges,
               sizeof(HalfEdge) * he_mesh.numHalfEdges);
        memcpy(face_bases_out, he_mesh.faceBaseHalfEdges,
               sizeof(uint32_t) * he_mesh.numFaces);
        memcpy(face_planes_out, he_mesh.facePlanes,
               sizeof(Plane) * he_mesh.numFaces);
        memcpy(pos_out, he_mesh.vertices,
               sizeof(Vector3) * he_mesh.numVertices);

        cur_halfedge_offset += he_mesh.numHalfEdges;
        cur_face_offset += he_mesh.numFaces;
        cur_vert_offset += he_mesh.numVertices;

        // FIXME this should be some kind of tmp allocation. See other FIXMEs
        freeHalfEdgeMesh(he_mesh); 

        he_mesh.halfEdges = he_out;
        he_mesh.faceBaseHalfEdges = face_bases_out;
        he_mesh.facePlanes = face_planes_out;
        he_mesh.vertices = pos_out;
    }

    return ImportedRigidBodies {
        .hullData = std::move(hull_data),
        .primitiveAABBs = std::move(prim_aabbs),
        .primOffsets = std::move(prim_offsets),
        .primCounts = std::move(prim_counts),
        .metadatas = std::move(metadatas),
        .objectAABBs = std::move(obj_aabbs),
    };
}

CountT PhysicsLoader::loadObjects(
    const RigidBodyMetadata *metadatas,
    const math::AABB *obj_aabbs,
    const uint32_t *prim_offsets,
    const uint32_t *prim_counts,
    CountT num_objs,
    const CollisionPrimitive *primitives_in,
    const math::AABB *primitive_aabbs,
    CountT total_num_primitives,
    const geometry::HalfEdge *hull_halfedges_in,
    CountT total_num_hull_halfedges,
    const uint32_t *hull_face_base_halfedges_in,
    const geometry::Plane *hull_face_planes_in,
    CountT total_num_hull_faces,
    const math::Vector3 *hull_verts_in,
    CountT total_num_hull_verts)
{
    CountT cur_obj_offset = impl_->curObjOffset;
    impl_->curObjOffset += num_objs;
    CountT cur_prim_offset = impl_->curPrimOffset;
    impl_->curPrimOffset += total_num_primitives;
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

    uint32_t *offsets_tmp = (uint32_t *)malloc(sizeof(uint32_t) * num_objs);
    for (CountT i = 0; i < num_objs; i++) {
        offsets_tmp[i] = prim_offsets[i] + cur_prim_offset;
    }

    HalfEdge *hull_halfedges;
    uint32_t *hull_face_base_halfedges;
    Plane *hull_face_planes;
    Vector3 *hull_verts;
    switch (impl_->storageType) {
    case StorageType::CPU: {
        memcpy(prim_aabbs_dst, primitive_aabbs,
               sizeof(AABB) * total_num_primitives);

        memcpy(obj_aabbs_dst, obj_aabbs,
               sizeof(AABB) * num_objs);
        memcpy(offsets_dst, offsets_tmp,
               sizeof(uint32_t) * num_objs);
        memcpy(counts_dst, prim_counts,
               sizeof(uint32_t) * num_objs);
        memcpy(metadatas_dst, metadatas,
               sizeof(RigidBodyMetadata) * num_objs);

        hull_halfedges =
            (HalfEdge *)malloc(sizeof(HalfEdge) * total_num_hull_halfedges);
        hull_face_base_halfedges =
            (uint32_t *)malloc(sizeof(uint32_t) * total_num_hull_faces);
        hull_face_planes =
            (Plane *)malloc(sizeof(Plane) * total_num_hull_faces);
        hull_verts =
            (Vector3 *)malloc(sizeof(Vector3) * total_num_hull_verts);

        memcpy(hull_halfedges, hull_halfedges_in,
               sizeof(HalfEdge) * total_num_hull_halfedges);
        memcpy(hull_face_base_halfedges, hull_face_base_halfedges_in,
               sizeof(uint32_t) * total_num_hull_faces);
        memcpy(hull_face_planes, hull_face_planes_in,
               sizeof(Plane) * total_num_hull_faces);
        memcpy(hull_verts, hull_verts_in,
               sizeof(Vector3) * total_num_hull_verts);
    } break;
    case StorageType::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        noCUDA();
#else
        cudaMemcpy(prim_aabbs_dst, primitive_aabbs,
                   sizeof(AABB) * total_num_primitives,
                   cudaMemcpyHostToDevice);

        cudaMemcpy(obj_aabbs_dst, obj_aabbs,
                   sizeof(AABB) * num_objs,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(offsets_dst, offsets_tmp,
                   sizeof(uint32_t) * num_objs,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(counts_dst, prim_counts,
                   sizeof(uint32_t) * num_objs,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(metadatas_dst, metadatas,
                   sizeof(RigidBodyMetadata) * num_objs,
                   cudaMemcpyHostToDevice);

        hull_halfedges = (HalfEdge *)cu::allocGPU(
            sizeof(HalfEdge) * total_num_hull_halfedges);
        hull_face_base_halfedges = (uint32_t *)cu::allocGPU(
            sizeof(uint32_t) * total_num_hull_faces);
        hull_face_planes = (Plane *)cu::allocGPU(
            sizeof(Plane) * total_num_hull_faces);
        hull_verts = (Vector3 *)cu::allocGPU(
            sizeof(Vector3) * total_num_hull_verts);

        cudaMemcpy(hull_halfedges, hull_halfedges_in,
                   sizeof(HalfEdge) * total_num_hull_halfedges,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(hull_face_base_halfedges, hull_face_base_halfedges_in,
                   sizeof(uint32_t) * total_num_hull_faces,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(hull_face_planes, hull_face_planes_in,
                   sizeof(Plane) * total_num_hull_faces,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(hull_verts, hull_verts_in,
                   sizeof(Vector3) * total_num_hull_verts,
                   cudaMemcpyHostToDevice);
#endif
    }
    }

    auto primitives_tmp = (CollisionPrimitive *)malloc(
        sizeof(CollisionPrimitive) * total_num_primitives);
    memcpy(primitives_tmp, primitives_in,
           sizeof(CollisionPrimitive) * total_num_primitives);

    for (CountT i = 0; i < total_num_primitives; i++) {
        CollisionPrimitive &cur_primitive = primitives_tmp[i];
        if (cur_primitive.type != CollisionPrimitive::Type::Hull) continue;

        HalfEdgeMesh &he_mesh = cur_primitive.hull.halfEdgeMesh;

        // FIXME: incoming HalfEdgeMeshes should have offsets or something
        CountT hedge_offset = he_mesh.halfEdges - hull_halfedges_in;
        CountT face_offset = he_mesh.facePlanes - hull_face_planes_in;
        CountT vert_offset = he_mesh.vertices - hull_verts_in;

        he_mesh.halfEdges = hull_halfedges + hedge_offset;
        he_mesh.faceBaseHalfEdges = hull_face_base_halfedges + face_offset;
        he_mesh.facePlanes = hull_face_planes + face_offset;
        he_mesh.vertices = hull_verts + vert_offset;
    }

    switch (impl_->storageType) {
    case StorageType::CPU: {
        memcpy(prims_dst, primitives_tmp,
               sizeof(CollisionPrimitive) * total_num_primitives);
    } break;
    case StorageType::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(prims_dst, primitives_tmp,
            sizeof(CollisionPrimitive) * total_num_primitives,
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
