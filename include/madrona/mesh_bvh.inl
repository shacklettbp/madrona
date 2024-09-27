#include <cassert>

#define MADRONA_MESHBVH_BACKFACE_CULLING

namespace madrona {

constexpr int SHARED_STACK_SIZE = 6;

//#define SHARED_STACK
#ifdef SHARED_STACK
    #define STACK_POP(X) { --stack_size; if (stack_size < SHARED_STACK_SIZE) X = shared_stack[stack_size]; else X = stack[stack_size - SHARED_STACK_SIZE]; }
    #define STACK_PUSH(X) { if (stack_size < SHARED_STACK_SIZE) shared_stack[stack_size] = X; else stack[stack_size - SHARED_STACK_SIZE] = X; stack_size++; }
#else
    #define STACK_POP(X) {X = stack[--stack_size];}
    #define STACK_PUSH(X) { stack[stack_size++] = X; }
#endif

template <typename Fn>
void MeshBVH::findOverlaps(const math::AABB &aabb, Fn &&fn) const
{
    using namespace madrona::math;

    int32_t stack[32];
    stack[0] = 0;
    CountT stack_size = 1;

    while (stack_size > 0) {
        int32_t node_idx = stack[--stack_size];
        const QBVHNode &node = nodes[node_idx];
        for (int32_t i = 0; i < QBVHNode::NodeWidth; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };


#ifdef MADRONA_GPU_MODE
#define U32TOFLOAT(x) (__uint_as_float(x))
#else
#define U32TOFLOAT(x) (std::bit_cast<float>(x))
#endif

            math::AABB child_aabb {
                .pMin = {
                    node.minPoint.x + U32TOFLOAT((node.expX + 127) << 23) * node.qMinX[i],
                    node.minPoint.y + U32TOFLOAT((node.expY + 127) << 23) * node.qMinY[i],
                    node.minPoint.z + U32TOFLOAT((node.expZ + 127) << 23) * node.qMinZ[i],
                },
                .pMax = {
                    node.minPoint.x + U32TOFLOAT((node.expX + 127) << 23) * node.qMaxX[i],
                    node.minPoint.y + U32TOFLOAT((node.expY + 127) << 23) * node.qMaxY[i],
                    node.minPoint.z + U32TOFLOAT((node.expZ + 127) << 23) * node.qMaxZ[i],
                },
            };
#undef U32TOFLOAT

            if (aabb.overlaps(child_aabb)) {
                if (node.isLeaf(i)) {
                    int32_t leaf_idx = node.leafIDX(i);
                    for (CountT leaf_offset = 0;
                         leaf_offset < node.triSize[i];
                         leaf_offset++) {
                        Vector3 a, b, c;
                        Vector2 uva, uvb, uvc;
                        bool tri_exists = fetchLeafTriangle(
                            leaf_idx, leaf_offset, &a, &b, &c, &uva, &uvb, &uvc);
                        if (!tri_exists) continue;

                        fn(a, b, c);
                    }
                } else {
                    // assert(stack_size < 32);
                    stack[stack_size++] = node.childrenIdx[i];
                }
            }
        }
    }
}

bool MeshBVH::traceRay(math::Vector3 ray_o,
                       math::Vector3 ray_d,
                       HitInfo *hit_info,
                       int32_t *stack,
                       int32_t &stack_size,
                       float t_max) const
{
    using namespace math;
    constexpr float diveps = 0.0000001f;

    Diag3x3 inv_d = Diag3x3::fromVec(ray_d).inv();

    RayIsectTxfm tri_isect_txfm =
        computeRayIsectTxfm(ray_o, ray_d, inv_d, rootAABB);

    int32_t previous_stack_size = stack_size;

    stack[stack_size++] = 0;

    bool ray_hit = false;

    while (stack_size > previous_stack_size) { 
        int32_t node_idx = stack[--stack_size];
        const QBVHNode &node = nodes[node_idx];

        float rayXInv = copysignf(ray_d.x == 0 ? 1/diveps : 1/ray_d.x,ray_d.x);
        float rayYInv = copysignf(ray_d.y == 0 ? 1/diveps : 1/ray_d.y,ray_d.y);
        float rayZInv = copysignf(ray_d.z == 0 ? 1/diveps : 1/ray_d.z,ray_d.z);
        //NVIDIA's method, transform for ray plane to quantized space. Shift to IEEE exponent bits.

#ifdef MADRONA_GPU_MODE
        float dirQuantX = __uint_as_float((node.expX + 127) << 23) * rayXInv;
        float dirQuantY = __uint_as_float((node.expY + 127) << 23) * rayYInv;
        float dirQuantZ = __uint_as_float((node.expZ + 127) << 23) * rayZInv;
#else
        float dirQuantX = std::bit_cast<float>((node.expX + 127) << 23) * rayXInv;
        float dirQuantY = std::bit_cast<float>((node.expY + 127) << 23) * rayYInv;
        float dirQuantZ = std::bit_cast<float>((node.expZ + 127) << 23) * rayZInv;
#endif

        float originQuantX = (node.minPoint.x - ray_o.x) * rayXInv;
        float originQuantY = (node.minPoint.y - ray_o.y) * rayYInv;
        float originQuantZ = (node.minPoint.z - ray_o.z) * rayZInv;

        for (CountT i = 0; i < MeshBVH::nodeWidth; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };


            float t_near_x = node.qMinX[i] * dirQuantX + originQuantX;
            float t_near_y = node.qMinY[i] * dirQuantY + originQuantY;
            float t_near_z = node.qMinZ[i] * dirQuantZ + originQuantZ;

            float t_far_x = node.qMaxX[i] * dirQuantX + originQuantX;
            float t_far_y = node.qMaxY[i] * dirQuantY + originQuantY;
            float t_far_z = node.qMaxZ[i] * dirQuantZ + originQuantZ;


            float t_near = fmaxf(fminf(t_near_x,t_far_x), fmaxf(fminf(t_near_y,t_far_y),
                fmaxf(fminf(t_near_z,t_far_z), 0.f)));
            float t_far = fminf(fmaxf(t_far_x,t_near_x), fminf(fmaxf(t_far_y,t_near_y),
                fminf(fmaxf(t_far_z,t_near_z), t_max)));

            if (t_near <= t_far) {
                if (node.isLeaf(i)) {
                    int32_t leaf_idx = node.leafIDX(i);
                    
                    bool leaf_hit = traceRayLeaf(leaf_idx, node.triSize[i], tri_isect_txfm,
                        ray_o, t_max, hit_info);

                    if (leaf_hit) {
                        ray_hit = true;
                        t_max = hit_info->tHit;
                    }
                } else {
                    // stack->push(node.children[i]);
                    stack[stack_size++] = node.childrenIdx[i];
                }
            }
        }
    }

    if (!ray_hit) {
        return false;
    }
    
    return ray_hit;
}

#if 0
bool MeshBVH::traceRay(math::Vector3 ray_o,
                       math::Vector3 ray_d,
                       float *out_hit_t,
                       math::Vector3 *out_hit_normal,
                       void* shared,
                       TraversalStack *stack,
                       float t_max) const
{
    using namespace math;
    constexpr float diveps = 0.0000001f;

    Diag3x3 inv_d = Diag3x3::fromVec(ray_d).inv();

    RayIsectTxfm tri_isect_txfm = computeRayIsectTxfm(ray_o, ray_d, inv_d);

    uint32_t previous_stack_size = stack->size;

    stack->push(0);

#ifdef SHARED_STACK
    const int32_t mwgpu_warp_id = threadIdx.x / 32;
    const int32_t mwgpu_warp_lane = threadIdx.x % 32;
    const int32_t num_smem_bytes_per_warp =
        (mwGPU::SharedMemStorage::numBytesPerWarp()/4)*4;

    auto sharedMem = ((char*)shared) + mwgpu_warp_id * num_smem_bytes_per_warp +
            SHARED_STACK_SIZE * sizeof(int32_t) * mwgpu_warp_lane;
    int32_t* shared_stack = (int32_t*)sharedMem;
    shared_stack[0] = 0;
#endif

    bool ray_hit = false;
    Vector3 closest_hit_normal = Vector3{0,0,0};

    while (stack->size > previous_stack_size) { 
        int32_t node_idx = stack->pop();
        const QBVHNode &node = nodes[node_idx];

        float rayXInv = copysignf(ray_d.x == 0 ? 1/diveps : 1/ray_d.x,ray_d.x);
        float rayYInv = copysignf(ray_d.y == 0 ? 1/diveps : 1/ray_d.y,ray_d.y);
        float rayZInv = copysignf(ray_d.z == 0 ? 1/diveps : 1/ray_d.z,ray_d.z);
        //NVIDIA's method, transform for ray plane to quantized space. Shift to IEEE exponent bits.

#ifdef MADRONA_GPU_MODE
        float dirQuantX = __uint_as_float((node.expX + 127) << 23) * rayXInv;
        float dirQuantY = __uint_as_float((node.expY + 127) << 23) * rayYInv;
        float dirQuantZ = __uint_as_float((node.expZ + 127) << 23) * rayZInv;
#else
        float dirQuantX = std::bit_cast<float>((node.expX + 127) << 23) * rayXInv;
        float dirQuantY = std::bit_cast<float>((node.expY + 127) << 23) * rayYInv;
        float dirQuantZ = std::bit_cast<float>((node.expZ + 127) << 23) * rayZInv;
#endif

        float originQuantX = (node.minX - ray_o.x) * rayXInv;
        float originQuantY = (node.minY - ray_o.y) * rayYInv;
        float originQuantZ = (node.minZ - ray_o.z) * rayZInv;

        for (CountT i = 0; i < MeshBVH::nodeWidth; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };


            float t_near_x = node.qMinX[i] * dirQuantX + originQuantX;
            float t_near_y = node.qMinY[i] * dirQuantY + originQuantY;
            float t_near_z = node.qMinZ[i] * dirQuantZ + originQuantZ;

            float t_far_x = node.qMaxX[i] * dirQuantX + originQuantX;
            float t_far_y = node.qMaxY[i] * dirQuantY + originQuantY;
            float t_far_z = node.qMaxZ[i] * dirQuantZ + originQuantZ;
/*
            madrona::math::AABB child_aabb {
                .pMin = {
                    node.minX + std::bit_cast<float>((node.expX + 127) << 23) * node.qMinX[i],
                    node.minY + std::bit_cast<float>((node.expY + 127) << 23) * node.qMinY[i],
                    node.minZ + std::bit_cast<float>((node.expZ + 127) << 23) * node.qMinZ[i],
                },
                .pMax = {
                    node.minX + std::bit_cast<float>((node.expX + 127) << 23) * node.qMaxX[i],
                    node.minY + std::bit_cast<float>((node.expY + 127) << 23) * node.qMaxY[i],
                    node.minZ + std::bit_cast<float>((node.expZ + 127) << 23) * node.qMaxZ[i]
                },
            };

            float t_near_x = (child_aabb[tri_isect_txfm.nearX] -
                              tri_isect_txfm.oNear.x) *
                                 tri_isect_txfm.invDirNear.x;
            float t_near_y = (child_aabb[tri_isect_txfm.nearY] -
                              tri_isect_txfm.oNear.y) *
                                 tri_isect_txfm.invDirNear.y;
            float t_near_z = (child_aabb[tri_isect_txfm.nearZ] -
                              tri_isect_txfm.oNear.z) *
                                 tri_isect_txfm.invDirNear.z;

            float t_far_x = (child_aabb[tri_isect_txfm.farX] -
                              tri_isect_txfm.oFar.x) *
                                 tri_isect_txfm.invDirFar.x;
            float t_far_y = (child_aabb[tri_isect_txfm.farY] -
                              tri_isect_txfm.oFar.y) *
                                 tri_isect_txfm.invDirFar.y;
            float t_far_z = (child_aabb[tri_isect_txfm.farZ] -
                              tri_isect_txfm.oFar.z) *
                                 tri_isect_txfm.invDirFar.z;
            float t_near = fmaxf(t_near_x, fmaxf(t_near_y,
                fmaxf(t_near_z, 0.f)));
            float t_far = fminf(t_far_x, fminf(t_far_y,
                fminf(t_far_z, t_max)));

                                 */

            float t_near = fmaxf(fminf(t_near_x,t_far_x), fmaxf(fminf(t_near_y,t_far_y),
                fmaxf(fminf(t_near_z,t_far_z), 0.f)));
            float t_far = fminf(fmaxf(t_far_x,t_near_x), fminf(fmaxf(t_far_y,t_near_y),
                fminf(fmaxf(t_far_z,t_near_z), t_max)));

            //printf("%f,%f,%f,%f,%f,%f\n",t_near_x,t_near_y,t_near_z,t_far_x,t_far_y,t_far_z);

            if (t_near <= t_far) {
                if (node.isLeaf(i)) {
                    int32_t leaf_idx = node.leafIDX(i);
                    
                    float hit_t;
                    Vector3 leaf_hit_normal;
                    bool leaf_hit = traceRayLeaf(leaf_idx, node.triSize[i], tri_isect_txfm,
                        ray_o, t_max, &hit_t, &leaf_hit_normal);

                    if (leaf_hit) {
                        ray_hit = true;
                        t_max = hit_t;
                        closest_hit_normal = leaf_hit_normal;
                    }
                } else {
                    // assert(stack->size < 32);
                    stack->push(node.childrenIdx[i]);
                }
            }
        }
    }

    if (!ray_hit) {
        return false;
    }
    
    *out_hit_t = t_max;
    *out_hit_normal = closest_hit_normal;
    return ray_hit;
}
#endif

bool MeshBVH::traceRayLeaf(int32_t leaf_idx,
                           int32_t num_tris,
                           RayIsectTxfm tri_isect_txfm,
                           math::Vector3 ray_o,
                           float t_max,
                           HitInfo *hit_info) const
{
    using namespace madrona::math;

    // Woop et al 2013 Watertight Ray/Triangle Intersection
    Vector3 hit_normal = { 0, 0, 0 };
    Vector3 baryout = { 0, 0, 0 };
    Vector3 realout = { 0, 0, 0 };
    Vector2 realuv = { 0, 0 };

    float hit_t;
    bool hit_tri = false;
    uint32_t hit_tri_idx = 0;

    for (CountT i = 0; i < num_tris; i++) {
        Vector3 a, b, c;
        Vector2 uva, uvb, uvc;

        bool tri_exists = fetchLeafTriangle(leaf_idx, i, &a, &b, &c, &uva, &uvb, &uvc);
        (void)tri_exists;

        bool intersects = rayTriangleIntersection(
            a, b, c,
            tri_isect_txfm.kx,  tri_isect_txfm.ky, tri_isect_txfm.kz, 
            tri_isect_txfm.Sx,  tri_isect_txfm.Sy, tri_isect_txfm.Sz, 
            ray_o,
            t_max,
            &hit_t,
            &baryout,
            &hit_normal);

        if (intersects) {
            hit_tri = true;

            realuv = uva*baryout.x + uvb*baryout.y + uvc*baryout.z;

            realout = hit_normal;
            hit_tri_idx = i;

            t_max = hit_t;
        }
    }

    if (hit_tri) {
        hit_info->tHit = hit_t;
        hit_info->normal = realout;
        hit_info->uv = realuv;

        hit_info->leafMaterialIDX = leaf_idx + hit_tri_idx;

        return true;
    } else {
        return false;
    }
}

bool MeshBVH::rayTriangleIntersection(
    math::Vector3 tri_a, math::Vector3 tri_b, math::Vector3 tri_c,
    int32_t kx, int32_t ky, int32_t kz,
    float Sx, float Sy, float Sz,
    math::Vector3 org,
    float t_max,
    float *out_hit_t,
    math::Vector3 *bary_out,
    math::Vector3 *out_hit_normal) const
{
    using namespace madrona::math;

    // Calculate vertices relative to ray origin
    const Vector3 A = tri_a - org;
    const Vector3 B = tri_b - org;
    const Vector3 C = tri_c - org;

    // Perform shear and scale of vertices
    const float Ax = fmaf(-Sx, A[kz], A[kx]);
    const float Ay = fmaf(-Sy, A[kz], A[ky]);
    const float Bx = fmaf(-Sx, B[kz], B[kx]);
    const float By = fmaf(-Sy, B[kz], B[ky]);
    const float Cx = fmaf(-Sx, C[kz], C[kx]);
    const float Cy = fmaf(-Sy, C[kz], C[ky]);

    // calculate scaled barycentric coordinates
    float U = fmaf(Cx, By, - Cy * Bx);
    float V = fmaf(Ax, Cy, - Ay * Cx);
    float W = fmaf(Bx, Ay, - By * Ax);

    // Perform edge tests
#ifdef MADRONA_MESHBVH_BACKFACE_CULLING
    if (U < 0.0f || V < 0.0f || W < 0.0f) {
        return false;
    }
#else
    if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
            (U > 0.0f || V > 0.0f || W > 0.0f)) {
        return false;
    }
#endif

    // fallback  to testing against edges using double precision
    if (U == 0.0f || V == 0.0f || W == 0.0f) [[unlikely]] {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);

        // Perform edge tests
#ifdef MADRONA_MESHBVH_BACKFACE_CULLING
        if (U < 0.0f || V < 0.0f || W < 0.0f) {
            return false;
        }
#else
        if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
                (U > 0.0f || V > 0.0f || W > 0.0f)) {
            return false;
        }
#endif
    }

    // Calculate determinant
    float det = U + V + W;
    if (det == 0.f) return false;

    // Calculate scaled z-coordinates of vertices and use them to calculate
    // the hit distance
    const float Az = Sz * A[kz];
    const float Bz = Sz * B[kz];
    const float Cz = Sz * C[kz];
    const float T = fmaf(U, Az, fmaf(V, Bz, W * Cz));

#ifdef MADRONA_MESHBVH_BACKFACE_CULLING
    if (T < 0.0f || T > t_max * det) {
        return false;
    }

#else
#ifdef MADRONA_GPU_MODE
    uint32_t det_sign_mask =
        __float_as_uint(det) & 0x8000'0000_u32;

    float xor_T = __uint_as_float(
        __float_as_uint(T) ^ det_sign_mask);
#else
    uint32_t det_sign_mask =
        std::bit_cast<uint32_t>(det) & 0x8000'0000_u32;

    float xor_T = std::bit_cast<float>(
        std::bit_cast<uint32_t>(T) ^ det_sign_mask);
#endif
    
    float xor_det = copysignf(det, 1.f);
    if (xor_T < 0.0f || xor_T > t_max * xor_det) {
        return false;
    }
#endif

    // normalize U, V, W, and T
    const float rcpDet = 1.0f / det;

#if 0
    hit.u = U * rcpDet;
    hit.v = V * rcpDet;
    hit.w = W * rcpDet;
    hit.t = T * rcpDet;
#endif

    *out_hit_t = T * rcpDet;
    *bary_out = Vector3{U,V,W} * rcpDet;

    // FIXME better way to get geo normal?
    *out_hit_normal = normalize(cross(B - A, C - A));

    return true;
}

bool MeshBVH::fetchLeafTriangle(CountT leaf_idx,
                                CountT offset,
                                math::Vector3 *a,
                                math::Vector3 *b,
                                math::Vector3 *c,
                                math::Vector2 *uv_a,
                                math::Vector2 *uv_b,
                                math::Vector2 *uv_c) const
{
    *a = vertices[(leaf_idx + offset)*3 + 0].pos;
    *uv_a = vertices[(leaf_idx + offset)*3 + 0].uv;

    *b = vertices[(leaf_idx + offset)*3 + 1].pos;
    *uv_b = vertices[(leaf_idx + offset)*3 + 1].uv;

    *c = vertices[(leaf_idx + offset)*3 + 2].pos;
    *uv_c = vertices[(leaf_idx + offset)*3 + 2].uv;


    return true;
}

MeshBVH::RayIsectTxfm MeshBVH::computeRayIsectTxfm(
    math::Vector3 o, math::Vector3 d, math::Diag3x3 inv_d) const
{
    return computeRayIsectTxfm(o, d, inv_d, rootAABB);
}

MeshBVH::RayIsectTxfm MeshBVH::computeRayIsectTxfm(
    math::Vector3 o, math::Vector3 d, math::Diag3x3 inv_d,
    math::AABB root_aabb)
{
    // Woop et al 2013
    float abs_x = fabsf(d.x);
    float abs_y = fabsf(d.y);
    float abs_z = fabsf(d.z);

    int32_t kz;
    if (abs_x > abs_y && abs_x > abs_z) {
        kz = 0;
    } else if (abs_y > abs_z) {
        kz = 1;
    } else {
        kz = 2;
    }

    int32_t kx = kz + 1;
    if (kx == 3) {
        kx = 0;
    }

    int32_t ky = kx + 1;
    if (ky == 3) {
        ky = 0;
    }

    // swap kx and ky dimensions to preserve winding direction of triangles
    if (d[kz] < 0.f) {
        std::swap(kx, ky);
    }

    // Calculate shear constants
    float Sx = d[kx] * inv_d[kz];
    float Sy = d[ky] * inv_d[kz];
    float Sz = inv_d[kz];

    // AABB check precomputations
    int32_t near_id[3] = { 0, 1, 2 };
    int32_t far_id[3] = { 3, 4, 5 };

    int32_t near_x = near_id[kx], far_x = far_id[kx];
    int32_t near_y = near_id[ky], far_y = far_id[ky];
    int32_t near_z = near_id[kz], far_z = far_id[kz];

    if (inv_d[kx] < 0.f) {
        std::swap(near_x, far_x);
    }

    if (inv_d[ky] < 0.f) {
        std::swap(near_y, far_y);
    }

    if (inv_d[kz] < 0.f) {
        std::swap(near_z, far_z);
    }

    constexpr float p = 1.00000012f;
    constexpr float m = 0.99999988f;

    auto up = [](float a) {
        return a > 0.f ? a * p : a * m;
    };

    auto dn = [](float a) {
        return a > 0.f ? a * m : a * p;
    };

    // Positive only
    auto upPos = [](float a) {
        return a * p;
    };
    auto dnPos = [](float a) {
        return a * m;
    };

    constexpr float eps = 2.98023224e-7f;

    math::Vector3 lower = o - root_aabb.pMin;
    math::Vector3 upper = o - root_aabb.pMax;

    lower.x = dnPos(fabsf(lower.x));
    lower.y = dnPos(fabsf(lower.y));
    lower.z = dnPos(fabsf(lower.z));

    upper.x = upPos(fabsf(upper.x));
    upper.y = upPos(fabsf(upper.y));
    upper.z = upPos(fabsf(upper.z));

    float max_z = fmaxf(lower[kz], upper[kz]);

    float err_near_x = upPos(lower[kx] + max_z);
    float err_near_y = upPos(lower[ky] + max_z);
    float o_near_x = up(o[kx] + upPos(eps * err_near_x));
    float o_near_y = up(o[ky] + upPos(eps * err_near_y));
    float o_near_z = o[kz];
    float err_far_x = upPos(upper[kx] + max_z);
    float err_far_y = upPos(upper[ky] + max_z);
    float o_far_x = dn(o[kx] - upPos(eps * err_far_x));
    float o_far_y = dn(o[ky] - upPos(eps * err_far_y));
    float o_far_z = o[kz];

    if (inv_d[kx] < 0.0f) {
        std::swap(o_near_x, o_far_x);
    }

    if (inv_d[ky] < 0.0f) {
        std::swap(o_near_y, o_far_y);
    }

    // Calculate corrected reciprocal direction for near
    // and far-plane distance calculations. We correct with one additional ulp
    // to also correctly round the substraction inside the traversal loop. This
    // works only because the ray is only allowed to hit geometry in front of
    // it.
    float rdir_near_x = dnPos(dnPos(inv_d[kx]));
    float rdir_near_y = dnPos(dnPos(inv_d[ky]));
    float rdir_near_z = dnPos(dnPos(inv_d[kz]));
    float rdir_far_x = upPos(upPos(inv_d[kx]));
    float rdir_far_y = upPos(upPos(inv_d[ky]));
    float rdir_far_z = upPos(upPos(inv_d[kz]));

    return RayIsectTxfm {
        .kx = kx,
        .ky = ky,
        .kz = kz,
        .Sx = Sx,
        .Sy = Sy,
        .Sz = Sz,
        .nearX = near_x,
        .nearY = near_y,
        .nearZ = near_z,
        .farX = far_x,
        .farY = far_y,
        .farZ = far_z,
        .oNear = {
            .x = o_near_x,
            .y = o_near_y,
            .z = o_near_z,
        },
        .oFar = {
            .x = o_far_x,
            .y = o_far_y,
            .z = o_far_z,
        },
        .invDirNear = {
            .x = rdir_near_x,
            .y = rdir_near_y,
            .z = rdir_near_z,
        },
        .invDirFar = {
            .x = rdir_far_x,
            .y = rdir_far_y,
            .z = rdir_far_z,
        },
    };
}

float MeshBVH::sphereCast(math::Vector3 ray_o,
                          math::Vector3 ray_d,
                          float sphere_r,
                          math::Vector3 *out_hit_normal,
                          float t_max)
{
    using namespace math;

    Diag3x3 inv_d = Diag3x3::fromVec(ray_d).inv();

    int32_t stack[32];
    stack[0] = 0;
    CountT stack_size = 1;

    Vector3 closest_hit_normal;

    float hit_t = t_max;
    while (stack_size > 0) { 
        int32_t node_idx = stack[--stack_size];
        const QBVHNode &node = nodes[node_idx];
        MADRONA_UNROLL
        for (CountT i = 0; i < (CountT)MeshBVH::nodeWidth; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };

#ifdef MADRONA_GPU_MODE
#define U32TOFLOAT(x) (__uint_as_float(x))
#else
#define U32TOFLOAT(x) (std::bit_cast<float>(x))
#endif

            math::AABB child_aabb {
                .pMin = {
                    node.minPoint.x + U32TOFLOAT(((uint32_t)node.expX + 127) << 23) * node.qMinX[i],
                    node.minPoint.y + U32TOFLOAT(((uint32_t)node.expY + 127) << 23) * node.qMinY[i],
                    node.minPoint.z + U32TOFLOAT(((uint32_t)node.expZ + 127) << 23) * node.qMinZ[i],
                },
                .pMax = {
                    node.minPoint.x + U32TOFLOAT(((uint32_t)node.expX + 127) << 23) * node.qMaxX[i],
                    node.minPoint.y + U32TOFLOAT(((uint32_t)node.expY + 127) << 23) * node.qMaxY[i],
                    node.minPoint.z + U32TOFLOAT(((uint32_t)node.expZ + 127) << 23) * node.qMaxZ[i],
                },
            };

#undef U32TOFLOAT

            if (sphereCastNodeCheck(ray_o, inv_d, hit_t, sphere_r, child_aabb)) {
                if (node.isLeaf(i)) {
                    int32_t leaf_idx = node.leafIDX(i);
                    
                    Vector3 leaf_hit_normal;
                    float leaf_hit_t = sphereCastLeaf(leaf_idx, ray_o, ray_d,
                        hit_t, sphere_r, &leaf_hit_normal);

                    if (leaf_hit_t < hit_t) {
                        hit_t = leaf_hit_t;
                        closest_hit_normal = leaf_hit_normal;
                    }
                } else {
                    // assert(stack_size < 32);
                    stack[stack_size++] = node.childrenIdx[i];
                }
            }
        }
    }

    if (hit_t < t_max) {
        *out_hit_normal = closest_hit_normal;
    }

    return hit_t;
}

bool MeshBVH::sphereCastNodeCheck(math::Vector3 o,
                                  math::Diag3x3 inv_d,
                                  float t_max,
                                  float sphere_r,
                                  math::AABB aabb) const
{
    using namespace math;

    // RTCD 5.5.7: Simplified since we don't care if this test incorrectly
    // classifies the corners.
    // Compute the AABB resulting from expanding aabb by sphere radius r
    AABB e = aabb;
    e.pMin.x -= sphere_r; e.pMin.y -= sphere_r; e.pMin.z -= sphere_r;
    e.pMax.x += sphere_r; e.pMax.y += sphere_r; e.pMax.z += sphere_r;

    // https://tavianator.com/2022/ray_box_boundary.html
    float t_min = 0.f;
    MADRONA_UNROLL
    for (CountT i = 0; i < 3; i++) {
        float inv_d_i = inv_d[i];
        float b_min, b_max;
        if (signbit(inv_d_i) == 0) {
            b_min = e.pMin[i];
            b_max = e.pMax[i];
        } else {
            b_min = e.pMax[i];
            b_max = e.pMin[i];
        }

        float i_min = (b_min - o[i]) * inv_d_i;
        float i_max = (b_max - o[i]) * inv_d_i;

        // Note max and min need to have this form for nan handling
        t_min = i_min > t_min ? i_min : t_min;
        t_max = i_max < t_max ? i_max : t_max;
    }

    return t_min < t_max;
}

float MeshBVH::sphereCastLeaf(int32_t leaf_idx,
                              math::Vector3 ray_o,
                              math::Vector3 ray_d,
                              float t_max,
                              float sphere_r,
                              math::Vector3 *out_hit_normal) const
{
    using namespace madrona::math;

    float hit_t = t_max;
    for (CountT i = 0; i < MeshBVH::numTrisPerLeaf; i++) {
        Vector3 a, b, c;
        Vector2 uva, uvb, uvc;
        bool tri_exists = fetchLeafTriangle(leaf_idx, i, &a, &b, &c, &uva, &uvb, &uvc);
        if (!tri_exists) continue;

        hit_t = sphereCastTriangle(
            a, b, c,
            ray_o,
            ray_d,
            hit_t,
            sphere_r,
            out_hit_normal);
    }

    return hit_t;
}

float MeshBVH::sphereCastTriangle(math::Vector3 tri_a,
                                  math::Vector3 tri_b,
                                  math::Vector3 tri_c,
                                  math::Vector3 ray_o,
                                  math::Vector3 ray_d,
                                  float t_max,
                                  float sphere_r,
                                  math::Vector3 *out_hit_normal) const
{
    // This is heavily based on Jolt's implementation.
    // License of the Jolt codebase:
    //
    // Copyright 2021 Jorrit Rouwe
    // Permission is hereby granted, free of charge, to any person obtaining a
    // copy of this software and associated documentation files (the
    // "Software"), to deal in the Software without restriction, including
    // without limitation the rights to use, copy, modify, merge, publish,
    // distribute, sublicense, and/or sell copies of the Software, and to
    // permit persons to whom the Software is furnished to do so, subject to
    // the following conditions:
    // 
    // The above copyright notice and this permission notice shall be included
    // in all copies or substantial portions of the Software.
    // 
    // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    // OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    // MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    // IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    // CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    // TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    // SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    
    using namespace math;

    // Compute edges before shifting to ray origin
    const Vector3 e01 = tri_b - tri_a;
    const Vector3 e02 = tri_c - tri_a;
    const Vector3 e12 = tri_c - tri_b;

    // Shift verts to ray origin at (0, 0, 0)
    const Vector3 v0 = tri_a - ray_o;
    const Vector3 v1 = tri_b - ray_o;
    const Vector3 v2 = tri_c - ray_o;

    Vector3 triangle_normal_unnormalized =
        geo::computeTriangleGeoNormal(e01, e02, e12);
    float triangle_normal_len = triangle_normal_unnormalized.length();
    Vector3 triangle_normal =
        triangle_normal_unnormalized / triangle_normal_len;

    const float normal_dot_direction = dot(triangle_normal, ray_d);

    const float sphere_r2 = sphere_r * sphere_r;

    // This first if statement is an optimization to avoid the Case 0 check 
    // if possible. The sphere is centered at (0, 0, 0), so we can avoid the
    // Case 0 (initial intersection) test if the triangle is more than
    // sphere_r away from the origin, measured along the triangle normal, so
    // all the vertex distances are equal.
    if (fabsf(dot(v0, triangle_normal)) <= sphere_r) {
	    // Case 0: The sphere may intersect at the start of the cast
		Vector3 q = geo::triangleClosestPointToOrigin(v0, v1, v2, e01, e02);
		float q_len2 = q.length2();
		if (q_len2 <= sphere_r2) {
			float q_len = sqrtf(q_len2);
            *out_hit_normal = q_len > 0.0f ? q / q_len : up;
			return 0.f;
		}
	} else if (
        // Case 1: Does the sphere intersect the 2 * sphere_r thick extruded 
        // triangle, ignoring the edges?
        // Skip this test if the sphere cast is parallel to the triangle plane,
        // because we know from the Case 0 if statement that the triangle is
        // more than sphere_r away from the sphere. Therefore, either the
        // sphere will miss the triangle, or it is far away and will hit one of
        // the edges, tested later.
        float abs_normal_dot_direction = fabsf(normal_dot_direction);
        abs_normal_dot_direction > 1.0e-6f
    ) {
        // Slightly modified version of RTCD 5.3.6. Ray - Triangle test
        // against either the top or bottom of the extruded triangle
        // abs_normal_dot_direction = d in textbook with normalized n

        // Calculate the point on the sphere that will hit the triangle's
        // plane first and calculate a t_hit where it will do so
        float normal_dir_sign = copysignf(1.f, normal_dot_direction);
        Vector3 extruded_delta = normal_dir_sign * sphere_r * triangle_normal;
        // Want to extrude triangle *towards* ray
        Vector3 v0_extruded = v0 - extruded_delta; // A in RTCD
        float tri_plane_hit_t =
            dot(v0_extruded, triangle_normal) / normal_dot_direction;
        
        // Early out checks: do we hit the plane but not within the range 
        // of the sphere cast?
        if (
	        // Sphere hits the plane before the sweep, cannot intersect
            tri_plane_hit_t * abs_normal_dot_direction < -sphere_r ||
            // Sphere hits the plane after the sweep / current hit_t
            // cannot intersect
            tri_plane_hit_t >= t_max
        ) {
        	return t_max;
        }
        
        // We can only report an interior hit if we're hitting the plane during
        // our sweep and not before
        if (tri_plane_hit_t >= 0.0f) {
            Vector3 e = cross(ray_d, v0_extruded);
            float v = -dot(e02, e) * normal_dir_sign;
            float w = dot(e01, e) * normal_dir_sign;

            if (v >= 0.f && w >= 0.f &&
                    v + w <= triangle_normal_len * abs_normal_dot_direction) {
                *out_hit_normal = -normal_dir_sign * triangle_normal;
                return tri_plane_hit_t;
            } 
        }
    }

    // Test edges
    // RTCD 5.3.7 (Ray - Cylinder) with end caps removed
    constexpr float edge_epsilon = 1e-6f;
    const float ray_d_len2 = ray_d.length2();
    auto testEdge = [ray_d_len2, ray_d, sphere_r2](
            Vector3 axis, Vector3 cylinder_base, float hit_t) {
	    // Make ray start relative to cylinder side A
        // (moving cylinder end_a to the origin)
	    Vector3 start = -cylinder_base;

	    // Test if segment is fully on the A side of the cylinder
	    const float start_dot_axis = dot(start, axis);
	    const float dir_dot_axis = dot(ray_d, axis);
	    const float end_dot_axis = start_dot_axis + dir_dot_axis;
	    if (start_dot_axis < 0.0f && end_dot_axis < 0.0f) {
	    	return hit_t;
        }

	    // Test if segment is fully on the B side of the cylinder
	    const float axis_len2 = axis.length2();
	    if (start_dot_axis > axis_len2 && end_dot_axis > axis_len2) {
	    	return hit_t;
        }

	    // Calculate a, b and c, the factors for quadratic equation
	    // We're basically solving the ray: x = start + direction * t
	    // The closest point to x on the segment A B is:
        // w = (x . axis) * axis / (axis . axis)
	    // The distance between x and w should be radius:
        // (x - w) . (x - w) = radius^2
	    // Solving this gives the following:
	    float a = axis_len2 * ray_d_len2 - dir_dot_axis * dir_dot_axis;
	    if (fabsf(a) < edge_epsilon) {
            // Segment runs parallel to cylinder axis, stop processing,
            // we will either hit at fraction = 0 or we'll hit a vertex
	    	return hit_t;
        }

        // should be multiplied by 2, instead we'll divide a and c by 2 when
        // we solve the quadratic equation
	    float b = axis_len2 * dot(start, ray_d) - dir_dot_axis * start_dot_axis;
	    float c = axis_len2 * (start.length2() - sphere_r2) -
            start_dot_axis * start_dot_axis;
        // normally 4 * a * c but since both a and c need to be divided by 2
        // we lose the 4
	    float det = b * b - a * c;
	    if (det < 0.0f) {
	    	return hit_t; // No solution to quadratic equation
        }
	    
	    // Solve fraction t where the ray hits the cylinder
        // normally divided by 2 * a but since a should be divided by 2
        // we lose the 2
	    float t = -(b + sqrtf(det)) / a;
	    if (t < 0.0f || t >= hit_t)  {
            // Intersection lies outside segment
	    	return hit_t;
        }

	    if (start_dot_axis + t * dir_dot_axis < 0.0f ||
            start_dot_axis + t * dir_dot_axis > axis_len2)  {
            // Intersection outside the end point of the cyclinder, stop processing,
            // we will possibly hit a vertex
	    	return hit_t;
        }

        return t;
    };

    auto testVert = [sphere_r2, ray_d, ray_o](Vector3 v, float hit_t) {
        Vector3 m = ray_o - v;

        float b = dot(m, ray_d);
        float c = dot(m, m) - sphere_r2;
        // Exit if r’s origin outside s (c > 0) and
        // r pointing away from s (b > 0)
        if (c > 0.0f && b > 0.0f) {
            return hit_t;
        }

        float discr = b * b - c;
        // A negative discriminant corresponds to ray missing sphere
        if (discr < 0.0f) {
            return hit_t;
        }

        // Ray now found to intersect sphere, compute smallest t value of
        // intersection. 
        float t = -b - sqrtf(discr);
        if (t < 0.f) {
            // FIXME: t shouldn't be negative at this point, but it 
            // happens during training
            return 0.f;
        }

        if (t >= hit_t) {
            return hit_t;
        }

        return t;
    };

    float hit_t = t_max;
    hit_t = testEdge(e01, v0, hit_t);
    hit_t = testEdge(e02, v0, hit_t);
    hit_t = testEdge(e12, v1, hit_t);

    hit_t = testVert(v0, hit_t);
    hit_t = testVert(v1, hit_t);
    hit_t = testVert(v2, hit_t);

    if (hit_t >= t_max) {
        return t_max;
    }

    Vector3 hit_pt_approx = ray_d * hit_t;
    Vector3 closest_tri_pt = geo::triangleClosestPointToOrigin(
        v0 - hit_pt_approx,
        v1 - hit_pt_approx,
        v2 - hit_pt_approx, e01, e02);

    *out_hit_normal = normalize(closest_tri_pt);
    return hit_t;
}

uint32_t MeshBVH::getMaterialIDX(const HitInfo &info) const
{
    if (materialIDX == -1) {
        return leafMats[info.leafMaterialIDX].material[0].matIDX;
    } else {
        return materialIDX;
    }
}

uint32_t MeshBVH::getMaterialIDX(int32_t mat_idx) const
{
    if (materialIDX == -1) {
        return leafMats[mat_idx].material[0].matIDX;
    } else {
        return materialIDX;
    }
}

}
