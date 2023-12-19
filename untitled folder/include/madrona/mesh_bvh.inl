namespace madrona::phys {

bool MeshBVH::Node::isLeaf(madrona::CountT child) const
{
    return children[child] & 0x80000000;
}

int32_t MeshBVH::Node::leafIDX(madrona::CountT child) const
{
    return children[child] & ~0x80000000;
}

void MeshBVH::Node::setLeaf(madrona::CountT child, int32_t idx)
{
    children[child] = 0x80000000 | idx;
}

void MeshBVH::Node::setInternal(madrona::CountT child, int32_t internal_idx)
{
    children[child] = internal_idx;
}

bool MeshBVH::Node::hasChild(madrona::CountT child) const
{
    return children[child] != sentinel;
}

void MeshBVH::Node::clearChild(madrona::CountT child)
{
    children[child] = sentinel;
}

template <typename Fn>
void MeshBVH::findOverlaps(const math::AABB &aabb, Fn &&fn) const
{
    using namespace madrona::math;

    int32_t stack[128];
    stack[0] = 0;
    CountT stack_size = 1;

    while (stack_size > 0) {
        int32_t node_idx = stack[--stack_size];
        const Node &node = nodes[node_idx];
        for (int32_t i = 0; i < MeshBVH::nodeWidth; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };

            madrona::math::AABB child_aabb {
                .pMin = {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                .pMax = {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            };

            if (aabb.overlaps(child_aabb)) {
                if (node.isLeaf(i)) {
                    int32_t leaf_idx = node.leafIDX(i);
                    for (int32_t leaf_offset = 0;
                         leaf_offset < numTrisPerLeaf;
                         leaf_offset++) {
                        Vector3 a, b, c;
                        bool tri_exists = fetchLeafTriangle(
                            leaf_idx, leaf_offset, &a, &b, &c);
                        if (!tri_exists) continue;

                        fn(a, b, c);
                    }
                } else {
                    stack[stack_size++] = node.children[i];
                }
            }
        }
    }
}

bool MeshBVH::traceRay(math::Vector3 o,
                       math::Vector3 d,
                       float *out_hit_t,
                       math::Vector3 *out_hit_normal,
                       float t_max) const
{
    using namespace math;

    Diag3x3 inv_d = Diag3x3::fromVec(d).inv();

    RayIsectTxfm tri_isect_txfm = computeRayIsectTxfm(o, d, inv_d);

    int32_t stack[128];
    stack[0] = 0;
    CountT stack_size = 1;

    bool ray_hit = false;
    Vector3 closest_hit_normal;

    while (stack_size > 0) { 
        int32_t node_idx = stack[--stack_size];
        const Node &node = nodes[node_idx];
        for (int32_t i = 0; i < MeshBVH::nodeWidth; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };

            madrona::math::AABB child_aabb {
                .pMin = {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                .pMax = {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
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

            if (t_near <= t_far) {
                if (node.isLeaf(i)) {
                    int32_t leaf_idx = node.leafIDX(i);
                    
                    float hit_t;
                    Vector3 leaf_hit_normal;
                    bool leaf_hit = traceRayIntoLeaf(leaf_idx, tri_isect_txfm,
                        o, t_max, &hit_t, &leaf_hit_normal);

                    if (leaf_hit) {
                        ray_hit = true;
                        t_max = hit_t;
                        closest_hit_normal = leaf_hit_normal;
                    }
                } else {
                    stack[stack_size++] = node.children[i];
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

bool MeshBVH::traceRayIntoLeaf(int32_t leaf_idx,
                               RayIsectTxfm tri_isect_txfm,
                               math::Vector3 ray_o,
                               float t_max,
                               float *out_hit_t,
                               math::Vector3 *out_hit_normal) const
{
    using namespace madrona::math;

    // Woop et al 2013 Watertight Ray/Triangle Intersection
    Vector3 hit_normal;
    float hit_t;
    bool hit_tri = false;
    for (CountT i = 0; i < (CountT)MeshBVH::numTrisPerLeaf; i++) {
        Vector3 a, b, c;
        bool tri_exists = fetchLeafTriangle(leaf_idx, i, &a, &b, &c);
        if (!tri_exists) continue;

        bool intersects = rayTriangleIntersection(
            a, b, c,
            tri_isect_txfm.kx,  tri_isect_txfm.ky, tri_isect_txfm.kz, 
            tri_isect_txfm.Sx,  tri_isect_txfm.Sy, tri_isect_txfm.Sz, 
            ray_o,
            t_max,
            &hit_t,
            &hit_normal);

        if (intersects) {
            hit_tri = true;
            t_max = hit_t;
        }
    }

    if (hit_tri) {
        *out_hit_t = hit_t;
        *out_hit_normal = hit_normal;

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

    // FIXME better way to get geo normal?
    *out_hit_normal = normalize(cross(B - A, C - A));

    return true;
}

bool MeshBVH::fetchLeafTriangle(CountT leaf_idx,
                                CountT offset,
                                math::Vector3 *a,
                                math::Vector3 *b,
                                math::Vector3 *c) const
{
    uint64_t packed = leafGeos[leaf_idx].packedIndices[offset];
    if (packed == 0xFFFF'FFFF'FFFF'FFFF) {
        return false;
    }

    uint32_t a_idx = uint32_t(packed >> 32);
    int16_t b_diff = int16_t((packed >> 16) & 0xFFFF);
    int16_t c_diff = int16_t(packed & 0xFFFF);
    uint32_t b_idx = uint32_t((int32_t)a_idx + b_diff);
    uint32_t c_idx = uint32_t((int32_t)a_idx + c_diff);

    *a = vertices[a_idx];
    *b = vertices[b_idx];
    *c = vertices[c_idx];

    return true;
}

MeshBVH::RayIsectTxfm MeshBVH::computeRayIsectTxfm(
    math::Vector3 o, math::Vector3 d, math::Diag3x3 inv_d) const
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

    math::Vector3 lower = o - rootAABB.pMin;
    math::Vector3 upper = o - rootAABB.pMax;

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

}
