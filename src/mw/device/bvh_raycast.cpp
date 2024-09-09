#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh.hpp>
#include <madrona/mw_gpu/host_print.hpp>

#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)

using namespace madrona;
using namespace madrona::math;
using namespace madrona::render;

namespace sm {
// Only shared memory to be used
extern __shared__ uint8_t buffer[];
}

extern "C" __constant__ BVHParams bvhParams;

inline Vector3 lighting(Vector3 diffuse,
                        Vector3 normal,
                        Vector3 raydir,
                        float roughness,
                        float metalness)
{
    constexpr float ambient = 0.4;
    Vector3 lightDir = Vector3{0.5,0.5,0};
    return (fminf(fmaxf(normal.dot(lightDir),0.f)+ambient, 1.0f)) * diffuse;
}

inline Vector3 calculateOutRay(PerspectiveCameraData *view_data,
                               uint32_t pixel_x, uint32_t pixel_y)
{
    Quat rot = view_data->rotation;
    Vector3 ray_start = view_data->position;
    Vector3 look_at = rot.inv().rotateVec({0, 1, 0});
 
    // const float h = tanf(theta / 2);
    const float h = 1.0f / (-view_data->yScale);

    const auto viewport_height = 2 * h;
    const auto viewport_width = viewport_height;
    const auto forward = look_at.normalize();

    auto u = rot.inv().rotateVec({1, 0, 0});
    auto v = cross(forward, u).normalize();

    auto horizontal = u * viewport_width;
    auto vertical = v * viewport_height;

    auto lower_left_corner = ray_start - horizontal / 2 - vertical / 2 + forward;
  
    float pixel_u = ((float)pixel_x + 0.5f) / (float)bvhParams.renderOutputResolution;
    float pixel_v = ((float)pixel_y + 0.5f) / (float)bvhParams.renderOutputResolution;

    Vector3 ray_dir = lower_left_corner + pixel_u * horizontal + 
        pixel_v * vertical - ray_start;
    ray_dir = ray_dir.normalize();

    return ray_dir;
}

struct TraceResult {
    bool hit;
    Vector3 color;
    float depth;
};

struct TraceInfo {
    Vector3 rayOrigin;
    Vector3 rayDirection;
    float tMin;
    float tMax;
};

struct TraceWorldInfo {
    QBVHNode *nodes;
    InstanceData *instances;
};

enum class GroupType : uint8_t {
    TopLevel,
    BottomLevelRoot,
    BottomLevel,
    Triangles,
    None
};

using NodeGroup = uint64_t;

// Node group just has 8 present bits
static NodeGroup encodeNodeGroup(
        uint32_t node_idx,
        uint32_t present_bits,
        GroupType type)
{
    return (uint64_t)node_idx |
           ((uint64_t)present_bits << 32) |
           ((uint64_t)type << 61);
}

static NodeGroup invalidNodeGroup()
{
    return encodeNodeGroup(0xFFFF'FFFF, 0, GroupType::None);
}

static NodeGroup getRootGroup(TraceWorldInfo world_info)
{
    uint32_t children_count = world_info.nodes[0].numChildren;
    uint8_t present_bits = (uint8_t)((1 << children_count) - 1);
    return encodeNodeGroup(0, present_bits, GroupType::TopLevel);
}

static GroupType getGroupType(NodeGroup grp)
{
    return (GroupType)((grp >> 61) & 0b111);
}

static NodeGroup unsetPresentBit(NodeGroup grp, uint32_t idx)
{
    grp &= ~(1 << (idx + 32));
    return grp;
}

static uint32_t getPresentBits(NodeGroup grp)
{
    return (uint32_t)((grp >> 32) & 0xFF);
}

static uint32_t getTrianglePresentBits(NodeGroup grp)
{
    // 24 bits used for triangle presence
    return (uint32_t)((grp >> 32) & ((1 << 24) - 1));
}

static Vector3 getDirQuant(QBVHNode node, Diag3x3 inv_ray_d)
{
    return Vector3 {
        __uint_as_float((node.expX + 127) << 23) * inv_ray_d.d0,
        __uint_as_float((node.expY + 127) << 23) * inv_ray_d.d1,
        __uint_as_float((node.expZ + 127) << 23) * inv_ray_d.d2,
    };
}

static Vector3 getOriginQuant(QBVHNode node, Vector3 ray_o, Diag3x3 inv_ray_d)
{
    return Vector3 {
        (node.minPoint.x - ray_o.x) * inv_ray_d.d0,
        (node.minPoint.y - ray_o.y) * inv_ray_d.d1,
        (node.minPoint.z - ray_o.z) * inv_ray_d.d2,            
    };
}

static std::pair<float, float> getNearFar(
        QBVHNode node, 
        uint32_t child_idx,
        Vector3 dir_quant,
        Vector3 origin_quant,
        float t_max)
{
    Vector3 t_near3 = {
        node.qMinX[child_idx] * dir_quant.x + origin_quant.x,
        node.qMinY[child_idx] * dir_quant.y + origin_quant.y,
        node.qMinZ[child_idx] * dir_quant.z + origin_quant.z,
    };

    Vector3 t_far3 = {
        node.qMaxX[child_idx] * dir_quant.x + origin_quant.x,
        node.qMaxY[child_idx] * dir_quant.y + origin_quant.y,
        node.qMaxZ[child_idx] * dir_quant.z + origin_quant.z,
    };

    float t_near = fmaxf(fminf(t_near3.x, t_far3.x), 
                         fmaxf(fminf(t_near3.y, t_far3.y),
                               fmaxf(fminf(t_near3.z, t_far3.z), 
                                     0.f)));

    float t_far = fminf(fmaxf(t_far3.x, t_near3.x), 
                        fminf(fmaxf(t_far3.y, t_near3.y),
                              fminf(fmaxf(t_far3.z, t_near3.z), 
                                    t_max)));

    return {
        t_near, t_far
    };
}

struct TriangleIsectInfo {
    // TODO: can reduce register usage by storing kx/y/z in a single int32_t
    int32_t kx, ky, kz;
    float Sx, Sy, Sz;
};

static TriangleIsectInfo computeRayIsectInfo(
        Vector3 o, Vector3 d,
        Diag3x3 inv_d)
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

    return {
        kx, ky, kz,
        Sx, Sy, Sz
    };
}

static bool instanceHasVolume(InstanceData *instance_data)
{
    return !(instance_data->scale.d0 == 0.f &&
             instance_data->scale.d1 == 0.f &&
             instance_data->scale.d2 == 0.f);
}

struct TriHitInfo {
    bool hit;

    float tHit;

    Vector3 normal;
    Vector2 uv;

    uint32_t leafMaterialIndex;

    MeshBVH *bvh;
};

struct TriangleFetch {
    Vector3 a, b, c;
    Vector2 uva, uvb, uvc;
};

static TriangleFetch fetchLeafTriangle(int32_t leaf_idx,
                                       int32_t offset,
                                       MeshBVH *mesh_bvh)
{
    TriangleFetch fetched = {
        .a = mesh_bvh->vertices[(leaf_idx + offset)*3 + 0].pos,
        .b = mesh_bvh->vertices[(leaf_idx + offset)*3 + 1].pos,
        .c = mesh_bvh->vertices[(leaf_idx + offset)*3 + 2].pos,
        .uva = mesh_bvh->vertices[(leaf_idx + offset)*3 + 0].uv,
        .uvb = mesh_bvh->vertices[(leaf_idx + offset)*3 + 1].uv,
        .uvc = mesh_bvh->vertices[(leaf_idx + offset)*3 + 2].uv,
    };

    return fetched;
}

static bool rayTriangleIntersection(
    Vector3 tri_a, Vector3 tri_b, Vector3 tri_c,
    int32_t kx, int32_t ky, int32_t kz,
    float Sx, float Sy, float Sz,
    Vector3 org,
    float t_max,
    float *out_hit_t,
    Vector3 *bary_out,
    Vector3 *out_hit_normal)
{
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

    *out_hit_t = T * rcpDet;
    *bary_out = Vector3{U,V,W} * rcpDet;

    // FIXME better way to get geo normal?
    *out_hit_normal = normalize(cross(B - A, C - A));

    return true;
}

static TriHitInfo triangleIntersect(int32_t leaf_idx,
                                    int32_t tri_idx,
                                    TriangleIsectInfo isect_info,
                                    Vector3 ray_o,
                                    float t_max,
                                    MeshBVH *mesh_bvh)
{
    // Woop et al 2013 Watertight Ray/Triangle Intersection
    Vector3 hit_normal = { 0, 0, 0 };
    Vector3 baryout = { 0, 0, 0 };

    float hit_t;

    TriangleFetch fetched = fetchLeafTriangle(
            leaf_idx, tri_idx, mesh_bvh);

    bool intersects = rayTriangleIntersection(
            fetched.a, fetched.b, fetched.c,
            isect_info.kx,  isect_info.ky, isect_info.kz, 
            isect_info.Sx,  isect_info.Sy, isect_info.Sz, 
            ray_o,
            t_max,
            &hit_t,
            &baryout,
            &hit_normal);

    if (intersects) {
        Vector2 realuv = fetched.uva * baryout.x +
                         fetched.uvb * baryout.y +
                         fetched.uvc * baryout.z;

        return {
            .hit = true,
            .tHit = hit_t,
            .normal = hit_normal,
            .uv = realuv,
            .leafMaterialIndex = (uint32_t)(leaf_idx + tri_idx),
            .bvh = mesh_bvh,
        };
    } else {
        return {
            .hit = false,
        };
    }
}

static __device__ TraceResult traceRay(
    TraceInfo trace_info,
    TraceWorldInfo world_info)
{
    // We create these so that we can keep track of the original ray origin,
    // direction in world space. We will need to transform them when we enter
    // a bottom level structure.
    Vector3 ray_o = trace_info.rayOrigin;
    Vector3 ray_d = trace_info.rayDirection;
    Diag3x3 inv_ray_d = Diag3x3::fromVec(ray_d).inv();
    float t_max = trace_info.tMax;

    TraceResult result = {
        .hit = false
    };

    NodeGroup stack[64];
    uint32_t stack_size = 0;

    NodeGroup current_grp = getRootGroup(world_info);
    NodeGroup triangle_grp = invalidNodeGroup();

    // Here is some stuff we need to store in case we are now in the bottom
    // level structure. Need to maintain some extra state unfortunately.
    TriangleIsectInfo isect_info = {};
    MeshBVH *current_bvh = nullptr;
    float t_scale = 1.f;

    QBVHNode *node_buffer = world_info.nodes;

    TriHitInfo tri_hit;

    for (;;) {
        if (GroupType parent_grp_type = getGroupType(current_grp); 
                parent_grp_type != GroupType::Triangles) {
            // TODO: Make sure to have the node traversal order
            // sorted according to the ray direction.
            // NOTE: This should never underflow
            uint32_t child_idx = __ffs(getPresentBits(current_grp)) - 1;
            current_grp = unsetPresentBit(current_grp, child_idx);

            if (getPresentBits(current_grp) != 0) {
                stack[stack_size++] = current_grp;
            }

            // Intersect with the children of the child to get a new node group
            // and calculate the present bits according to which were
            // intersected
            // TODO: Differentiate between TLAS and BLAS. For now, we are just
            // rewriting the TLAS tracing code for testing.
            uint32_t child_node_idx =
                node_buffer[current_grp & 0xFFFF'FFFF].childrenIdx[child_idx];
            bool child_is_leaf = (child_node_idx & 0x8000'0000);

            GroupType new_grp_type = GroupType::TopLevel;

            if (parent_grp_type == GroupType::TopLevel &&
                child_is_leaf) {
                // Need to compute new ray o/d/etc...
                int32_t instance_idx = (int32_t)(child_node_idx & ~0x8000'0000);
                InstanceData *instance_data = world_info.instances + 
                                              instance_idx;
                current_bvh = bvhParams.bvhs + 
                              world_info.instances[instance_idx].objectID;

                // Should be able to just do a continue in this case - we'll
                // just resume processing the parent node
                if (!instanceHasVolume(instance_data))
                    continue;

                ray_o = instance_data->scale.inv() *
                    instance_data->rotation.inv().rotateVec(
                            (ray_o - instance_data->position));
                ray_d = instance_data->scale.inv() *
                    instance_data->rotation.inv().rotateVec(
                            ray_d);
                t_scale = ray_d.length();
                t_max *= t_scale;

                ray_d /= t_scale;
                inv_ray_d = Diag3x3::fromVec(ray_d).inv();
                isect_info = computeRayIsectInfo(ray_o, ray_d, inv_ray_d);

                node_buffer = current_bvh->nodes;

                new_grp_type = GroupType::BottomLevelRoot;
            } else if (parent_grp_type == GroupType::BottomLevelRoot) {
                new_grp_type = GroupType::BottomLevel;
            }

            QBVHNode new_current = node_buffer[child_node_idx];

            Vector3 dir_quant = getDirQuant(new_current, inv_ray_d);
            Vector3 origin_quant = getOriginQuant(new_current, ray_o, inv_ray_d);

            uint32_t grp_present_bits = 0;
            uint32_t tri_present_bits = 0;

            for (int i = 0; i < new_current.numChildren; ++i) {
                auto [t_near, t_far] = getNearFar(
                        new_current, i,
                        dir_quant,
                        origin_quant,
                        t_max);

                if (t_near <= t_far) {
                    // If we are in BLAS, check for possible triangles!
                    if ((new_grp_type == GroupType::BottomLevel ||
                        new_grp_type == GroupType::BottomLevelRoot) &&
                            new_current.childrenIdx[i] & 0x8000'0000) {
                        // Is this child triangles?? It is if this is the leaf
                        // of a bottom level tree
                        tri_present_bits |= (((1 << new_current.triSize[i]) - 1) << 
                                (MADRONA_BLAS_LEAF_WIDTH * i));
                    } else {
                        grp_present_bits |= (1 << i);
                    }
                }
            }

            current_grp = encodeNodeGroup(
                    child_node_idx, grp_present_bits, new_grp_type);

            if (tri_present_bits) {
                triangle_grp = encodeNodeGroup(
                        child_node_idx, tri_present_bits, GroupType::Triangles);
            } else {
                triangle_grp = invalidNodeGroup();
            }
        } else {
            triangle_grp = current_grp;
            current_grp = invalidNodeGroup();
        }

        while (getTrianglePresentBits(triangle_grp) != 0) {
            // TODO: check active mask against heuristic to exit if not enough
            // threads are working on this

            uint32_t node_tri_idx = __ffs(getTrianglePresentBits(triangle_grp)) - 1;
            uint32_t leaf_idx = node_tri_idx / MeshBVH::numTrisPerLeaf;
            uint32_t tri_idx = node_tri_idx % MeshBVH::numTrisPerLeaf;

            uint32_t glob_offset = 
                ((triangle_grp & 0xFFFF'FFFF) & ~0x8000'000);

            TriHitInfo hit_info = triangleIntersect(
                    glob_offset + leaf_idx,
                    tri_idx,
                    isect_info,
                    ray_o,
                    t_max,
                    current_bvh);

            if (hit_info.hit) {
                t_max = hit_info.tHit;

                tri_hit = hit_info;
            }
        }

        // If the current node group is empty, make sure to pop something or
        // break out of the tracing loop
        if (getPresentBits(current_grp) == 0) {
            if (stack_size == 0)
                break; // Break out of tracing loop

            // If we are breaking out of BLAS, reset all the appropriate state.
            if (getGroupType(current_grp) == GroupType::BottomLevelRoot) {
                t_max = t_max / t_scale;
                t_scale = 1.f;

                // Just restore all the trace info.
                ray_o = trace_info.rayOrigin;
                ray_d = trace_info.rayDirection;
                inv_ray_d = Diag3x3::fromVec(ray_d).inv();

                node_buffer = world_info.nodes;
            }

            printf("popping! %d\n", stack_size);
            current_grp = stack[--stack_size];
        }
    }

    if (tri_hit.hit) {
        tri_hit.tHit = t_max;

        if (bvhParams.raycastRGBD) {
            int32_t material_idx = tri_hit.bvh->getMaterialIDX(
                    tri_hit.leafMaterialIndex);

            Material *mat = &bvhParams.materials[material_idx];

            Vector3 color = {mat->color.x, mat->color.y, mat->color.z};

            if (mat->textureIdx != -1) {
                cudaTextureObject_t *tex = &bvhParams.textures[mat->textureIdx];

                float4 sampled_color = tex2D<float4>(*tex,
                    tri_hit.uv.x, tri_hit.uv.y);

                Vector3 tex_color = { sampled_color.x,
                                      sampled_color.y,
                                      sampled_color.z };

                color.x *= tex_color.x;
                color.y *= tex_color.y;
                color.z *= tex_color.z;
            }

            result.color = lighting(
                    color, tri_hit.normal, 
                    trace_info.rayDirection, 1, 1);
        }
        
        result.depth = tri_hit.tHit;
        result.hit = true;
    }

    return result;
}







#if 0
static __device__ TraceResult traceRayTLAS(
        TraceInfo trace_info,
        TraceWorldInfo world_info)
{
    static constexpr float inv_epsilon = 100000.0f;

    // Stack (needs to be declared locally due to a weird CUDA compiler bug).
    int32_t stack[32];
    int32_t stack_size = 0;
    stack[stack_size++] = 1;

    MeshBVH::HitInfo closest_hit_info = {};

    Diag3x3 inv_ray_d = {
        copysignf(trace_info.rayDirection.x == 0.f ? inv_epsilon : 
                1.f / trace_info.rayDirection.x, trace_info.rayDirection.x),
        copysignf(trace_info.rayDirection.y == 0.f ? inv_epsilon : 
                1.f / trace_info.rayDirection.y, trace_info.rayDirection.y),
        copysignf(trace_info.rayDirection.z == 0.f ? inv_epsilon : 
                1.f / trace_info.rayDirection.z, trace_info.rayDirection.z),
    };

    TraceResult result = {
        .hit = false
    };

    while (stack_size > 0) {
        int32_t node_idx = stack[--stack_size] - 1;
        QBVHNode node = world_info.nodes[node_idx];

        Vector3 dir_quant = {
            __uint_as_float((node.expX + 127) << 23) * inv_ray_d.d0,
            __uint_as_float((node.expY + 127) << 23) * inv_ray_d.d1,
            __uint_as_float((node.expZ + 127) << 23) * inv_ray_d.d2,
        };

        Vector3 origin_quant = {
            (node.minPoint.x - trace_info.rayOrigin.x) * inv_ray_d.d0,
            (node.minPoint.y - trace_info.rayOrigin.y) * inv_ray_d.d1,
            (node.minPoint.z - trace_info.rayOrigin.z) * inv_ray_d.d2,            
        };
        
        for (int i = 0; i < node.numChildren; ++i) {
            Vector3 t_near3 = {
                node.qMinX[i] * dir_quant.x + origin_quant.x,
                node.qMinY[i] * dir_quant.y + origin_quant.y,
                node.qMinZ[i] * dir_quant.z + origin_quant.z,
            };

            Vector3 t_far3 = {
                node.qMaxX[i] * dir_quant.x + origin_quant.x,
                node.qMaxY[i] * dir_quant.y + origin_quant.y,
                node.qMaxZ[i] * dir_quant.z + origin_quant.z,
            };

            float t_near = fmaxf(fminf(t_near3.x, t_far3.x), 
                                 fmaxf(fminf(t_near3.y, t_far3.y),
                                       fmaxf(fminf(t_near3.z, t_far3.z), 
                                             0.f)));

            float t_far = fminf(fmaxf(t_far3.x, t_near3.x), 
                                fminf(fmaxf(t_far3.y, t_near3.y),
                                      fminf(fmaxf(t_far3.z, t_near3.z), 
                                            trace_info.tMax)));

            if (t_near <= t_far) {
                if (node.childrenIdx[i] < 0) {
                    // This child is a leaf.
                    int32_t instance_idx = (int32_t)(-node.childrenIdx[i] - 1);

                    MeshBVH *model_bvh = bvhParams.bvhs +
                        world_info.instances[instance_idx].objectID;

                    InstanceData *instance_data =
                        &world_info.instances[instance_idx];

                    // Skip the instance if it doesn't have any scale
                    if (instance_data->scale.d0 == 0.0f &&
                        instance_data->scale.d1 == 0.0f &&
                        instance_data->scale.d2 == 0.0f) {
                        continue;
                    }

                    Vector3 txfm_ray_o = instance_data->scale.inv() *
                        instance_data->rotation.inv().rotateVec(
                            (trace_info.rayOrigin - instance_data->position));

                    Vector3 txfm_ray_d = instance_data->scale.inv() *
                        instance_data->rotation.inv().rotateVec(
                                trace_info.rayDirection);

                    float t_scale = txfm_ray_d.length();

                    txfm_ray_d /= t_scale;

                    MeshBVH::HitInfo hit_info = {};

                    bool leaf_hit = model_bvh->traceRay(
                            txfm_ray_o, txfm_ray_d, &hit_info, 
                            stack, stack_size, trace_info.tMax * t_scale);

                    if (leaf_hit) {
                        result.hit = true;

                        trace_info.tMax = hit_info.tHit / t_scale;

                        closest_hit_info = hit_info;
                        closest_hit_info.normal = 
                            instance_data->rotation.rotateVec(
                                instance_data->scale * closest_hit_info.normal);

                        closest_hit_info.normal = 
                            closest_hit_info.normal.normalize();

                        closest_hit_info.bvh = model_bvh;
                    }
                } else {
                    stack[stack_size++] = node.childrenIdx[i];
                }
            }
        }
    }

    if (result.hit) {
        if (bvhParams.raycastRGBD) {
            int32_t material_idx = 
                closest_hit_info.bvh->getMaterialIDX(closest_hit_info);

            Material *mat = &bvhParams.materials[material_idx];

            Vector3 color = {mat->color.x, mat->color.y, mat->color.z};

            if (mat->textureIdx != -1) {
                cudaTextureObject_t *tex = &bvhParams.textures[mat->textureIdx];

                float4 sampled_color = tex2D<float4>(*tex,
                    closest_hit_info.uv.x, closest_hit_info.uv.y);

                Vector3 tex_color = { sampled_color.x,
                                      sampled_color.y,
                                      sampled_color.z };

                color.x *= tex_color.x;
                color.y *= tex_color.y;
                color.z *= tex_color.z;
            }

            result.color = lighting(
                    color, closest_hit_info.normal, 
                    trace_info.rayDirection, 1, 1);
        }
        
        result.depth = trace_info.tMax;
    }

    return result;
}
#endif

static __device__ void writeRGB(uint32_t pixel_byte_offset,
                           const Vector3 &color)
{
    uint8_t *rgb_out = (uint8_t *)bvhParams.rgbOutput + pixel_byte_offset;

    *(rgb_out + 0) = (color.x) * 255;
    *(rgb_out + 1) = (color.y) * 255;
    *(rgb_out + 2) = (color.z) * 255;
    *(rgb_out + 3) = 255;
}

static __device__ void writeDepth(uint32_t pixel_byte_offset,
                             float depth)
{
    float *depth_out = (float *)
        ((uint8_t *)bvhParams.depthOutput + pixel_byte_offset);
    *depth_out = depth;
}

extern "C" __global__ void bvhRaycastEntry()
{
    uint32_t pixels_per_block = blockDim.x;

    const uint32_t total_num_views = bvhParams.internalData->numViews;

    // This is the number of views currently being processed.
    const uint32_t num_resident_views = gridDim.x;

    // This is the offset into the resident view processors that we are
    // currently in.
    const uint32_t resident_view_offset = blockIdx.x;

    uint32_t current_view_offset = resident_view_offset;

    uint32_t bytes_per_view =
        bvhParams.renderOutputResolution * bvhParams.renderOutputResolution * 4;

    uint32_t num_processed_pixels = 0;

    uint32_t pixel_x = blockIdx.y * pixels_per_block + threadIdx.x;
    uint32_t pixel_y = blockIdx.z * pixels_per_block + threadIdx.y;

#if 0
    if (pixel_x != 0 || pixel_y != 0)
        return;
#endif

    while (current_view_offset < total_num_views) {
        // While we still have views to generate, trace.
        PerspectiveCameraData *view_data = 
            &bvhParams.views[current_view_offset];

        uint32_t world_idx = (uint32_t)view_data->worldIDX;

        Vector3 ray_start = view_data->position;
        Vector3 ray_dir = calculateOutRay(view_data, pixel_x, pixel_y);

        uint32_t internal_nodes_offset = bvhParams.instanceOffsets[world_idx];

        TraceResult result = traceRay(
            TraceInfo {
                .rayOrigin = ray_start,
                .rayDirection = ray_dir,
                .tMin = bvhParams.nearSphere,
                .tMax = 10000.f,
            },
            TraceWorldInfo {
                .nodes = bvhParams.internalData->traversalNodes + 
                         internal_nodes_offset,
                .instances = bvhParams.instances + internal_nodes_offset
            }
        );

#if 0
        TraceResult result = traceRayTLAS(
            TraceInfo {
                .rayOrigin = ray_start,
                .rayDirection = ray_dir,
                .tMin = bvhParams.nearSphere,
                .tMax = 10000.f,
            },
            TraceWorldInfo {
                .nodes = bvhParams.internalData->traversalNodes + 
                         internal_nodes_offset,
                .instances = bvhParams.instances + internal_nodes_offset
            }
        );
#endif

        uint32_t linear_pixel_idx = 4 * 
            (pixel_y + pixel_x * bvhParams.renderOutputResolution);

        uint32_t global_pixel_byte_off = current_view_offset * bytes_per_view +
            linear_pixel_idx;

        if (bvhParams.raycastRGBD) {
            // Write both depth and color information
            if (result.hit) {
                writeRGB(global_pixel_byte_off, result.color);
                writeDepth(global_pixel_byte_off, result.depth);
            } else {
                writeRGB(global_pixel_byte_off, { 0.f, 0.f, 0.f });
                writeDepth(global_pixel_byte_off, 0.f);
            }
        } else {
            // Only write depth information
            if (result.hit) {
                writeDepth(global_pixel_byte_off, result.depth);
            } else {
                writeDepth(global_pixel_byte_off, 0.f);
            }
        }

        current_view_offset += num_resident_views;

        num_processed_pixels++;

        __syncwarp();
    }
}

