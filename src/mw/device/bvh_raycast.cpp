#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#if 0
#include <cuda/barrier>
#include <cuda/pipeline>
#endif

#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh.hpp>
#include <madrona/mw_gpu/host_print.hpp>

#if 0
#define LOG_RECURSE(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG_RECURSE(...)
#endif

#if 0
#define LOG_INST(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG_INST(...)
#endif

#if 1
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG(...)
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::render;

namespace smem {
// Only shared memory to be used
static constexpr size_t kBufSize = 9 * 1024;
extern __shared__ uint8_t buffer[kBufSize];
}

extern "C" __constant__ BVHParams bvhParams;

inline Vector3 lighting(Vector3 diffuse,
                        Vector3 normal,
                        Vector3 light_dir)
{
    constexpr float ambient = 0.4;
    return (fminf(fmaxf(normal.dot(light_dir),0.f)+ambient, 1.0f)) * diffuse;
}

inline Vector3 lightingShadow(
        Vector3 diffuse,
        Vector3 normal)
{
    constexpr float ambient = 0.4;
    return ambient * diffuse;
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
    const auto viewport_width = viewport_height * view_data->yScale / view_data->xScale;
    const auto forward = look_at.normalize();

    auto u = rot.inv().rotateVec({1, 0, 0});
    auto v = cross(forward, u).normalize();

    auto horizontal = u * viewport_width;
    auto vertical = v * viewport_height;

    auto lower_left_corner = ray_start - horizontal / 2 - vertical / 2 + forward;
  
    float pixel_u = ((float)pixel_x + 0.5f) / (float)bvhParams.renderOutputWidth;
    float pixel_v = ((float)pixel_y + 0.5f) / (float)bvhParams.renderOutputHeight;

    Vector3 ray_dir = lower_left_corner + pixel_u * horizontal + 
        pixel_v * vertical - ray_start;
    ray_dir = ray_dir.normalize();

    return ray_dir;
}

struct TraceResult {
    bool hit;
    Vector3 color;
    Vector3 normal;
    float metalness;
    float roughness;
    float depth;
};

struct TraceInfo {
    Vector3 rayOrigin;
    Vector3 rayDirection;
    float tMin;
    float tMax;
    bool dOnly; // Depth only
};

struct TraceWorldInfo {
    QBVHNode *nodes;
    InstanceData *instances;

    LightDesc *lights;
    uint32_t numLights;
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
    uint8_t present_bits = (uint8_t)((1ull << children_count) - 1);
    return encodeNodeGroup(0, present_bits, GroupType::TopLevel);
}

static GroupType getGroupType(NodeGroup grp)
{
    return (GroupType)((grp >> 61ull) & 0b111);
}

static NodeGroup unsetPresentBit(NodeGroup grp, uint32_t idx)
{
    grp &= ~(1ull << (idx + 32ull));
    return grp;
}

static uint32_t getPresentBits(NodeGroup grp)
{
    return (uint32_t)((grp >> 32) & 0xFF);
}

static uint32_t getTrianglePresentBits(NodeGroup grp)
{
    // 24 bits used for triangle presence
    return (uint32_t)((grp >> 32) & ((1ull << 24) - 1));
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

    int32_t instanceIdx;

    Vector3 normal;
    Vector2 uv;

    uint32_t leafMaterialIndex;

    MeshBVH *bvh;

    InstanceData *instance;
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

    const float a_kz = A[kz];
    const float a_kx = A[kx];
    const float a_ky = A[ky];
    const float b_kz = B[kz];
    const float b_kx = B[kx];
    const float b_ky = B[ky];
    const float c_kz = C[kz];
    const float c_kx = C[kx];
    const float c_ky = C[ky];

    // Perform shear and scale of vertices
    const float Ax = fmaf(-Sx, a_kz, a_kx);
    const float Ay = fmaf(-Sy, a_kz, a_ky);
    const float Bx = fmaf(-Sx, b_kz, b_kx);
    const float By = fmaf(-Sy, b_kz, b_ky);
    const float Cx = fmaf(-Sx, c_kz, c_kx);
    const float Cy = fmaf(-Sy, c_kz, c_ky);

    // calculate scaled barycentric coordinates
    float U = fmaf(Cx, By, - Cy * Bx);
    float V = fmaf(Ax, Cy, - Ay * Cx);
    float W = fmaf(Bx, Ay, - By * Ax);

    static constexpr float eps_tol = 1e-7;

    if (U > -eps_tol && U < eps_tol)
        U = 0.f;
    if (V > -eps_tol && V < eps_tol)
        V = 0.f;
    if (W > -eps_tol && W < eps_tol)
        W = 0.f;

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
    const float Az = Sz * a_kz;
    const float Bz = Sz * b_kz;
    const float Cz = Sz * c_kz;
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

#if 0
static void prefetchNode(uint32_t node_idx,
                         QBVHNode *node_buffer,
                         QBVHNode *result_smem,
                         cuda::pipeline<cuda::thread_scope_thread> &pipe)
{
    pipe.producer_acquire();
    {
        cuda::memcpy_async(result_smem, 
                           &node_buffer[node_idx],
                           sizeof(QBVHNode),
                           pipe);
    }
    pipe.producer_commit();
}
#endif

#if 0
static QBVHNode readNode(QBVHNode *node_smem,
                         cuda::pipeline<cuda::thread_scope_thread> &pipe)
{
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    QBVHNode read_node = *node_smem;
    pipe.consumer_release();

    return read_node;
}
#endif

static Vector3 hexToRgb(uint32_t hex)
{
    // Extract each color component and normalize to [0, 1] range
    float r = ((hex >> 16) & 0xFF) / 255.0f;
    float g = ((hex >> 8) & 0xFF) / 255.0f;
    float b = (hex & 0xFF) / 255.0f;

    return {r, g, b};
}

static __device__ TraceResult traceRay(
    TraceInfo trace_info,
    TraceWorldInfo world_info)
{
#if 0
    uint8_t *smem_scratch = nullptr;
    {
        uint32_t linear_tid =
            threadIdx.x + 
            threadIdx.y * blockDim.x + 
            threadIdx.z * blockDim.x * blockDim.y;
        uint32_t bytes_per_thread = smem::kBufSize /
            (blockDim.x * blockDim.y * blockDim.z);
        smem_scratch = &smem::buffer[linear_tid * bytes_per_thread];
    }
#endif

#if 0
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
#endif

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
    int32_t instance_idx = -1;
    float t_scale = 1.f;

    QBVHNode *node_buffer = world_info.nodes;

    TriHitInfo tri_hit = {
        .hit = false,
        .instanceIdx = -1
    };

    for (;;) {
        if (GroupType parent_grp_type = getGroupType(current_grp); 
                parent_grp_type != GroupType::Triangles &&
                getPresentBits(current_grp) != 0) {
            // TODO: Make sure to have the node traversal order
            // sorted according to the ray direction.
            // NOTE: This should never underflow
            uint32_t child_idx = __ffs(getPresentBits(current_grp)) - 1;
            current_grp = unsetPresentBit(current_grp, child_idx);

            // Bottom level root needs to be pushed no matter what
            // so that when popping, we can restore the state of the
            // ray data.
            if (getPresentBits(current_grp) != 0 ||
                    (parent_grp_type == GroupType::BottomLevelRoot)) {
                stack[stack_size++] = current_grp;
            }

            // Intersect with the children of the child to get a new node group
            // and calculate the present bits according to which were
            // intersected
            uint32_t child_node_idx =
                node_buffer[current_grp & 0xFFFF'FFFF].childrenIdx[child_idx];
            bool child_is_leaf = (child_node_idx & 0x8000'0000);
            child_node_idx = child_node_idx & (~0x8000'0000);

            GroupType new_grp_type = GroupType::TopLevel;

            if (parent_grp_type == GroupType::TopLevel &&
                child_is_leaf) {
                // Need to compute new ray o/d/etc...
                instance_idx = (int32_t)(child_node_idx & ~0x8000'0000);
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

                // mat_id_override = instance_data->matID;
                // instance_idx = instance_idx;

                ray_d /= t_scale;
                inv_ray_d = Diag3x3::fromVec(ray_d).inv();
                isect_info = computeRayIsectInfo(ray_o, ray_d, inv_ray_d);

                node_buffer = current_bvh->nodes;

                new_grp_type = GroupType::BottomLevelRoot;

                // Set this to 0 so that when we read from node_buffer,
                // we're reading the root node of the mesh BVH.
                child_node_idx = 0;
            } else if (parent_grp_type == GroupType::BottomLevelRoot ||
                       parent_grp_type == GroupType::BottomLevel) {
                new_grp_type = GroupType::BottomLevel;
            }

            QBVHNode new_current = node_buffer[child_node_idx];

            Vector3 dir_quant = getDirQuant(new_current, inv_ray_d);
            Vector3 origin_quant = getOriginQuant(new_current, ray_o, inv_ray_d);

            uint32_t grp_present_bits = 0;
            uint32_t tri_present_bits = 0;

#pragma unroll
            for (int i = 0; i < MADRONA_BVH_WIDTH; ++i) {
                if (new_current.childrenIdx[i] != 0xFFFF'FFFF) {
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
        }

        if (getTrianglePresentBits(triangle_grp) != 0) {
            QBVHNode parent = node_buffer[triangle_grp & 0xFFFF'FFFF];

            while (getTrianglePresentBits(triangle_grp) != 0) {
                // TODO: check active mask against heuristic to exit if not enough
                // threads are working on this
                uint32_t local_node_tri_idx = 
                    __ffs(getTrianglePresentBits(triangle_grp)) - 1;

                uint32_t local_leaf_idx = 
                    local_node_tri_idx / MeshBVH::numTrisPerLeaf;
                uint32_t tri_idx = 
                    local_node_tri_idx % MeshBVH::numTrisPerLeaf;

                uint32_t glob_leaf_idx =
                    parent.childrenIdx[local_leaf_idx] & (~0x8000'0000);

                TriHitInfo hit_info = triangleIntersect(
                        glob_leaf_idx,
                        tri_idx,
                        isect_info,
                        ray_o,
                        t_max,
                        current_bvh);

                if (hit_info.hit) {
                    t_max = hit_info.tHit;

                    tri_hit = hit_info;
                    tri_hit.instanceIdx = instance_idx;
                }

                triangle_grp = unsetPresentBit(triangle_grp, local_node_tri_idx);
            }
        }

        // If the current node group is empty, make sure to pop something or
        // break out of the tracing loop
        if (getPresentBits(current_grp) == 0) {
            if (stack_size == 0) {
                t_max = t_max / t_scale;
                t_scale = 1.f;

                break;
            } else {
                if (getGroupType(current_grp) == GroupType::BottomLevelRoot) {
                    t_max = t_max / t_scale;
                    t_scale = 1.f;

                    // Just restore all the trace info.
                    ray_o = trace_info.rayOrigin;
                    ray_d = trace_info.rayDirection;
                    inv_ray_d = Diag3x3::fromVec(ray_d).inv();
                    instance_idx = -1;

                    node_buffer = world_info.nodes;
                }

                current_grp = stack[--stack_size];
            }
        }
    }

    if (tri_hit.hit) {
        tri_hit.tHit = t_max;

        if (bvhParams.raycastRGBD && (!trace_info.dOnly)) {
            InstanceData *instance = world_info.instances + tri_hit.instanceIdx;

            int32_t override_mat_id = instance->matID;
            int32_t material_idx = override_mat_id;

            if (override_mat_id == MaterialOverride::UseDefaultMaterial) {
                material_idx = tri_hit.bvh->getMaterialIDX(
                        tri_hit.leafMaterialIndex);
            }

            Vector3 color = { 1.f, 1.f, 1.f };

            if (override_mat_id == MaterialOverride::UseOverrideColor) {
                color = hexToRgb(instance->color);
            } else if (material_idx != -1) {
                Material *mat = &bvhParams.materials[material_idx];

                if (mat->textureIdx != -1) {
                    cudaTextureObject_t *tex = &bvhParams.textures[mat->textureIdx];

                    float4 sampled_color = tex2D<float4>(*tex,
                            tri_hit.uv.x, 1.f - tri_hit.uv.y);

                    Vector3 tex_color = { sampled_color.x,
                        sampled_color.y,
                        sampled_color.z };

                    color.x = tex_color.x * mat->color.x;
                    color.y = tex_color.y * mat->color.y;
                    color.z = tex_color.z * mat->color.z;
                } else {
                    color.x = mat->color.x;
                    color.y = mat->color.y;
                    color.z = mat->color.z;
                }

                result.metalness = mat->metalness;
                result.roughness = mat->roughness;
            }

            result.color = color;
            result.normal = instance->rotation.rotateVec(tri_hit.normal);
        }
        
        result.depth = tri_hit.tHit;
        result.hit = true;
    }

    return result;
}

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

struct FragmentResult {
    bool hit;
    Vector3 color;
    Vector3 normal;
    float depth;
};

static __device__ FragmentResult computeFragment(
    TraceInfo trace_info,
    TraceWorldInfo world_info)
{
    // Direct hit first
    TraceResult first_hit = traceRay(trace_info, world_info);

    __syncwarp();

    if (!bvhParams.raycastRGBD) {
        if (first_hit.hit) {
            return FragmentResult {
                .hit = true,
                .depth = first_hit.depth
            };
        }
    } else if (first_hit.hit) {
        // Now we need to test against lights.
        Vector3 hit_pos = trace_info.rayOrigin +
                          first_hit.depth * trace_info.rayDirection;

        Vector3 acc_color = Vector3{ 0.f, 0.f, 0.f };

        float light_contrib = 0.f;

        for (int i = 0; i < world_info.numLights; ++i) {
            LightDesc desc = world_info.lights[i];

            float cutoff = -1.f;

            Vector3 light_dir = -desc.direction;
            if (desc.type == LightDesc::Type::Spotlight) {
                light_dir = (desc.position - hit_pos).normalize();
                cutoff = desc.cutoff;
            }

            if (cutoff != -1.f && desc.type == LightDesc::Type::Spotlight) {
                // Dot the vector going from point to light with the direction
                // of the light.
                float d = (-light_dir).dot(desc.direction);
                d /= (light_dir.length() * desc.direction.length());
                float angle = acosf(d);

                // This pixel isn't affected by this light.
                if (std::abs(angle) > std::abs(desc.cutoff)) {
                    continue;
                }
            }

            // Make sure the surface is actually pointing to the light
            // when casting shadow.
            if (desc.castShadow) {
                if (light_dir.dot(first_hit.normal) > 0.f) {
                    // TODO: Definitely do some sort of ray fetching because there will
                    // be threads doing nothing potentially.
                    TraceResult shadow_hit = traceRay(
                            TraceInfo {
                            .rayOrigin = hit_pos,
                            .rayDirection = light_dir,
                            .tMin = 0.000001f,
                            .tMax = 10000.f,
                            .dOnly = true
                            }, world_info);

                    if (!shadow_hit.hit) {
                        light_contrib += fminf(fmaxf(first_hit.normal.dot(light_dir), 0.f), 1.f);
                    }
                }
            } else {
                light_contrib += fminf(fmaxf(first_hit.normal.dot(light_dir), 0.f), 1.f);
            }
        }

        acc_color = fmaxf(0.2, light_contrib) * first_hit.color;
        acc_color.x = fminf(1.f, acc_color.x);
        acc_color.y = fminf(1.f, acc_color.y);
        acc_color.z = fminf(1.f, acc_color.z);

        // If we are still here, just do normal lighting calculation.
        return FragmentResult {
            .hit = true,
            .color = acc_color,
            .normal = first_hit.normal,
            .depth = first_hit.depth
        };
    }

    return FragmentResult {
        .hit = false
    };
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
        bvhParams.renderOutputWidth * bvhParams.renderOutputHeight * 4;

    uint32_t num_processed_pixels = 0;

    uint32_t pixel_x = blockIdx.y * pixels_per_block + threadIdx.x;
    uint32_t pixel_y = blockIdx.z * pixels_per_block + threadIdx.y;

    while (current_view_offset < total_num_views) {
        // While we still have views to generate, trace.
        PerspectiveCameraData *view_data = 
            &bvhParams.views[current_view_offset];

        uint32_t world_idx = (uint32_t)view_data->worldIDX;

        Vector3 ray_start = view_data->position;
        Vector3 ray_dir = calculateOutRay(view_data, pixel_x, pixel_y);

        uint32_t internal_nodes_offset = bvhParams.instanceOffsets[world_idx];



        // This does both the tracing / lighting, etc... just like a fragment
        // shader does in GLSL.
        FragmentResult result = computeFragment(
            TraceInfo {
                .rayOrigin = ray_start,
                .rayDirection = ray_dir,
                .tMin = bvhParams.nearSphere,
                .tMax = 10000.f,
                .dOnly = false
            },
            TraceWorldInfo {
                .nodes = bvhParams.internalData->traversalNodes + 
                         internal_nodes_offset,
                .instances = bvhParams.instances + internal_nodes_offset,
                .lights = &bvhParams.lights[bvhParams.lightOffsets[world_idx]],
                .numLights = (uint32_t)bvhParams.lightCounts[world_idx]
            }
        );




        uint32_t linear_pixel_idx = 4 * 
            (pixel_x + pixel_y * bvhParams.renderOutputWidth);

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

